"""Textual TUI: an IRC-style channel driving a `DebateEngine`.

Engine work runs in a Textual worker on the app's own event loop, so the input
stays live during a run and `/pause` (or Esc) reaches the engine, which stops
at the next turn boundary.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from privy_council.engine.protocol import DebateEngine
from privy_council.models.session import (
    ErrorEvent,
    Event,
    MessageEvent,
    RosterChanged,
    RoundStarted,
    SessionState,
    SessionStateChanged,
    StateUpdated,
    TextDelta,
    TurnStarted,
)
from privy_council.tui.commands import (
    AS_USAGE,
    COMMANDS,
    INPUT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    Command,
    Judgment,
    export_command,
    flaws_text,
    parse,
    parse_count,
    split_as,
    split_ask,
    state_text,
)
from privy_council.tui.widgets import (
    LOCAL_ONLY_BADGE,
    ORANGE,
    RED,
    STATE_LABELS,
    ChatInput,
    ChatLog,
    NickList,
    StatePanel,
    TopBar,
    TypingArea,
    budget_text,
)

EngineJob = Callable[[], Awaitable[Any]]

CSS = """
TopBar { height: 1; background: $boost; padding: 0 1; }
#main { height: 1fr; }
#chat-col { width: 1fr; }
ChatLog { height: 1fr; padding: 0 1; }
TypingArea { height: auto; padding: 0 1; border-top: dashed $panel; display: none; }
TypingArea.visible { display: block; }
#unread { height: 1; padding: 0 1; background: $boost; color: #d19a66; display: none; }
#unread.visible { display: block; }
StatePanel { width: 42; border-left: solid $panel; padding: 0 1; display: none; }
StatePanel.visible { display: block; }
NickList { width: 24; border-left: solid $panel; padding: 0 1; }
Input { dock: bottom; }
"""


def judgment_of(event: StateUpdated) -> Judgment | None:
    """Return the moderator's judgment carried by a state update (None before M7)."""
    if event.consensus is None:
        return None
    return Judgment(event.consensus, event.stalled_rounds, event.decision_open, event.new_arguments)


class CouncilApp(App[None]):
    """The council channel."""

    CSS = CSS
    TITLE = "council"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "pause", "Pause", show=False),
        Binding("f2", "toggle_state", "Debate state", show=False),
        # Priority: the input has focus and binds End itself (see action_chat_end).
        Binding("end", "chat_end", "Newest messages", show=False, priority=True),
        Binding("pageup", "chat_page(-1)", "Scroll up", show=False),
        Binding("pagedown", "chat_page(1)", "Scroll down", show=False),
    ]

    def __init__(
        self,
        engine: DebateEngine,
        on_exit: Callable[[], Coroutine[Any, Any, None]] | None = None,
        attach: Sequence[Path] = (),
    ) -> None:
        """Wrap an engine; `on_exit` releases backend resources on unmount.

        `attach`: files from `council new --attach`, attached before the opening.
        """
        super().__init__()
        self.engine = engine
        self.attach = list(attach)
        self.on_exit_hook = on_exit
        self.fallback: set[str] = set()
        self._confirm_run: int | None = None
        self._confirm_quit = False

    def compose(self) -> ComposeResult:
        """Lay out bar, chat, state pane, nick list and input."""
        engine = self.engine
        yield TopBar(id="bar")
        with Horizontal(id="main"):
            with Vertical(id="chat-col"):
                yield ChatLog(wrap=True, markup=False, min_width=20, id="chat")
                yield TypingArea(self.engine.nick, self.color_of, id="typing")
                yield Static(id="unread")
            yield StatePanel(id="state")
            yield NickList([engine.moderator, *engine.members.values()], id="nicks")
        yield ChatInput(placeholder=INPUT_PLACEHOLDER)

    async def on_mount(self) -> None:
        """Subscribe to the engine and run the opening."""
        self.engine.subscribe(self.on_engine_event)
        self.query_one(NickList).refresh_list()
        self.refresh_bar()
        self.query_one(ChatInput).focus()
        if self.engine.resumed and not self.engine.awaiting_topic:
            self.replay_history()
            return
        if self.engine.awaiting_topic:
            self.chat.notice(
                "Type your question to start (a multi-line paste stays one message)", ORANGE
            )
            if self.attach:
                self.start(self.attach_files)
            return
        self.cmd_log()
        self.start(self.open_session)

    async def attach_files(self) -> None:
        """Attach the `--attach` files."""
        for path in self.attach:
            await self.engine.attach(str(path))

    async def open_session(self) -> None:
        """Attach the `--attach` files, then run the opening."""
        await self.attach_files()
        await self.engine.step()

    def replay_history(self) -> None:
        """Show a resumed session's past events, then where it stands (no step runs)."""
        for event in self.engine.history:
            self.on_engine_event(event)
        self.fallback.clear()  # the Claude limit is re-checked from scratch after a resume
        self.query_one(NickList).reset(keep=set())
        self.chat.follow()
        self.cmd_log()
        self.engine.announce_resume()

    async def on_unmount(self) -> None:
        """Release backend resources."""
        if self.on_exit_hook is not None:
            await self.on_exit_hook()

    @property
    def chat(self) -> ChatLog:
        """Return the chat log."""
        return self.query_one(ChatLog)

    # --- Engine events -------------------------------------------------------

    def on_engine_event(self, event: Event) -> None:
        """Mirror an engine event in the widgets (same loop: direct updates are safe)."""
        self.chat.add_event(event, self.engine.nick, self.color_of)
        nicks = self.query_one(NickList)
        typing = self.query_one(TypingArea)
        self.update_placeholder(event)
        if isinstance(event, TextDelta):
            typing.add(event.speaker, event.text, event.restart)
            if event.restart:
                nicks.set_status(event.speaker, "typing…")
        elif isinstance(event, ErrorEvent):
            typing.clear()
        if isinstance(event, TurnStarted):
            nicks.set_status(event.speaker, "thinking…")
        elif isinstance(event, MessageEvent) and event.message.speaker in nicks.status:
            speaker = event.message.speaker
            typing.done(speaker)
            if event.message.fallback:
                self.fallback.add(speaker)
            nicks.set_status(speaker, "fallback" if speaker in self.fallback else "idle")
        elif isinstance(event, StateUpdated):
            self.query_one(StatePanel).show_state(
                state_text(event.state, judgment_of(event), self.engine.nick)
            )
        elif isinstance(event, RosterChanged):
            engine = self.engine
            nicks.set_personas([engine.moderator, *engine.members.values()])
        elif isinstance(event, SessionStateChanged):
            typing.clear()
            if event.state is not SessionState.ROUND_RUNNING:
                nicks.reset(keep=self.fallback)
            self.refresh_bar()

    def update_placeholder(self, event: Event) -> None:
        """Remind how to resume while the Chair's question is open (until the next round)."""
        if isinstance(event, MessageEvent) and event.message.to_user:
            self.query_one(ChatInput).placeholder = QUESTION_PLACEHOLDER
        elif isinstance(event, RoundStarted):
            self.query_one(ChatInput).placeholder = INPUT_PLACEHOLDER

    def color_of(self, slug: str) -> str:
        """Return a speaker's nick color."""
        persona = self.engine.persona_of(slug)
        return persona.meta.color if persona else "white"

    def refresh_bar(self) -> None:
        """Redraw the top bar from the engine state and the usage ledger."""
        engine = self.engine
        rt = engine.meta.runtime
        state = STATE_LABELS[engine.state]
        if engine.busy and engine.state is SessionState.PAUSED:
            state = "running"
        self.query_one(TopBar).show(
            channel=engine.council.name,
            topic=engine.meta.topic or "(type your question)",
            round_text=f"round {rt.round}/{engine.max_rounds} · {engine.meta.language}",
            state=state,
            # No Claude budget to show in local-only mode (M15): the badge instead.
            budget=LOCAL_ONLY_BADGE
            if engine.meta.local_only
            else budget_text(engine.usage(), engine.settings.claude.session_soft_cap_calls),
        )

    # --- Input ---------------------------------------------------------------

    async def on_input_submitted(self, message: ChatInput.Submitted) -> None:
        """Parse and run one input line (multi-line pastes arrive with newline marks)."""
        message.input.value = ""
        self.chat.follow()
        cmd = parse(ChatInput.restore(message.value))
        if cmd is not None:
            await self.dispatch(cmd)

    async def dispatch(self, cmd: Command) -> None:
        """Run a slash command."""
        handlers: dict[str, Callable[[str], None]] = {
            "step": lambda _: self.start(self.engine.step),
            "run": self.cmd_run,
            "more": self.cmd_more,
            "pause": lambda _: self.action_pause(),
            "say": self.cmd_say,
            "as": self.cmd_as,
            "ask": self.cmd_ask,
            "summon": self.engine.summon,
            "kick": self.engine.kick,
            "attach": lambda arg: self.start(lambda: self.engine.attach(arg)),
            "research": lambda arg: self.start(lambda: self.engine.research(arg or None)),
            "vote": lambda _: self.start(self.engine.vote),
            "summary": lambda _: self.start(self.engine.summary),
            "state": lambda _: self.action_toggle_state(),
            "flaws": lambda _: self.chat.notice(
                flaws_text(self.engine.meta.runtime.flaws, self.engine.nick)
            ),
            "export": lambda arg: self.chat.notice(export_command(self.engine, arg)),
            "lang": self.cmd_lang,
            "budget": lambda _: self.cmd_budget(),
            "log": lambda _: self.cmd_log(),
            "close": lambda _: self.start(self.engine.close),
            "keep": lambda arg: self.start(lambda: self.engine.keep(arg)),
            "forget": self.engine.forget,
            "quit": lambda _: self.cmd_quit(),
            "help": lambda _: self.cmd_help(),
        }
        handler = handlers.get(cmd.name)
        if handler is None:
            self.chat.notice(f"unknown command /{cmd.name}; /help lists commands", RED)
        else:
            handler(cmd.arg)

    def start(self, job: EngineJob) -> None:
        """Run an engine coroutine in a worker, unless one is already running."""
        if self.engine.busy or any(w.group == "engine" and w.is_running for w in self.workers):
            self.chat.notice("Busy: /pause (or Esc) to stop at the next turn")
            return

        async def work() -> None:
            """Run the job, then clear "thinking…" statuses and refresh the bar."""
            try:
                await job()
            finally:
                # Not every job ends with the speaker's message or a state change
                # (/research posts a notice and facts), so a status could stick.
                self.query_one(NickList).reset(keep=self.fallback)
                self.refresh_bar()

        self.run_worker(work(), group="engine", exclusive=True)
        self.refresh_bar()

    def cmd_lang(self, arg: str) -> None:
        """`/lang [en|fr]`: show or change the debate language, then refresh the bar."""
        self.engine.set_language(arg)
        self.refresh_bar()

    def cmd_run(self, arg: str) -> None:
        """`/run N`: show the Claude estimate and ask to confirm if it crosses a cap."""
        rounds = parse_count(arg)
        if rounds is None:
            self.chat.notice("usage: /run N (N ≥ 1)", RED)
            return
        estimate = self.engine.estimate_claude_calls(rounds)
        warnings = self.engine.budget_warnings(estimate)
        if warnings and self._confirm_run != rounds:
            for warning in warnings:
                self.chat.notice(f"! budget: {warning}", ORANGE)
            self.chat.notice(f"~{estimate} Claude calls. Type /run {rounds} again to confirm.")
            self._confirm_run = rounds
            return
        self._confirm_run = None
        self.chat.notice(f"Running up to {rounds} round(s), ~{estimate} Claude calls; Esc pauses")
        self.start(lambda: self.engine.run(rounds))

    def cmd_more(self, arg: str) -> None:
        """`/more N`: add N rounds past max_rounds, then refresh the round counter."""
        rounds = parse_count(arg)
        if rounds is None:
            self.chat.notice("usage: /more N (N ≥ 1)", RED)
            return
        self.engine.more(rounds)
        self.refresh_bar()

    def cmd_say(self, arg: str) -> None:
        """`/say text` or plain text; the first line sets the question if none was given."""
        if arg and self.engine.awaiting_topic:
            if self.engine.set_topic(arg):
                self.refresh_bar()
                self.cmd_log()
                self.start(self.engine.step)
        elif arg:
            self.engine.say(arg)
        else:
            self.chat.notice("usage: /say <text>", RED)

    def cmd_as(self, arg: str) -> None:
        """`/as name text`: speak as another person."""
        said = split_as(arg)
        if said is None:
            self.chat.notice(AS_USAGE, RED)
        else:
            self.engine.say(*said)

    def cmd_ask(self, arg: str) -> None:
        """`/ask @nick question`."""
        parts = split_ask(arg)
        if parts is None:
            self.chat.notice("usage: /ask @nick <question>", RED)
            return
        self.start(lambda: self.engine.ask(*parts))

    def cmd_quit(self) -> None:
        """`/quit`: exit, asking once if shared facts still await `/keep`."""
        if self.engine.memory_review_pending and not self._confirm_quit:
            self._confirm_quit = True
            self.chat.notice(
                "Shared facts await /keep; /quit again to leave without saving them", ORANGE
            )
            return
        self.exit()

    def cmd_budget(self) -> None:
        """`/budget`: Claude usage for the session and today."""
        report = self.engine.usage()
        if report is None:
            self.chat.notice("no usage ledger configured")
            return
        caps = self.engine.settings.claude
        self.chat.notice(
            f"session: {report.session.calls}/{caps.session_soft_cap_calls} Claude calls, "
            f"{report.session.input_tokens:,} in / {report.session.output_tokens:,} out tokens; "
            f"today: {report.today.calls}/{caps.daily_soft_cap_calls}"
        )

    def cmd_log(self) -> None:
        """`/log`: where the live transcript is written."""
        path = self.engine.live_log
        self.chat.notice(f"Live log: {path}" if path else "This session is not saved")

    def cmd_help(self) -> None:
        """`/help`: list commands and keys."""
        for name, text in COMMANDS.items():
            self.chat.notice(f"/{name:<7} {text}")
        self.chat.notice(
            "keys: Esc pause · F2 debate state · PgUp/PgDn scroll · End newest · Ctrl+Q quit"
        )
        self.chat.notice(
            "copy text: drag over the chat, then Ctrl+C (Shift+drag: terminal selection)"
        )
        self.chat.notice("multi-line paste is kept whole: line breaks show as ⏎ until you send")

    def action_pause(self) -> None:
        """Pause at the next turn boundary."""
        if self.engine.busy:
            self.engine.request_pause()
        else:
            self.chat.notice("Nothing is running")

    def action_toggle_state(self) -> None:
        """Show or hide the debate-state pane."""
        self.query_one(StatePanel).toggle_class("visible")

    def on_chat_log_unread(self, message: ChatLog.Unread) -> None:
        """Show or hide the "new messages" line under the chat."""
        line = self.query_one("#unread", Static)
        plural = "s" if message.count > 1 else ""
        line.update(f"↓ {message.count} new message{plural} (End to jump)")
        line.set_class(message.count > 0, "visible")

    def action_chat_end(self) -> None:
        """End: jump to the newest messages, or move the cursor to the line end if there."""
        if self.chat.following:
            self.query_one(ChatInput).action_end()
        else:
            self.chat.follow()

    def action_chat_page(self, direction: int) -> None:
        """PgUp/PgDn: scroll the chat while the input keeps focus."""
        if direction < 0:
            self.chat.scroll_page_up(animate=False)
        else:
            self.chat.scroll_page_down(animate=False)
