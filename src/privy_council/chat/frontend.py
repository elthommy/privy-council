"""The chat front end, platform-free: permissions, commands, posts, question waits.

A third front end next to the TUI and the REPL: it subscribes to engine events and
calls engine methods; `engine/` imports nothing from here. The platform adapter
(`discord_bot.py`) only moves messages through `ChatTransport`.

Rules from the M16 security review: chat text never reaches the REPL/TUI command
parser (commands are this module's own table, without /attach, /export, /research,
memory or roster commands); the owner alone starts debates, and runs and closes
them unless they hand the open one over to a friend on the list (`/handover`: that
debate only, never saved, so a restart or `/resume` takes it back); people are
known by user ID, never by name; posts never ping anyone (the adapter).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Protocol

from privy_council.chat.names import NameBook
from privy_council.chat.render import Post, RoleOf, render_event
from privy_council.chat.store import ChatThread, ThreadStore
from privy_council.config import DiscordConfig
from privy_council.engine.notices import NOTICE_QUESTIONS_PAUSE
from privy_council.engine.protocol import DebateEngine
from privy_council.models.persona import Role
from privy_council.models.session import Event, MessageEvent, Phase, RoundStarted, TurnStarted
from privy_council.tui.commands import flaws_text, parse_count

log = logging.getLogger(__name__)

# A friend the owner hands the open debate over to may run these too.
HANDOVER_COMMANDS = {"run", "pause", "vote", "close", "offline"}
OWNER_COMMANDS = {"ask", "resume", "handover", "takeback"} | HANDOVER_COMMANDS
OPEN_COMMANDS = {"status", "flaws"}
NOT_ALLOWED = "Only the council's owner can do that."
NOT_HANDED = "Only the owner, or a friend they hand the debate over to, can do that."
HANDED = (
    "**The owner handed the floor to @{name}**: they can /run, /pause, /vote, /close and "
    "/offline this debate."
)
TAKEN_BACK = "**The owner took the floor back.**"
NOT_LISTED = "You're not on this council's list."
NO_DEBATE = "No debate is open: the owner starts one with /ask."
BUSY = "The council is busy; wait for this step to end, or /pause."
ONE_AT_A_TIME = "A debate is already open: /close it first (one debate at a time)."
OFFLINE = "**Council offline.** The session is saved; the owner can /resume it later."
WAITING = "Waiting up to {minutes} min for {names} to answer."
NO_ANSWER = "No answer from {names}: the debate goes on with what it knows."
LINE_LIMIT = (
    "{name}: {count} lines this round is the limit; the rest is ignored until the next round."
)
TYPED_COMMAND = (
    "{command} arrived as a plain message, so nothing happened: type it, then pick it from "
    "Discord's command list."
)
_QPAUSE = NOTICE_QUESTIONS_PAUSE.split("{", 1)[0]
# A line that is one of the commands typed as text ("/offline", "/run 2").
_TYPED = re.compile(r"^/(?P<name>[a-z]+)(?:\s|$)")
# Discord's raw mentions in a person's line: <@123>, <@!123>, <@&role>, <#channel>.
_MENTION = re.compile(r"<(?P<kind>@!?|@&|#)(?P<id>\d+)>")
SOMEONE = "@someone"
_ASKED = re.compile(r"^@([^\W_][\w-]*)", re.MULTILINE)

NewEngine = Callable[[str, str], DebateEngine]  # (council, question) -> engine
ResumeEngine = Callable[[], DebateEngine | None]  # the latest unfinished chat session


POST_LIMIT = 2000  # Discord's message size; posts are split under it
MAX_CHOICES = 25  # Discord's limit for a slash-command option
MAX_CHOICE_CHARS = 100


def council_choices(councils: Mapping[str, str]) -> list[tuple[str, str]]:
    """Return (label, council name) pairs for the `/ask` dropdown, sorted by name.

    The label is "name: description", cut to the platform's 100 characters.
    """
    out: list[tuple[str, str]] = []
    for name in sorted(councils)[:MAX_CHOICES]:
        label = f"{name}: {councils[name]}" if councils[name] else name
        if len(label) > MAX_CHOICE_CHARS:
            label = label[: MAX_CHOICE_CHARS - 1].rstrip() + "…"
        out.append((label, name))
    return out


class ChatSetupError(Exception):
    """A debate can't start (unknown council, bad config); the message is for the owner."""


class ChatTransport(Protocol):
    """What the platform adapter provides."""

    async def open_thread(self, title: str, opening: str) -> int:
        """Create a thread (or forum post) for one question, starting with `opening`."""
        ...

    async def post(self, thread: int, post: Post) -> None:
        """Send one message, as a persona (`post.persona`) or as the bot."""
        ...

    async def typing(self, thread: int, on: bool) -> None:
        """Start or stop the "is typing" indicator in a thread."""
        ...


class ChatFrontend:
    """Runs one debate at a time for the owner and the friends on the list."""

    def __init__(
        self,
        config: DiscordConfig,
        transport: ChatTransport,
        threads: ThreadStore,
        *,
        new_engine: NewEngine,
        resume_engine: ResumeEngine,
    ) -> None:
        """Wire the front end; nothing is sent until `start()`."""
        self.config = config
        self.transport = transport
        self.threads = threads
        self.new_engine = new_engine
        self.resume_engine = resume_engine
        self.engine: DebateEngine | None = None
        self.thread: int | None = None
        self.names = NameBook(())
        self.stopped = asyncio.Event()  # set by /offline: the adapter disconnects
        self._outbox: asyncio.Queue[tuple[int, Post | bool]] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._asked: set[int] = set()
        self._waiting: dict[int, asyncio.Event] = {}
        self._lines: dict[int, int] = {}
        self._hosts: set[int] = set()  # friends handed the open debate (never saved)
        self._offline = False

    # --- lifecycle ------------------------------------------------------------------

    async def start(self) -> None:
        """Start sending queued posts (call once the platform is connected)."""
        if self._sender is None:
            self._sender = asyncio.create_task(self._send_loop())

    async def drain(self) -> None:
        """Wait until every queued post is sent, and the running step if any."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        await self._outbox.join()

    async def _send_loop(self) -> None:
        """Send posts in order; a failed send is logged, never fatal."""
        while True:
            thread, item = await self._outbox.get()
            try:
                if isinstance(item, bool):
                    await self.transport.typing(thread, item)
                else:
                    await self.transport.typing(thread, False)
                    await self.transport.post(thread, item)
            except Exception:  # the platform's errors are many; keep the session going
                log.exception("chat send failed")
            finally:
                self._outbox.task_done()

    def _say(self, text: str) -> None:
        """Queue a line from the bot in the current thread."""
        if self.thread is not None:
            self._outbox.put_nowait((self.thread, Post(text)))

    # --- the engine -----------------------------------------------------------------

    def _attach(self, engine: DebateEngine, thread: int, entry: ChatThread | None) -> None:
        """Make `engine` the open debate, posting its events in `thread`."""
        self.engine, self.thread = engine, thread
        self._hosts = set()  # a handover never outlives its debate
        reserved = [engine.user_name, *engine.personas, *(p.name for p in engine.personas.values())]
        self.names = NameBook(reserved, entry.names if entry else None)
        self._remember()
        engine.subscribe(self._on_event)

    def _remember(self) -> None:
        """Save the session's thread and names, for /resume."""
        if self.engine is not None and self.thread is not None:
            entry = ChatThread(thread=self.thread, names=dict(self.names.by_id))
            self.threads.put(self.engine.meta.id, entry)

    def _on_event(self, event: Event) -> None:
        """Queue the posts for an engine event; note who the moderator asked."""
        engine, thread = self.engine, self.thread
        if engine is None or thread is None:
            return
        if isinstance(event, TurnStarted):
            self._outbox.put_nowait((thread, True))
            return
        if isinstance(event, RoundStarted):
            self._lines.clear()
        if isinstance(event, MessageEvent):
            if event.message.to_user:
                self._asked |= self._askees(event.message.text, engine)
            elif event.message.speaker == "system" and event.message.text.startswith(_QPAUSE):
                return  # the REPL's "/run paused"; here the front end waits instead
        posts = render_event(
            event,
            engine.nick,
            engine.moderator.name,
            POST_LIMIT,
            _role_of(engine),
            OWNER_COMMANDS | OPEN_COMMANDS,
        )
        for post in posts:
            self._outbox.put_nowait((thread, post))
        if posts and engine.busy:  # a step goes on: the next model call is under way
            self._outbox.put_nowait((thread, True))

    def _askees(self, text: str, engine: DebateEngine) -> set[int]:
        """Return the user IDs of the people a question message addresses."""
        out: set[int] = set()
        for name in _ASKED.findall(text):
            if name.casefold() == engine.user_name.casefold():
                out.add(self.config.owner)
            elif (uid := self.names.user_of(name)) is not None:
                out.add(uid)
        return out

    def _start(self, work: Callable[[], Coroutine[Any, Any, None]]) -> str | None:
        """Run engine work in the background; return why it can't start, or None."""
        if self._task is not None and not self._task.done():
            return BUSY
        self._task = asyncio.create_task(self._guarded(work))
        return None

    async def _guarded(self, work: Callable[[], Awaitable[None]]) -> None:
        """Run work; an unexpected error is logged and reported generically."""
        if self.thread is not None:
            self._outbox.put_nowait((self.thread, True))  # slow local calls: show activity
        try:
            await work()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("chat step failed")
            self._say("Something went wrong on the host; the details are in its local log.")
        finally:
            if self.thread is not None:
                self._outbox.put_nowait((self.thread, False))
            if self._offline:
                self._go_offline()

    # --- people's messages ----------------------------------------------------------

    def role(self, user_id: int) -> str | None:
        """Return "owner", "guest", or None for anyone not on the list."""
        if user_id == self.config.owner:
            return "owner"
        return "guest" if user_id in self.config.allowed_users else None

    def on_message(self, user_id: int, username: str, thread: int | None, text: str) -> None:
        """Pass a listed person's line in the debate's thread to the engine."""
        engine = self.engine
        if self.role(user_id) is None or engine is None or thread != self.thread:
            return
        text = self._plain_mentions(text.strip()[: self.config.max_message_chars], engine)
        if self._typed_command(user_id, text) or not text or engine.phase is Phase.DONE:
            return
        count = self._lines.get(user_id, 0) + 1
        self._lines[user_id] = count
        cap = self.config.max_lines_per_round
        if count > cap:
            if count == cap + 1:  # say it once per round
                self._say(LINE_LIMIT.format(name=self._name(user_id, username), count=cap))
            return
        engine.say(text, None if user_id == self.config.owner else self._name(user_id, username))
        if user_id in self._waiting:
            self._waiting[user_id].set()

    def _plain_mentions(self, text: str, engine: DebateEngine) -> str:
        """Write Discord mentions as debate names, so no raw ID reaches the council.

        The owner is the debate's user name, a guest who has spoken their debate name;
        anyone else, a role or a channel is "@someone".
        """

        def name(m: re.Match[str]) -> str:
            """Return the debate name for one mention."""
            if m["kind"] not in ("@", "@!"):
                return SOMEONE
            uid = int(m["id"])
            if uid == self.config.owner:
                return f"@{engine.user_name}"
            known = self.names.by_id.get(uid)
            return f"@{known}" if known else SOMEONE

        return _MENTION.sub(name, text)

    def _typed_command(self, user_id: int, text: str) -> bool:
        """Keep a command sent as text out of the debate; tell the owner, not guests.

        Discord sends "/offline" as an ordinary line unless it is picked from the
        command list. Only the owner (and a friend they handed over to) gets the
        hint, so other guests can't make the bot post.
        """
        m = _TYPED.match(text)
        if m is None or m["name"] not in OWNER_COMMANDS | OPEN_COMMANDS:
            return False
        log.info("chat: /%s typed as text by %s, not run", m["name"], user_id)
        if user_id == self.config.owner or self._is_host(user_id):
            self._say(TYPED_COMMAND.format(command=f"/{m['name']}"))
        return True

    def _name(self, user_id: int, username: str) -> str:
        """Return a guest's debate name, giving one the first time they speak."""
        known = user_id in self.names.by_id
        name = self.names.name_for(user_id, username, self.config.names.get(str(user_id)))
        if not known:
            self._remember()
        return name

    # --- commands -------------------------------------------------------------------

    async def command(
        self, user_id: int, name: str, arg: str = "", council: str = "", username: str = ""
    ) -> str:
        """Run one slash command; return the private reply for the person who sent it.

        Each command is logged (name, user ID, role, reply), never its argument: a
        question is personal. /status and /flaws replies are debate content: not logged.
        `/handover` takes the friend's user ID as `arg` and their `username`.
        """
        role = self.role(user_id)
        if role == "guest" and self._is_host(user_id):
            role = "host"
        log.info("chat command /%s from %s (%s)", name, user_id, role or "not listed")
        if name == "handover" and role == "owner":
            reply = self._handover(arg, username)
        else:
            reply = await self._command(user_id, role, name, arg, council)
        log.info("chat command /%s: %s", name, "replied" if name in OPEN_COMMANDS else reply)
        return reply

    async def _command(
        self, user_id: int, role: str | None, name: str, arg: str, council: str
    ) -> str:
        """Check the person's rights, then run the command."""
        if role is None:
            return NOT_LISTED
        if name in HANDOVER_COMMANDS and role not in ("owner", "host"):
            return NOT_HANDED
        if name in OWNER_COMMANDS - HANDOVER_COMMANDS and role != "owner":
            return NOT_ALLOWED
        if name not in OWNER_COMMANDS | OPEN_COMMANDS:
            return f"Unknown command /{name}."
        if name == "takeback":
            return self._takeback()
        if name == "ask":
            return await self._ask(arg, council or self.config.council)
        if name == "resume":
            return await self._resume()
        if name == "offline":
            return self._request_offline()
        engine = self.engine
        if engine is None:
            return NO_DEBATE
        if name == "status":
            return self._status(engine)
        if name == "flaws":
            return flaws_text(engine.meta.runtime.flaws, engine.nick)[:1900]
        if name == "pause":
            engine.request_pause()
            return "Pausing at the next turn boundary."
        return self._engine_command(engine, name, arg)

    def _is_host(self, user_id: int) -> bool:
        """Whether the open, unfinished debate was handed over to this friend."""
        engine = self.engine
        return user_id in self._hosts and engine is not None and engine.phase is not Phase.DONE

    def _handover(self, arg: str, username: str) -> str:
        """Let a friend on the list run, pause, vote, close and /offline the open debate."""
        engine = self.engine
        if engine is None or engine.phase is Phase.DONE:
            return NO_DEBATE
        uid = int(arg) if arg.isdigit() else None
        if uid == self.config.owner:
            return "You already hold the floor."
        if uid is None or self.role(uid) != "guest":
            return "Only a friend on the council's list can take the floor."
        name = self._name(uid, username)
        if uid not in self._hosts:
            self._hosts.add(uid)
            self._say(HANDED.format(name=name))
        return (
            f"@{name} can now run, pause, vote, close and /offline this debate; /takeback ends it."
        )

    def _takeback(self) -> str:
        """End every handover of the open debate."""
        if not self._hosts:
            return "Nobody else holds the floor."
        self._hosts = set()
        self._say(TAKEN_BACK)
        return "You hold the floor alone again."

    def _engine_command(self, engine: DebateEngine, name: str, arg: str) -> str:
        """Start /run, /vote or /close in the background."""
        if name == "run":
            left = max(engine.max_rounds - engine.meta.runtime.round, 1)
            rounds = parse_count(arg) if arg.strip() else left
            if rounds is None:
                return "Usage: /run [N], N ≥ 1."
            return (
                self._start(lambda: self._run_rounds(engine, rounds))
                or f"Running {rounds} round(s)."
            )
        if name == "vote":
            return self._start(engine.vote) or "Calling a vote."
        return self._start(engine.close) or "Closing: the synthesis comes next."

    async def _ask(self, question: str, council: str) -> str:
        """Start a debate on the owner's question, in a new thread."""
        question = " ".join(question.split())
        if not question:
            return "Usage: /ask question:<your question>."
        if self.engine is not None and self.engine.phase is not Phase.DONE:
            return ONE_AT_A_TIME
        try:
            engine = self.new_engine(council, question)
        except ChatSetupError as exc:
            return str(exc)
        opening = (
            f"**{engine.user_name} asks:** {question}\n*Council: {council} · local models only*"
        )
        thread = await self.transport.open_thread(question[:90], opening)
        self._attach(engine, thread, None)
        self._start(engine.step)  # the opening (and its research)
        return f"Started in <#{thread}>; /run when the opening is in."

    async def _resume(self) -> str:
        """Reopen the latest unfinished chat session in its thread."""
        if self.engine is not None and self.engine.phase is not Phase.DONE:
            return ONE_AT_A_TIME
        engine = self.resume_engine()
        entry = self.threads.get(engine.meta.id) if engine is not None else None
        if engine is None or entry is None:
            return "No unfinished chat session to resume."
        self._attach(engine, entry.thread, entry)
        self._say("**Council back online.** The owner continues with /run.")
        engine.announce_resume()
        return f"Resumed in <#{entry.thread}>."

    def _status(self, engine: DebateEngine) -> str:
        """Return a one-line summary of the open debate."""
        rt = engine.meta.runtime
        busy = "busy" if engine.busy else "idle"
        waiting = ", ".join(self._display(uid, engine) for uid in self._waiting) or "nobody"
        hosts = [self._display(uid, engine) for uid in sorted(self._hosts) if self._is_host(uid)]
        handed = f" Handed over to {', '.join(hosts)}." if hosts else ""
        return (
            f"{engine.council.name}: {engine.phase.value}, round {rt.round}/{engine.max_rounds}, "
            f"{busy}; waiting for {waiting}.{handed}"
        )

    def _display(self, user_id: int, engine: DebateEngine) -> str:
        """Return the @name of a person."""
        if user_id == self.config.owner:
            return f"@{engine.user_name}"
        return f"@{self.names.by_id.get(user_id, 'guest')}"

    # --- running rounds and waiting for answers -------------------------------------

    async def _run_rounds(self, engine: DebateEngine, rounds: int) -> None:
        """Run rounds; when the moderator asks people, wait for them, then go on."""
        left = rounds
        while left > 0 and not self._offline:
            self._asked = set()
            left -= await engine.run(left)
            # /offline during the step: stop now, not after the 10-min wait for answers.
            if self._offline or not self._asked or engine.phase in (Phase.SYNTHESIS, Phase.DONE):
                break
            await self._wait_answers(engine)

    async def _wait_answers(self, engine: DebateEngine) -> None:
        """Wait for each person asked, up to the timeout; then the debate goes on."""
        self._waiting = {uid: asyncio.Event() for uid in self._asked}
        names = ", ".join(self._display(uid, engine) for uid in self._waiting)
        timeout = self.config.question_timeout_s
        self._say(WAITING.format(minutes=max(round(timeout / 60), 1), names=names))
        try:
            waits = [event.wait() for event in self._waiting.values()]
            await asyncio.wait_for(asyncio.gather(*waits), timeout)
        except TimeoutError:
            missing = [self._display(u, engine) for u, e in self._waiting.items() if not e.is_set()]
            self._say(NO_ANSWER.format(names=", ".join(missing)))
        finally:
            self._waiting = {}

    # --- going offline --------------------------------------------------------------

    def _request_offline(self) -> str:
        """Stop at the next turn boundary (never mid-call), save, post, then disconnect."""
        self._offline = True
        running = self._task is not None and not self._task.done()
        log.info(
            "chat: offline requested (step running: %s, waiting: %s)", running, bool(self._waiting)
        )
        if not running:
            self._go_offline()
            return "Going offline."
        if self._waiting:  # nothing in flight while waiting for answers: stop now
            assert self._task is not None
            self._task.cancel()
            self._go_offline()
        elif self.engine is not None:
            self.engine.request_pause()
        return "Going offline after the current turn."

    def _go_offline(self) -> None:
        """Post the offline line and let the adapter disconnect once it's sent."""
        if self.stopped.is_set():
            return
        log.info("chat: going offline (thread %s)", self.thread)
        self._say(OFFLINE)
        self.stopped.set()


def _role_of(engine: DebateEngine) -> RoleOf:
    """Return a lookup of a speaker's persona role, for the turn headers."""

    def role(slug: str) -> Role | None:
        """Return the role, or None for anyone who isn't a persona."""
        persona = engine.persona_of(slug)
        return persona.meta.role if persona else None

    return role
