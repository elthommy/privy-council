"""IRC-style widgets: top bar, chat log, nick list and debate-state panel."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.segment import Segment
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import Paste, Resize
from textual.message import Message as TextualMessage
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static

from privy_council.engine import prompts
from privy_council.engine.budget import BudgetReport
from privy_council.engine.notices import RESEARCH_SOURCE_LABELS
from privy_council.models.persona import BackendKind, Persona, Role
from privy_council.models.session import (
    BudgetWarning,
    DossierAdded,
    ErrorEvent,
    Event,
    MemoryProposed,
    Message,
    MessageEvent,
    ResearchAdded,
    RoundStarted,
    SessionState,
    VoteTaken,
)
from privy_council.tui.commands import (
    QUESTION_HINT,
    VOTE_TITLE,
    dossier_title,
    proposal_text,
    vote_rows,
)

GREY = "#5c6370"
ORANGE = "#d19a66"
RED = "#e06c75"
USER_COLOR = "#c8ccd4"

STATE_LABELS = {
    SessionState.CREATED: "starting",
    SessionState.OPENING: "opening",
    SessionState.ROUND_RUNNING: "running",
    SessionState.PAUSED: "paused",
    SessionState.SYNTHESIZING: "synthesizing",
    SessionState.CLOSING: "closing",
    SessionState.CLOSED: "closed",
    SessionState.ERROR_RECOVERABLE: "error — /step retries",
}


def badge(persona: Persona) -> str:
    """Return the IRC badge: @ for the moderator, + for Claude-backed personas."""
    if persona.meta.role in (Role.MODERATOR, Role.SCRIBE):
        return "@"
    return "+" if persona.meta.backend is BackendKind.CLAUDE_CLI else " "


class TopBar(Static):
    """Channel, topic, round counter, state and Claude budget."""

    def on_mount(self) -> None:
        """Start with no parts; `show` fills them."""
        self._parts: tuple[str, str, str, str, str] | None = None

    def show(
        self,
        *,
        channel: str,
        topic: str,
        round_text: str,
        state: str,
        budget: str,
    ) -> None:
        """Redraw the bar."""
        self._parts = (channel, topic, round_text, state, budget)
        self._draw()

    def on_resize(self) -> None:
        """Refit the topic to the new width."""
        self._draw()

    def _draw(self) -> None:
        """Fit the bar on one line: only the topic is shortened, so the counters stay visible."""
        if self._parts is None:
            return
        channel, topic, round_text, state, budget = self._parts
        sep = ("  │  ", GREY)
        head = Text(f"#{channel} ", style="bold")
        tail = Text.assemble(sep, round_text, sep, (state, "bold"), sep, budget)
        room = self.content_size.width - head.cell_len - tail.cell_len
        title = Text(" ".join(topic.split()), style="italic")  # a multi-line question
        title.truncate(max(room, 10), overflow="ellipsis")
        self.update(Text.assemble(head, title, tail))


LOCAL_ONLY_BADGE = "local-only"  # shown instead of the Claude budget (M15)


def budget_text(report: BudgetReport | None, session_cap: int) -> str:
    """Summarize Claude usage for the top bar."""
    if report is None:
        return "Claude: n/a"
    cap = f"/{session_cap}" if session_cap else ""
    quota = report.quota.five_hour if report.quota else None
    five = f" · 5h {quota:.0%}" if quota is not None else ""
    return f"Claude {report.session.calls}{cap}{five}"


class ChatLog(RichLog):
    """The scrolling chat; multi-line replies are rendered as Markdown.

    RichLog doesn't re-wrap written lines, so the chat keeps its renderables and
    rewrites them when its width changes (e.g. when the state pane toggles).

    It follows new messages only while the view is at the bottom. Scrolled up, it
    stays put and counts the unread items (posted as `Unread`).
    """

    class Unread(TextualMessage):
        """The number of items written since the user scrolled up (0: following)."""

        def __init__(self, count: int) -> None:
            """Carry the count."""
            super().__init__()
            self.count = count

    def on_mount(self) -> None:
        """Start with an empty history, following new items."""
        self._items: list[RenderableType] = []
        self._starts: list[int] = []  # first line of each item, for keeping the view on resize
        self._width = 0
        # A flag, not `is_vertical_scroll_end`: right after a write the end has moved
        # but the deferred scroll hasn't run yet, so the view looks scrolled up.
        self.following = True
        self.unread = 0
        self._rewrapping = False

    def add(self, renderable: RenderableType) -> None:
        """Write a renderable and remember it for re-wrapping."""
        self._items.append(renderable)
        self._starts.append(len(self.lines))
        self.write(renderable, scroll_end=self.following)
        if not self.following:
            self._set_unread(self.unread + 1)

    def follow(self) -> None:
        """Jump to the newest item and follow again."""
        self.following = True
        self._set_unread(0)
        self.scroll_end(animate=False, immediate=False, x_axis=False)

    def _set_unread(self, count: int) -> None:
        """Update the unread count and tell the app."""
        if count != self.unread:
            self.unread = count
            self.post_message(self.Unread(count))

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Stop following when the user scrolls up; follow again at the bottom."""
        super().watch_scroll_y(old_value, new_value)
        if self._rewrapping:
            return
        self.following = round(new_value) >= self.max_scroll_y
        if self.following:
            self._set_unread(0)

    def on_resize(self, event: Resize) -> None:
        """Re-wrap the history when the width changes (after RichLog's own handling)."""
        super().on_resize(event)
        width = self.scrollable_content_region.width
        if width != self._width:
            self._width = width
            self._rewrap()
        elif self.following:  # shorter (e.g. the typing area opened): keep the newest in view
            self.scroll_end(animate=False, immediate=False, x_axis=False)

    def _rewrap(self) -> None:
        """Rewrite every item at the new width, keeping the top item in view if scrolled up."""
        top = round(self.scroll_y)
        index = max(bisect_right(self._starts, top) - 1, 0)
        offset = top - self._starts[index] if self._starts else 0
        self._rewrapping = True
        try:
            self.clear()
            self._starts = []
            for item in self._items:
                self._starts.append(len(self.lines))
                self.write(item, scroll_end=self.following)
            if not self.following and self._starts:
                end = self._starts[index + 1] if index + 1 < len(self._starts) else len(self.lines)
                y = min(self._starts[index] + offset, max(end - 1, self._starts[index]))
                self.scroll_to(y=y, animate=False, immediate=True)
        finally:
            self._rewrapping = False

    def render_line(self, y: int) -> Strip:
        """Render a row with selection offsets, highlighting the selected span."""
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        line = scroll_y + y
        selection = self.text_selection
        span = selection.get_span(line) if selection and line < len(self.lines) else None
        if span is not None:
            length = strip.cell_length
            start = max(span[0] - scroll_x, 0)
            end = length if span[1] == -1 else min(span[1] - scroll_x, length)
            if start < end:
                before, middle, after = strip.divide([start, end, length])
                style = self.screen.get_component_rich_style("screen--selection")
                middle = Strip(Segment.apply_style(middle, post_style=style), middle.cell_length)
                strip = Strip.join([before, middle, after])
        return strip.apply_offsets(scroll_x, line)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return the selected text, without the padding that fills each row."""
        text = "\n".join(strip.text.rstrip() for strip in self.lines)
        return selection.extract(text), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        """Repaint the highlight."""
        self.refresh()

    def add_event(
        self, event: Event, nick_of: Callable[[str], str], color_of: Callable[[str], str]
    ) -> None:
        """Write one engine event, if it has a chat representation."""
        if isinstance(event, MessageEvent):
            self.add(render_message(event.message, nick_of, color_of))
        elif isinstance(event, RoundStarted):
            self.add(Text(f"-- {event.label or f'round {event.round}'} --", style=GREY))
        elif isinstance(event, ResearchAdded):
            source = RESEARCH_SOURCE_LABELS[event.note.source]
            title = Text(f"-- research ({source}): {event.note.focus} --", style=GREY)
            self.add(Group(title, Padding(Markdown(event.note.text), (0, 0, 0, 2))))
        elif isinstance(event, BudgetWarning):
            self.add(Text(f"! budget: {event.text}", style=ORANGE))
        elif isinstance(event, ErrorEvent):
            self.add(Text(f"!! {event.text}", style=RED))
        elif isinstance(event, MemoryProposed):
            self.add(Text(proposal_text(event.facts, event.rewrite), style=ORANGE))
        elif isinstance(event, DossierAdded):
            f = event.file
            title = Text(dossier_title(f.name, f.tokens, f.summarized), style=GREY)
            # Show the summary and computed figures so the user can check them; a verbatim
            # file is theirs already.
            shown = [f.text] if f.summarized else []
            shown += [f.stats] if f.stats else []
            body = [Padding(Markdown("\n\n".join(shown)), (0, 0, 0, 2))] if shown else []
            self.add(Group(title, *body))
        elif isinstance(event, VoteTaken):
            self.add(vote_table(event, nick_of, color_of))

    def notice(self, text: str, style: str = GREY) -> None:
        """Write a UI-only line (help, usage), not part of the transcript."""
        self.add(Text(text, style=style))


def vote_table(
    event: VoteTaken, nick_of: Callable[[str], str], color_of: Callable[[str], str]
) -> RenderableType:
    """Render a `/vote` as a titled table, one row per member."""
    table = Table(box=None, padding=(0, 1), show_edge=False, header_style=GREY)
    table.add_column("member", no_wrap=True)
    table.add_column("conf.", justify="right", no_wrap=True)
    table.add_column("position", ratio=1)
    for vote, (nick, confidence, position) in zip(
        event.votes, vote_rows(event.votes, nick_of), strict=True
    ):
        style = GREY if vote.error is not None else ""
        table.add_row(
            Text(nick, style=f"bold {color_of(vote.speaker)}"),
            Text(confidence, style=style),
            Text(position, style=style),
        )
    title = Text(VOTE_TITLE.format(round=event.round), style=GREY)
    return Group(title, Padding(table, (0, 0, 0, 2)))


def render_message(
    msg: Message, nick_of: Callable[[str], str], color_of: Callable[[str], str]
) -> RenderableType:
    """Render a chat message IRC-style: `[21:48] <Nick> text`."""
    if msg.speaker == "system":
        return Text(f"* {msg.text}", style=GREY)
    if msg.to_user:  # the moderator asks the user: make it impossible to miss
        header = Text.assemble(
            (f"[{msg.ts.astimezone():%H:%M}] ", GREY),
            (f"▶ <{msg.display_nick(nick_of)}> question for you:", f"bold reverse {ORANGE}"),
        )
        body = Text(msg.text, style=f"bold {ORANGE}")
        hint = Text(QUESTION_HINT, style=f"italic {ORANGE}")
        return Group(header, Padding(Group(body, hint), (0, 0, 0, 2)))
    color = USER_COLOR if msg.speaker == "user" else color_of(msg.speaker)
    header = Text.assemble(
        (f"[{msg.ts.astimezone():%H:%M}] ", GREY),
        (f"<{msg.display_nick(nick_of)}>", f"bold {color}"),
        (" [fallback]" if msg.fallback else "", ORANGE),
        " ",
    )
    if "\n" not in msg.text.strip():
        return header + Text(msg.text)
    return Group(header, Padding(Markdown(msg.text), (0, 0, 0, 2)))


class TypingArea(Static):
    """Replies being typed, under the chat: the first one live, the others as a line.

    Turns can run in parallel (Claude in the blind round), so only the reply that
    started first is shown as it grows; the others say "X is typing…" and appear
    whole in the chat when they finish. Plain text here; the chat renders Markdown.
    """

    MIN_LINES = 8  # a long reply shows its end: at least this many lines…
    COLUMN_SHARE = 0.5  # …or this share of the chat column, whichever is more

    def __init__(
        self,
        nick_of: Callable[[str], str],
        color_of: Callable[[str], str],
        *,
        id: str | None = None,
    ) -> None:
        """Start empty (hidden)."""
        super().__init__(id=id)
        self.nick_of = nick_of
        self.color_of = color_of
        self.buffers: dict[str, str] = {}  # speaker -> text so far, in start order

    def add(self, speaker: str, text: str, restart: bool) -> None:
        """Append a chunk to a speaker's reply (`restart`: a retry starts over)."""
        self.buffers[speaker] = ("" if restart else self.buffers.get(speaker, "")) + text
        self.redraw()

    def done(self, speaker: str) -> None:
        """Drop a finished (or failed) reply."""
        if self.buffers.pop(speaker, None) is not None:
            self.redraw()

    def clear(self) -> None:
        """Drop everything (the step ended)."""
        if self.buffers:
            self.buffers.clear()
            self.redraw()

    def on_resize(self, event: Resize) -> None:
        """Re-wrap at the new width."""
        self.redraw()

    def redraw(self) -> None:
        """Render the first reply's tail and one line per other typist."""
        self.set_class(bool(self.buffers), "visible")
        if not self.buffers:
            self.update("")
            return
        (speaker, text), *others = self.buffers.items()
        body = Text.assemble(
            (f"<{self.nick_of(speaker)}> ", f"bold {self.color_of(speaker)}"), text
        )
        # Hidden until now, the area has no width yet: wrap to the chat column instead.
        column = self.parent.size if isinstance(self.parent, Widget) else None
        width = max((self.size.width or (column.width if column else 0)) - 2, 20)
        room = max(self.MIN_LINES, int((column.height if column else 0) * self.COLUMN_SHARE))
        room -= len(others) + 1  # the "is typing…" lines and the border
        lines = body.wrap(self.app.console, width)[-max(room, 1) :]
        parts: list[RenderableType] = list(lines)
        parts += [Text(f"{self.nick_of(s)} is typing…", style=GREY) for s, _ in others]
        self.update(Group(*parts))


class NickList(Static):
    """Members with badges, backend/model and status (idle, thinking…, fallback)."""

    def __init__(self, personas: list[Persona], *, id: str | None = None) -> None:
        """Start with everyone idle."""
        super().__init__(id=id)
        self.personas = personas
        self.status: dict[str, str] = {p.slug: "idle" for p in personas}

    def set_status(self, slug: str, status: str) -> None:
        """Set one persona's status and redraw."""
        if slug in self.status:
            self.status[slug] = status
            self.refresh_list()

    def reset(self, keep: set[str]) -> None:
        """Set everyone not in `keep` back to idle (`keep` holds e.g. fallback personas)."""
        for slug in self.status:
            if slug not in keep:
                self.status[slug] = "idle"
        self.refresh_list()

    def set_personas(self, personas: list[Persona]) -> None:
        """Show a new roster (after `/summon` or `/kick`); newcomers start idle."""
        self.personas = personas
        self.status = {p.slug: self.status.get(p.slug, "idle") for p in personas}
        self.refresh_list()

    def refresh_list(self) -> None:
        """Redraw the list."""
        lines = []
        for p in self.personas:
            status = self.status[p.slug]
            style = ORANGE if status == "fallback" else ("bold" if status != "idle" else GREY)
            lines.append(
                Text.assemble(
                    (f"{badge(p)}{p.name}", f"bold {p.meta.color}"),
                    "\n  ",
                    (p.meta.model, GREY),
                    " ",
                    (status, style),
                )
            )
        self.update(Group(*lines))


class StatePanel(VerticalScroll):
    """Collapsible side pane with the moderator's debate state."""

    def compose(self) -> ComposeResult:
        """Hold one Static with the rendered state."""
        yield Static(prompts.EMPTY, id="state-body")

    def show_state(self, text: str) -> None:
        """Replace the displayed state."""
        self.query_one("#state-body", Static).update(Text(text))


# Shown in place of line breaks in a multi-line paste; turned back into "\n" on submit.
NEWLINE_MARK = " ⏎ "


class ChatInput(Input):
    """Single-line input that keeps multi-line pastes instead of only their first line.

    Line breaks are shown as NEWLINE_MARK so the text stays editable on one line;
    `restore` turns them back into newlines when the line is submitted.
    """

    def _on_paste(self, event: Paste) -> None:
        """Keep every line of a multi-line paste (Input's own handler keeps only the first)."""
        lines = event.text.strip("\r\n").splitlines()
        if len(lines) <= 1:
            return  # Textual then runs Input's own handler (handlers run along the MRO)
        event.prevent_default()  # skip Input's handler, which would re-insert line 1
        text = NEWLINE_MARK.join(lines)
        if self.selection.is_empty:
            self.insert_text_at_cursor(text)
        else:
            self.replace(text, *self.selection)
        event.stop()

    @staticmethod
    def restore(value: str) -> str:
        """Turn newline marks back into newlines."""
        return value.replace(NEWLINE_MARK, "\n")
