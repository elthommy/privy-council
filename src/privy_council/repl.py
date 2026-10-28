"""Line-based plain-text front end (`council new ... --no-tui`).

Runs every engine call on one asyncio loop (`asyncio.Runner`), reading input
between steps, so no threads are needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any

import typer

from privy_council.engine.notices import RESEARCH_SOURCE_LABELS
from privy_council.engine.protocol import DebateEngine
from privy_council.models.session import (
    BudgetWarning,
    DossierAdded,
    ErrorEvent,
    Event,
    MemoryProposed,
    MessageEvent,
    Phase,
    ResearchAdded,
    RoundStarted,
    StateUpdated,
    TurnStarted,
    VoteTaken,
)
from privy_council.tui.commands import (
    AS_USAGE,
    COMMANDS,
    QUESTION_HINT,
    VOTE_TITLE,
    Command,
    Judgment,
    dossier_title,
    export_command,
    flaws_text,
    parse,
    parse_count,
    proposal_text,
    split_as,
    split_ask,
    state_text,
    vote_rows,
)
from privy_council.tui.widgets import LOCAL_ONLY_BADGE

Output = Callable[[str, str | None], None]  # (line, hex color or None)
PROMPT = "> "
TOPIC_PROMPT = "Question: "
ELLIPSIS = " […]"
REPLAY_TAIL = 8  # events shown again on resume


def format_event(event: Event, engine: DebateEngine) -> tuple[str, str | None] | None:
    """Render an event as one display line and its color; None hides it."""
    if isinstance(event, MessageEvent):
        return _format_message(event, engine)
    if isinstance(event, RoundStarted):
        return f"-- {event.label or f'round {event.round}'} --", None
    if isinstance(event, ResearchAdded):
        note = event.note
        source = RESEARCH_SOURCE_LABELS[note.source]
        return f"-- research ({source}): {note.focus} --\n{note.text}", None
    if isinstance(event, TurnStarted):
        return f"   {engine.nick(event.speaker)} is thinking…", "#5c6370"
    if isinstance(event, StateUpdated):
        return "* debate state updated (/state to view)", "#5c6370"
    if isinstance(event, BudgetWarning):
        return f"! budget: {event.text}", "#d19a66"
    if isinstance(event, ErrorEvent):
        return f"!! {event.text}", "#e06c75"
    if isinstance(event, MemoryProposed):
        return proposal_text(event.facts, event.rewrite), "#d19a66"
    if isinstance(event, VoteTaken):
        return vote_text(event, engine), None
    if isinstance(event, DossierAdded):
        f = event.file
        title = dossier_title(f.name, f.tokens, f.summarized)
        shown = [title, *([f.text] if f.summarized else []), *([f.stats] if f.stats else [])]
        return "\n".join(shown), None
    return None


def vote_text(event: VoteTaken, engine: DebateEngine) -> str:
    """Render a `/vote` as aligned plain-text columns."""
    rows = vote_rows(event.votes, engine.nick)
    width = max(len(nick) for nick, _, _ in rows)
    lines = [f"  {nick:<{width}}  {conf:>4}  {pos}" for nick, conf, pos in rows]
    return "\n".join([VOTE_TITLE.format(round=event.round), *lines])


def _format_message(event: MessageEvent, engine: DebateEngine) -> tuple[str, str | None]:
    """Render a chat message IRC-style, clipped for display (the transcript keeps it all)."""
    msg = event.message
    if msg.speaker == "system":
        return f"* {msg.text}", "#5c6370"
    if msg.to_user:
        when = f"{msg.ts.astimezone():%H:%M}"
        header = f"[{when}] ▶ <{msg.display_nick(engine.nick)}> question for you:"
        return f"{header}\n{msg.text}\n{QUESTION_HINT}", "#d19a66"
    limit = engine.settings.engine.max_turn_chars_display
    # The moderator's opening and synthesis are meant to be read whole.
    clip = msg.speaker != engine.moderator.slug and len(msg.text) > limit
    text = msg.text[:limit] + ELLIPSIS if clip else msg.text
    tag = " [fallback]" if msg.fallback else ""
    persona = engine.persona_of(msg.speaker)
    color = persona.meta.color if persona else None
    return f"[{msg.ts.astimezone():%H:%M}] <{msg.display_nick(engine.nick)}>{tag} {text}", color


def echo(line: str, color: str | None) -> None:
    """Print a line, in the given hex color if any."""
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) if color else None
    typer.secho(line, fg=rgb)


def budget_lines(engine: DebateEngine) -> list[str]:
    """Return `/budget` output lines."""
    ledger = engine.registry.ledger
    if ledger is None:
        return ["no usage ledger configured"]
    report = ledger.report(session=engine.meta.id)
    caps = engine.settings.claude
    return [
        f"session: {report.session.calls}/{caps.session_soft_cap_calls} Claude calls, "
        f"{report.session.input_tokens:,} in / {report.session.output_tokens:,} out tokens",
        f"today:   {report.today.calls}/{caps.daily_soft_cap_calls} Claude calls",
    ]


async def _run(engine: DebateEngine, arg: str, out: Output) -> None:
    """Handle `/run N`, showing budget warnings first (the REPL can't be interrupted)."""
    rounds = parse_count(arg)
    if rounds is None:
        out("usage: /run N (N ≥ 1)", None)
        return
    for warning in engine.budget_warnings(engine.estimate_claude_calls(rounds)):
        out(f"! budget: {warning}", "#d19a66")
    await engine.run(rounds)


async def dispatch(engine: DebateEngine, cmd: Command, out: Output) -> bool:
    """Run one command; return False when the REPL should exit."""
    if cmd.name == "step":
        await engine.step()
    elif cmd.name == "run":
        await _run(engine, cmd.arg, out)
    elif cmd.name == "more":
        rounds = parse_count(cmd.arg)
        if rounds is None:
            out("usage: /more N (N ≥ 1)", None)
        else:
            engine.more(rounds)
    elif cmd.name == "ask":
        parts = split_ask(cmd.arg)
        if parts is None:
            out("usage: /ask @nick <question>", None)
        else:
            await engine.ask(*parts)
    elif cmd.name == "research":
        await engine.research(cmd.arg or None)
    elif cmd.name == "summon":
        engine.summon(cmd.arg)
    elif cmd.name == "kick":
        engine.kick(cmd.arg)
    elif cmd.name == "attach":
        await engine.attach(cmd.arg)
    elif cmd.name == "pause":
        out("/pause needs the TUI; in the REPL each /step already pauses", None)
    elif cmd.name == "as":
        said = split_as(cmd.arg)
        if said is None:
            out(AS_USAGE, None)
        else:
            engine.say(*said)
    elif cmd.name == "say":
        if cmd.arg:
            engine.say(cmd.arg)
        else:
            out("usage: /say <text>", None)
    elif cmd.name == "vote":
        await engine.vote()
    elif cmd.name == "summary":
        await engine.summary()
    elif cmd.name == "state":
        rt = engine.meta.runtime
        judged = (  # nothing is judged before round 1
            Judgment(rt.consensus, rt.stalled_rounds, rt.decision_open, rt.new_arguments)
            if rt.round
            else None
        )
        out(state_text(rt.debate_state, judged, engine.nick), None)
    elif cmd.name == "flaws":
        out(flaws_text(engine.meta.runtime.flaws, engine.nick), None)
    elif cmd.name == "export":
        out(export_command(engine, cmd.arg), None)
    elif cmd.name == "lang":
        engine.set_language(cmd.arg)
    elif cmd.name == "budget":
        for line in budget_lines(engine):
            out(line, None)
    elif cmd.name == "close":
        await engine.close()
        return not _finished(engine)  # stay if the synthesis failed or facts await /keep
    elif cmd.name == "keep":
        await engine.keep(cmd.arg)
        return not _finished(engine)
    elif cmd.name == "forget":
        engine.forget(cmd.arg)
    elif cmd.name == "quit":
        if engine.memory_review_pending:
            out("Unreviewed shared facts were not saved.", "#d19a66")
        out(f"Saved session {engine.meta.id}", None)
        return False
    elif cmd.name == "log":
        out(f"Live log: {engine.live_log}", None)
    elif cmd.name == "help":
        for name, text in COMMANDS.items():
            out(f"/{name:<7} {text}", None)
    else:
        out(f"unknown command /{cmd.name}; /help lists commands", "#e06c75")
    return True


def _finished(engine: DebateEngine) -> bool:
    """Return whether the session is closed with nothing left to approve."""
    return engine.phase is Phase.DONE and not engine.memory_review_pending


def _replay_tail(engine: DebateEngine, out: Output) -> None:
    """Print the last few events of a resumed session (the transcript has them all)."""
    shown = [e for e in engine.history if not isinstance(e, StateUpdated)]
    tail = shown[-REPLAY_TAIL:]
    if len(shown) > len(tail):
        out(f"… {len(shown) - len(tail)} earlier lines in {engine.live_log}", "#5c6370")
    for event in tail:
        line = format_event(event, engine)
        if line is not None:
            out(*line)


def _read_topic(engine: DebateEngine, read_line: Callable[[str], str]) -> bool:
    """Ask for the question of a session started without one; False on EOF."""
    while engine.awaiting_topic:
        try:
            engine.set_topic(read_line(TOPIC_PROMPT))
        except EOFError:
            return False
    return True


def run_repl(
    engine: DebateEngine,
    *,
    read_line: Callable[[str], str] = input,
    out: Output = echo,
    on_exit: Callable[[], Coroutine[Any, Any, None]] | None = None,
    attach: Sequence[Path] = (),
) -> None:
    """Open the session (the moderator's opening) and loop over input lines.

    `attach`: files from `council new --attach`, attached before the opening.
    """

    def show(event: Event) -> None:
        """Print an engine event."""
        line = format_event(event, engine)
        if line is not None:
            out(*line)

    engine.subscribe(show)
    if engine.awaiting_topic and not _read_topic(engine, read_line):
        out("No question given; nothing to debate.", None)
        return
    badge = f"  [{LOCAL_ONLY_BADGE}]" if engine.meta.local_only else ""
    out(f"#{engine.council.name} — {engine.meta.topic}{badge}  (/help for commands)", None)
    if engine.live_log is not None:
        out(f"Live log: {engine.live_log}", None)
    with asyncio.Runner() as runner:
        try:
            if engine.resumed:
                _replay_tail(engine, out)
                engine.announce_resume()
            else:
                for path in attach:
                    runner.run(engine.attach(str(path)))
                runner.run(engine.step())
            while True:
                try:
                    cmd = parse(read_line(PROMPT))
                except EOFError:
                    cmd = Command("quit")
                if cmd is not None and not runner.run(dispatch(engine, cmd, out)):
                    break
        finally:
            if on_exit is not None:
                runner.run(on_exit())
