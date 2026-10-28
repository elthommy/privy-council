"""Slash-command parser shared by the plain-text REPL and (from M4) the TUI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from privy_council.engine import prompts
from privy_council.export import EXPORTS_DIR, ExportError, ExportFormat, export_session
from privy_council.models.flaw import Flaw
from privy_council.models.session import Consensus, DebateState
from privy_council.models.vote import Vote

if TYPE_CHECKING:
    from privy_council.engine.protocol import DebateEngine

# Brief §11 commands implemented so far (M5); the rest arrive in M6.
COMMANDS = {
    "step": "run the next phase (opening, a round, or the synthesis), then pause",
    "run": "/run N: run up to N rounds on their own (Esc or /pause stops them)",
    "more": "/more N: add N rounds past max_rounds (then /step or /run N)",
    "pause": "pause at the next turn boundary (TUI only; also Esc)",
    "say": "/say <text>: interject; the moderator integrates it (plain text works too)",
    # (with `council new <council>` and no question, the first message is the question)
    "ask": "/ask @nick <text>: a direct question to one member, outside the rounds",
    "as": "/as <name> <text>: speak as another person (tries a shared chat locally)",
    "research": "/research [focus]: the moderator searches the web; everyone sees the facts",
    "summon": "/summon <persona>: add a persona to the channel; it speaks next round",
    "kick": "/kick <nick>: remove a member from the channel",
    "attach": "/attach <path>: add a .txt/.json/.md/.csv file; everyone sees it",
    "vote": "each member states a final position and a confidence (shown as a table)",
    "summary": "the moderator writes an interim synthesis; the debate continues",
    "state": "show the debate state",
    "flaws": "errors the engine and the moderator caught in members' messages",
    "export": "/export [md|html|all]: write a readable report of the session (data/exports/)",
    "lang": "/lang [en|fr]: show the debate language, or change it before round 1",
    "budget": "Claude calls and tokens used this session and today",
    "log": "show where the live transcript file is (follow it with tail -f)",
    "close": "write the final synthesis and memory notes, and close the session",
    "keep": "/keep 1 3 | all | none: approve proposed shared facts about you",
    "forget": "/forget @nick | shared | all: wipe memory (type it twice to confirm)",
    "quit": "save and exit without the synthesis",
    "help": "list commands",
}


QUESTION_HINT = "↳ answer in plain text, then /step (or /run N) to continue"
INPUT_PLACEHOLDER = "/help for commands; plain text talks to the council"
QUESTION_PLACEHOLDER = "answer the Chair here, then /step (or /run N) to continue"

PROPOSAL_FACTS_TITLE = "-- shared memory: proposed facts (from your own words) --"
PROPOSAL_FACTS_HINT = "/keep 1 3 | all | none to choose what everyone remembers about you"
PROPOSAL_REWRITE_TITLE = "-- shared memory outgrew its cap: proposed merged version --"
PROPOSAL_REWRITE_HINT = "/keep all to use it (old copy kept as user.md.bak) | /keep none"


def proposal_text(facts: list[str], rewrite: str | None) -> str:
    """Render a shared-memory proposal for the chat (REPL and TUI)."""
    if facts:
        lines = [f"  {i}. {fact.removeprefix('- ')}" for i, fact in enumerate(facts, 1)]
        return "\n".join([PROPOSAL_FACTS_TITLE, *lines, PROPOSAL_FACTS_HINT])
    return "\n".join([PROPOSAL_REWRITE_TITLE, rewrite or "", PROPOSAL_REWRITE_HINT])


STATE_DYNAMICS = "Consensus: {consensus} · decision: {decision} · stalled rounds: {stalled}"
STATE_NEW_TITLE = "New last round:"
STATE_NOTHING_NEW = "- (nothing new)"


@dataclass(frozen=True)
class Judgment:
    """The moderator's latest round judgment, as shown above the debate state."""

    consensus: Consensus
    stalled: int
    decision_open: bool | None = None  # None: from a transcript older than the field
    new_arguments: list[str] | None = None


def state_text(state: DebateState, judgment: Judgment | None, nick_of: Callable[[str], str]) -> str:
    """Render the debate state for display, under the moderator's judgment (once made)."""
    body = prompts.render_debate_state(state, nick_of)
    if judgment is None:
        return body
    decision = {True: "open", False: "settled", None: "?"}[judgment.decision_open]
    lines = [
        STATE_DYNAMICS.format(
            consensus=judgment.consensus.value, decision=decision, stalled=judgment.stalled
        )
    ]
    if judgment.new_arguments is not None:
        lines.append(STATE_NEW_TITLE)
        lines += [f"- {a}" for a in judgment.new_arguments] or [STATE_NOTHING_NEW]
    return "\n".join([*lines, body])


FLAWS_TITLE = "-- flaws caught (flagged only; the messages are unchanged) --"
NO_FLAWS = "No flaws caught so far."
FLAW_LINE = "  r{round} {nick} [{kind}, {by}]: {text}"


def flaws_text(flaws: list[Flaw], nick_of: Callable[[str], str]) -> str:
    """Render the session's flaws for `/flaws`, oldest first."""
    if not flaws:
        return NO_FLAWS
    lines = [FLAWS_TITLE]
    for f in flaws:
        text = f'"{f.excerpt}"' + (f" ({f.note})" if f.note else "")
        if f.by == "engine" and f.note:  # the engine's note is the figure in question
            text = f'{f.note} in "{f.excerpt}"'
        lines.append(
            FLAW_LINE.format(
                round=f.round, nick=nick_of(f.member), kind=f.kind.value, by=f.by, text=text
            )
        )
    return "\n".join(lines)


def dossier_title(name: str, tokens: int, summarized: bool) -> str:
    """Return the chat line announcing an attached file."""
    how = "summary" if summarized else "verbatim"
    return f"-- dossier: {name} (~{tokens:,} tokens, {how}) --"


VOTE_TITLE = "-- vote after round {round} --"
NO_VOTE = "—"


VOTE_DISPLAY_CHARS = 200  # a longer position is shortened in the table (kept whole on disk)


def shorten(text: str, limit: int) -> str:
    """Cut text at a word boundary to at most `limit` characters, marking the cut with "…"."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0] or text[: limit - 1]
    return cut.rstrip(" ,;:") + "…"


def vote_rows(votes: list[Vote], nick_of: Callable[[str], str]) -> list[tuple[str, str, str]]:
    """Return a vote's rows as (nick, confidence, position) display strings.

    Long positions are shortened for the chat; transcript.md and the synthesis get them whole.
    """
    rows = []
    for v in votes:
        nick = nick_of(v.speaker) + (" [fallback]" if v.fallback else "")
        if v.error is not None:
            rows.append((nick, NO_VOTE, f"(no vote: {v.error})"))
        else:
            rows.append((nick, f"{v.confidence}%", shorten(v.position, VOTE_DISPLAY_CHARS)))
    return rows


@dataclass(frozen=True)
class Command:
    """A parsed input line: command name (lowercase, no slash) and its argument."""

    name: str
    arg: str = ""


def parse(line: str) -> Command | None:
    """Parse one input line; plain text is `/say`, blank lines are None."""
    text = line.strip()
    if not text:
        return None
    if not text.startswith("/"):
        return Command("say", text)
    name, _, arg = text[1:].partition(" ")
    return Command(name.lower(), arg.strip())


def split_ask(arg: str) -> tuple[str, str] | None:
    """Split `/ask` arguments into (nick, question), or None if either is missing."""
    who, _, question = arg.partition(" ")
    if not who or not question.strip():
        return None
    return who, question.strip()


AS_USAGE = "usage: /as <name> <text>"


def split_as(arg: str) -> tuple[str, str] | None:
    """Split `/as` arguments into (text, author), or None if either is missing."""
    who, _, text = arg.partition(" ")
    who = who.lstrip("@")
    if not who or not text.strip():
        return None
    return text.strip(), who


def parse_count(arg: str) -> int | None:
    """Parse the N of `/run N` (a positive integer), or None if invalid."""
    return int(arg) if arg.isdigit() and int(arg) > 0 else None


EXPORT_USAGE = "usage: /export [md|html|all]"


def export_command(engine: DebateEngine, arg: str) -> str:
    """Run `/export`: write the session's report, return the line to show."""
    if engine.store is None:
        return "This session is not saved"
    try:
        fmt = ExportFormat(arg.strip().lower() or ExportFormat.MD.value)
    except ValueError:
        return EXPORT_USAGE
    try:
        paths = export_session(
            engine.store.directory,
            engine.personas,
            user_name=engine.user_name,
            exports_dir=engine.settings.data_dir / EXPORTS_DIR,
            fmt=fmt,
        )
    except ExportError as exc:
        return f"Export failed: {exc}"
    return "Exported: " + ", ".join(str(p) for p in paths)
