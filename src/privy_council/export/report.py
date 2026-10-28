"""What an exported session shows, built from the saved files only (no model calls)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from privy_council.models.flaw import FIGURE_KINDS, Flaw
from privy_council.models.persona import Persona
from privy_council.models.session import (
    DossierAdded,
    ErrorEvent,
    Event,
    MessageEvent,
    Phase,
    ResearchAdded,
    RoundStarted,
    SessionMeta,
    SessionState,
    SessionStateChanged,
    VoteTaken,
)
from privy_council.storage.sessions import SYNTHESIS_MD, SessionStore

DEFAULT_COLOR = "#abb2bf"
# What the transcript part of a report shows; the rest is engine bookkeeping or
# personal (memory proposals) and stays out of a file meant to be reread or shared.
ShownEvent = MessageEvent | RoundStarted | ResearchAdded | DossierAdded | VoteTaken | ErrorEvent


class ExportError(Exception):
    """The session can't be read or the report can't be written."""


class Seat(BaseModel):
    """One participant: the moderator, a member, or a member kicked along the way."""

    model_config = ConfigDict(frozen=True)

    slug: str
    nick: str
    role: str
    color: str
    models: list[str]  # "backend:model", in the order they were used
    recorded: bool  # False: taken from today's persona file (sessions before M13)


class Report(BaseModel):
    """A saved session, ready to render."""

    model_config = ConfigDict(frozen=True)

    meta: SessionMeta
    user_name: str
    seats: list[Seat]
    synthesis: str | None
    events: list[ShownEvent]
    flaws: list[Flaw]
    final_vote: VoteTaken | None

    def seat(self, slug: str) -> Seat | None:
        """Return the participant with this slug, if any."""
        return next((s for s in self.seats if s.slug == slug), None)

    def nick(self, slug: str) -> str:
        """Return the display nick of a speaker slug ("user" is the user's name)."""
        if slug == "user":
            return self.user_name
        seat = self.seat(slug)
        return seat.nick if seat else slug

    def color(self, slug: str) -> str:
        """Return a speaker's color (the default grey for the user and the unknown)."""
        seat = self.seat(slug)
        return seat.color if seat else DEFAULT_COLOR


def build_report(directory: Path, personas: Mapping[str, Persona], user_name: str) -> Report:
    """Read a session directory into a Report; raises ExportError if it is unreadable."""
    store = SessionStore(directory)
    try:
        meta = store.load_meta()
        events = store.events()
        synthesis_path = directory / SYNTHESIS_MD
        synthesis = synthesis_path.read_text(encoding="utf-8") if synthesis_path.is_file() else None
    except (OSError, ValueError) as exc:
        raise ExportError(f"can't read session {directory.name}: {exc}") from exc
    votes = [e for e in events if isinstance(e, VoteTaken)]
    return Report(
        meta=meta,
        user_name=user_name,
        seats=_seats(meta, events, personas),
        synthesis=synthesis.strip() if synthesis else None,
        events=_shown(events, meta.moderator),
        flaws=[f for f in meta.runtime.flaws if f.by == "moderator" or f.kind in FIGURE_KINDS],
        final_vote=votes[-1] if votes else None,
    )


def _shown(events: list[Event], moderator: str) -> list[ShownEvent]:
    """Keep the transcript events, minus the synthesis (shown at the top instead)."""
    shown: list[ShownEvent] = []
    synthesizing = False
    for event in events:
        if isinstance(event, SessionStateChanged):
            synthesizing = event.state is SessionState.SYNTHESIZING
        elif isinstance(event, MessageEvent):
            if not (synthesizing and event.message.speaker == moderator):
                shown.append(event)
        elif isinstance(event, ShownEvent):
            shown.append(event)
    return shown


def _seats(meta: SessionMeta, events: list[Event], personas: Mapping[str, Persona]) -> list[Seat]:
    """List the moderator, the members, then anyone else who spoke (since kicked)."""
    slugs = [meta.moderator, *meta.members]
    for event in events:
        if isinstance(event, MessageEvent):
            speaker = event.message.speaker
            if speaker not in ("user", "system") and speaker not in slugs:
                slugs.append(speaker)
    return [_seat(slug, meta, personas.get(slug)) for slug in slugs]


def _seat(slug: str, meta: SessionMeta, persona: Persona | None) -> Seat:
    """Describe one participant from the session's record, else its persona file."""
    recorded = meta.models.get(slug, [])
    current = [f"{persona.meta.backend.value}:{persona.meta.model}"] if persona else []
    return Seat(
        slug=slug,
        nick=persona.name if persona else slug,
        role=persona.meta.role.value if persona else "",
        color=persona.meta.color if persona else DEFAULT_COLOR,
        models=recorded or current,
        recorded=bool(recorded),
    )


def closed(meta: SessionMeta) -> bool:
    """Return whether the debate is over (its synthesis written)."""
    return meta.runtime.phase is Phase.DONE


def facts(report: Report, labels: dict[str, str]) -> list[tuple[str, str]]:
    """Return the header's label/value pairs: council, date, language, rounds, status."""
    meta = report.meta
    status = labels["closed"] if closed(meta) else labels["in_progress"]
    return [
        (labels["council"], meta.council),
        (labels["started"], f"{meta.created_at.astimezone():%Y-%m-%d %H:%M}"),
        (labels["language"], labels[f"lang_{meta.language}"]),
        (labels["rounds"], str(meta.runtime.round)),
        (labels["status"], status),
    ]


def model_cell(seat: Seat) -> str:
    """Return a seat's model(s), "a → b" when it changed during the session."""
    return " → ".join(seat.models) or "?"


def vote_rows(
    report: Report, vote: VoteTaken, labels: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Return (nick, confidence, position) rows of a vote."""
    rows = []
    for v in vote.votes:
        if v.error is not None:
            rows.append((report.nick(v.speaker), "—", f"{labels['no_vote']} {v.error}"))
        else:
            rows.append((report.nick(v.speaker), f"{v.confidence}%", v.position))
    return rows


def flaws_by_member(report: Report) -> list[tuple[str, list[Flaw]]]:
    """Group the report's flaws by member nick, in seat order."""
    groups = [(s.nick, [f for f in report.flaws if f.member == s.slug]) for s in report.seats]
    return [(nick, flaws) for nick, flaws in groups if flaws]


def flaw_text(flaw: Flaw) -> str:
    """Describe one flaw on one line, as `/flaws` does."""
    if flaw.by == "engine" and flaw.note:  # the engine's note is the figure in question
        what = f'{flaw.note} in "{flaw.excerpt}"'
    else:
        what = f'"{flaw.excerpt}"' + (f": {flaw.note}" if flaw.note else "")
    return f"r{flaw.round} [{flaw.kind.value}, {flaw.by}] {what}"
