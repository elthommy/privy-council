"""What a comparison report shows per variant, read from saved files (no model calls)."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from privy_council.compare.store import VariantRun, VariantStatus
from privy_council.engine.budget import Ledger
from privy_council.export.report import Report, Seat, build_report
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    Event,
    MessageEvent,
    StateUpdated,
    TurnStarted,
    VoteTaken,
)
from privy_council.storage.sessions import SessionStore

# A longer silence between two events is a stop (or a crash) before a resume, not work.
IDLE_GAP = timedelta(minutes=30)


class SeatTimes(BaseModel):
    """How long one member's turns took."""

    model_config = ConfigDict(frozen=True)

    seat: Seat
    turns: int
    median_s: float
    max_s: float


class VariantMetrics(BaseModel):
    """One variant, summarized for the comparison report."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str  # the changes, as typed
    status: VariantStatus
    error: str | None = None
    report: Report | None = None  # None until its session exists
    claude_calls: int = 0
    run_s: float = 0.0
    times: list[SeatTimes] = []
    consensus: list[str] = []  # the moderator's call after each round

    @property
    def rounds(self) -> int:
        """Return the rounds debated."""
        return self.report.meta.runtime.round if self.report else 0

    @property
    def vote(self) -> VoteTaken | None:
        """Return the final vote, if taken."""
        return self.report.final_vote if self.report else None

    def confidence(self) -> tuple[int, float, int] | None:
        """Return the final vote's (min, mean, max) confidence, or None without votes."""
        vote = self.vote
        values = [v.confidence for v in vote.votes if v.confidence is not None] if vote else []
        if not values:
            return None
        return min(values), statistics.fmean(values), max(values)

    def flaw_counts(self) -> list[tuple[str, int]]:
        """Return (member nick, flaws caught) for every member, in seat order."""
        if self.report is None:
            return []
        report = self.report
        members = [s for s in report.seats if s.slug != report.meta.moderator]
        return [(s.nick, sum(f.member == s.slug for f in report.flaws)) for s in members]


def variant_metrics(
    run: VariantRun,
    sessions_dir: Path,
    personas: Mapping[str, Persona],
    user_name: str,
    ledger: Ledger | None,
) -> VariantMetrics:
    """Summarize one variant from its session directory and the usage ledger."""
    base = VariantMetrics(
        name=run.variant.name, label=run.variant.label(), status=run.status, error=run.error
    )
    if run.session is None or not (sessions_dir / run.session).is_dir():
        return base
    directory = sessions_dir / run.session
    report = build_report(directory, personas, user_name)
    events = SessionStore(directory).events()
    calls = [
        e
        for e in (ledger.entries() if ledger else [])
        if e.session == report.meta.id and e.backend == BackendKind.CLAUDE_CLI.value
    ]
    return base.model_copy(
        update={
            "report": report,
            "claude_calls": len(calls),
            "run_s": run_seconds(events),
            "times": seat_times(events, report),
            "consensus": [
                e.consensus.value for e in events if isinstance(e, StateUpdated) and e.consensus
            ],
        }
    )


def run_seconds(events: list[Event]) -> float:
    """Return the time the debate ran, leaving out idle gaps (stops before a resume)."""
    stamps = [e.ts for e in events]
    gaps = (b - a for a, b in itertools.pairwise(stamps))
    return sum((g for g in gaps if g <= IDLE_GAP), timedelta()).total_seconds()


def seat_times(events: list[Event], report: Report) -> list[SeatTimes]:
    """Return each member's turn times: from its TurnStarted to its message.

    A vote, or another start by the same speaker, before the message means the
    start wasn't a chat turn (a vote, a memory note), so it isn't timed.
    """
    started: dict[str, datetime] = {}
    durations: dict[str, list[float]] = {}
    for event in events:
        if isinstance(event, TurnStarted):
            started[event.speaker] = event.ts
        elif isinstance(event, VoteTaken):
            started.clear()
        elif isinstance(event, MessageEvent):
            speaker = event.message.speaker
            if speaker in started and not event.message.to_user:
                seconds = (event.ts - started.pop(speaker)).total_seconds()
                durations.setdefault(speaker, []).append(seconds)
    moderator = report.meta.moderator
    return [
        SeatTimes(
            seat=seat,
            turns=len(durations[seat.slug]),
            median_s=statistics.median(durations[seat.slug]),
            max_s=max(durations[seat.slug]),
        )
        for seat in report.seats
        if seat.slug != moderator and durations.get(seat.slug)
    ]
