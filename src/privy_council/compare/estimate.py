"""Before a comparison runs: how many Claude calls, how long (ceilings, not promises).

Turn times come from past sessions: member turns from their transcripts (every
backend), moderator calls from the usage ledger (Claude only). Without history,
the defaults below stand in.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from privy_council.compare.metrics import seat_times
from privy_council.engine.budget import Ledger
from privy_council.engine.research import ResearchMode, searches_first
from privy_council.export.report import ExportError, build_report
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona
from privy_council.storage.sessions import SessionStore, list_sessions

HISTORY_SESSIONS = 40  # newest sessions scanned for member turn times
DEFAULT_TURN_S = {BackendKind.OLLAMA: 120.0, BackendKind.CLAUDE_CLI: 30.0}
DEFAULT_MODERATOR_S = 45.0  # one moderator call: opening, update or synthesis
# A local moderator (local-only mode): Super took 4-5 min per call in the M15 bench.
# Ollama calls aren't in the ledger, so there is no history to take it from.
DEFAULT_LOCAL_MODERATOR_S = 280.0
RESEARCH_S = 120.0


@dataclass(frozen=True)
class VariantEstimate:
    """Ceilings for one variant's debate."""

    claude_calls: int
    seconds: float


def seat(persona: Persona) -> str:
    """Return a persona's "backend:model" key."""
    return f"{persona.meta.backend.value}:{persona.meta.model}"


def turn_history(data_dir: Path, personas: Mapping[str, Persona]) -> dict[str, float]:
    """Return the median member turn time per "backend:model" over recent sessions."""
    samples: dict[str, list[float]] = {}
    for info in list_sessions(data_dir)[:HISTORY_SESSIONS]:
        try:
            report = build_report(info.directory, personas, "")
            events = SessionStore(info.directory).events()
        except (OSError, ValueError, ExportError):
            continue
        for times in seat_times(events, report):
            # Sessions before M13 didn't record models: today's persona file may differ.
            if times.seat.recorded and len(times.seat.models) == 1:
                samples.setdefault(times.seat.models[0], []).extend([times.median_s] * times.turns)
    return {key: statistics.median(values) for key, values in samples.items() if values}


def moderator_history(ledger: Ledger | None, moderator: Persona) -> float:
    """Return the median latency of past moderator calls on the moderator's model."""
    if moderator.meta.backend is not BackendKind.CLAUDE_CLI:
        return DEFAULT_LOCAL_MODERATOR_S
    model = moderator.meta.model
    purposes = {"opening", "moderate", "synthesis"}
    entries = ledger.entries() if ledger else []
    values = [e.latency_s for e in entries if e.model == model and e.purpose in purposes and e.ok]
    return statistics.median(values) if values else DEFAULT_MODERATOR_S


def estimate_variant(
    council: CouncilPreset,
    personas: Mapping[str, Persona],
    *,
    rounds: int,
    research: ResearchMode,
    history: Mapping[str, float],
    moderator_s: float,
) -> VariantEstimate:
    """Estimate one variant: every member speaks every round, then votes."""
    members = [personas[slug] for slug in council.members]
    moderator = personas[council.moderator]
    on_claude = moderator.meta.backend is BackendKind.CLAUDE_CLI
    searches = searches_first(research) and on_claude
    claude_members = sum(p.meta.backend is BackendKind.CLAUDE_CLI for p in members)
    # Opening, one update per round, synthesis; members: one turn per round and the vote.
    calls = (rounds + 2) * on_claude + searches + claude_members * (rounds + 1)
    member_s = sum(_turn_s(p, history) for p in members)
    seconds = (rounds + 2) * moderator_s + member_s * (rounds + 1) + RESEARCH_S * searches
    return VariantEstimate(claude_calls=calls, seconds=seconds)


def _turn_s(persona: Persona, history: Mapping[str, float]) -> float:
    """Return a member's expected turn time."""
    return history.get(seat(persona), DEFAULT_TURN_S[persona.meta.backend])


def total_calls(estimates: Iterable[VariantEstimate], judge: bool) -> int:
    """Return the comparison's Claude calls, the judge included."""
    return sum(e.claude_calls for e in estimates) + judge
