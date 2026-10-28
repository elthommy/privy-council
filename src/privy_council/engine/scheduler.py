"""GPU-aware turn ordering for rounds whose turns are independent (the blind round)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from privy_council.models.persona import BackendKind


@dataclass
class Schedule[T]:
    """Turns split by how they may run: Claude concurrently, Ollama one after another."""

    concurrent: list[T] = field(default_factory=list)
    sequential: list[T] = field(default_factory=list)


def schedule[T](
    turns: Sequence[T], placement: Callable[[T], tuple[BackendKind, str]]
) -> Schedule[T]:
    """Split turns; Ollama turns are grouped by model to minimize GPU model swaps.

    `placement` returns (backend kind, model) for a turn. Groups keep the order
    in which each model first appears, and turns keep their order within a group.
    """
    plan: Schedule[T] = Schedule()
    by_model: dict[str, list[T]] = {}
    for turn in turns:
        kind, model = placement(turn)
        if kind is BackendKind.OLLAMA:
            by_model.setdefault(model, []).append(turn)
        else:
            plan.concurrent.append(turn)
    plan.sequential = [turn for group in by_model.values() for turn in group]
    return plan
