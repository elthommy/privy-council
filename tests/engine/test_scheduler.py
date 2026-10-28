from __future__ import annotations

from privy_council.engine.scheduler import schedule
from privy_council.models.persona import BackendKind

PLACES = {
    "a": (BackendKind.OLLAMA, "qwen3:8b"),
    "b": (BackendKind.CLAUDE_CLI, "haiku"),
    "c": (BackendKind.OLLAMA, "gemma3"),
    "d": (BackendKind.OLLAMA, "qwen3:8b"),
    "e": (BackendKind.CLAUDE_CLI, "sonnet"),
}


def test_claude_concurrent_ollama_grouped_by_model() -> None:
    plan = schedule(list("abcde"), PLACES.__getitem__)
    assert plan.concurrent == ["b", "e"]
    assert plan.sequential == ["a", "d", "c"]  # qwen3 group first: it appeared first


def test_empty() -> None:
    plan = schedule(list[str](), PLACES.__getitem__)
    assert plan.concurrent == [] and plan.sequential == []
