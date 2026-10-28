"""Opt-in live checks against the real backends: `uv run pytest -m live`.

The claude tests spend two small haiku calls and one sonnet call of Pro quota;
the ollama test is free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.backends.base import ChatMessage, CompletionRequest
from privy_council.backends.claude_cli import ClaudeCliBackend
from privy_council.backends.ollama import OllamaBackend
from privy_council.config import load_settings
from privy_council.engine.budget import Ledger
from privy_council.models.moderator import ModeratorDecision
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.live

PROMPT = [ChatMessage(role="user", content="Reply with exactly one word: pong")]


async def test_live_ollama() -> None:
    backend = OllamaBackend(load_settings(REPO_ROOT).ollama)
    try:
        req = CompletionRequest(system="You are terse.", messages=PROMPT, model="qwen3:8b")
        result = await backend.complete(req)
    finally:
        await backend.aclose()
    assert "pong" in result.text.lower()
    assert result.usage is not None and result.usage.output_tokens > 0


async def test_live_claude_cli(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    backend = ClaudeCliBackend(load_settings(REPO_ROOT).claude, ledger)
    req = CompletionRequest(system="You are terse.", messages=PROMPT, model="haiku")
    result = await backend.complete(req)
    assert "pong" in result.text.lower()
    assert result.model.startswith("claude-")
    assert not result.truncated
    [entry] = ledger.entries()
    assert entry.quota is not None and entry.quota.five_hour is not None


async def test_live_claude_hard_cap() -> None:
    backend = ClaudeCliBackend(load_settings(REPO_ROOT).claude)
    essay = [ChatMessage(role="user", content="Write a 600-word essay about rivers.")]
    req = CompletionRequest(
        system="You are a verbose essayist.", messages=essay, model="haiku", max_output_tokens=50
    )
    result = await backend.complete(req)
    assert result.truncated
    assert 200 <= len(result.text) < 400  # cap is ~200 chars; cut within one delta


async def test_live_claude_moderator_schema() -> None:
    backend = ClaudeCliBackend(load_settings(REPO_ROOT).claude)
    prompt = (
        "Round 1 answers on 'Max my PEA before a bigger emergency fund?':\n"
        "rigor: No, build 6 months of cash first. Confidence 80.\n"
        "compounder: Yes, time in market matters more. Confidence 70.\n"
        "Produce the debate state and pick who speaks next and on what."
    )
    req = CompletionRequest(
        system="You are Chair, a neutral debate moderator. Never give your own opinion.",
        messages=[ChatMessage(role="user", content=prompt)],
        model="sonnet",
        json_schema=ModeratorDecision.model_json_schema(),
        max_output_tokens=4000,
    )
    result = await backend.complete(req)
    assert result.structured is not None, result.text
    decision = ModeratorDecision.model_validate(result.structured)
    assert decision.next_turns or decision.stop
