from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.backends.base import (
    BackendRateLimitError,
    ChatMessage,
    CompletionRequest,
    CompletionResult,
)
from privy_council.backends.claude_cli import ClaudeCliBackend
from privy_council.backends.fake import FakeBackend
from privy_council.backends.ollama import OllamaBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.loaders.personas import load_personas
from privy_council.models.persona import BackendKind, Persona
from tests.conftest import REPO_ROOT, repo_settings


def request(model: str = "m") -> CompletionRequest:
    """Build a minimal request."""
    return CompletionRequest(
        system="s", messages=[ChatMessage(role="user", content="q")], model=model
    )


async def test_fake_follows_script_then_default() -> None:
    fake = FakeBackend(["one", {"stop": True}, BackendRateLimitError("limit")], default="dflt")
    assert (await fake.complete(request())).text == "one"
    structured = await fake.complete(request())
    assert structured.structured == {"stop": True}
    assert structured.text == '{"stop": true}'
    with pytest.raises(BackendRateLimitError):
        await fake.complete(request())
    assert (await fake.complete(request())).text == "dflt"
    assert len(fake.requests) == 4


async def test_fake_responder_and_strict_mode() -> None:
    echo = FakeBackend(responder=lambda req: f"model={req.model}")
    assert (await echo.complete(request("x"))).text == "model=x"
    strict = FakeBackend(default=None)
    with pytest.raises(Exception, match="script exhausted"):
        await strict.complete(request())


@pytest.fixture
def personas() -> dict[str, Persona]:
    """The repo's example personas."""
    loaded, _ = load_personas(REPO_ROOT / "personas")
    return loaded


def _settings(tmp_path: Path, fallback: bool) -> Settings:
    """Settings with fallback_to_local toggled, rooted at the repo."""
    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"[claude]\nfallback_to_local = {str(fallback).lower()}\nfallback_model = 'qwen3:4b'\n",
        encoding="utf-8",
    )
    return load_settings(tmp_path)


async def test_registry_routes_by_backend(tmp_path: Path, personas: dict[str, Persona]) -> None:
    registry = BackendRegistry(repo_settings())
    chair = registry.route(personas["chair"])
    rigor = registry.route(personas["rigor"])
    assert isinstance(chair.backend, ClaudeCliBackend) and chair.model == "sonnet"
    assert isinstance(rigor.backend, OllamaBackend) and rigor.model == "nemotron-3.5-super:latest"
    assert registry.route(personas["pragmatist"]).backend is rigor.backend  # shared instance
    await registry.aclose()


def test_fallback_only_when_enabled_and_exhausted(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    fake_claude, fake_ollama = FakeBackend(name="c"), FakeBackend(name="o")
    overrides = {BackendKind.CLAUDE_CLI: fake_claude, BackendKind.OLLAMA: fake_ollama}
    on = BackendRegistry(_settings(tmp_path, True), overrides=overrides)
    route = on.route(personas["chair"], claude_exhausted=True)
    assert (route.backend, route.model, route.fallback) == (fake_ollama, "qwen3:4b", True)
    assert on.route(personas["chair"]).backend is fake_claude
    off = BackendRegistry(_settings(tmp_path, False), overrides=overrides)
    assert off.route(personas["chair"], claude_exhausted=True).backend is fake_claude


def test_required_models_include_fallback(tmp_path: Path, personas: dict[str, Persona]) -> None:
    local = {"qwen3.6:27b", "gemma4:26b", "gpt-oss:120b", "nemotron-3.5-super:latest"}
    on = BackendRegistry(_settings(tmp_path, True))
    assert on.required_ollama_models(personas.values()) == local | {"qwen3:4b"}
    off = BackendRegistry(_settings(tmp_path, False))
    assert off.required_ollama_models(personas.values()) == local


async def test_fake_returns_scripted_result_as_is() -> None:
    cut = CompletionResult(text="partial", latency_s=0, truncated=True, backend="fake", model="m")
    assert (await FakeBackend([cut]).complete(request())) is cut
