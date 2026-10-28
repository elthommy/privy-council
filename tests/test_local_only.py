from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.backends.base import CompletionRequest, LocalOnlyError
from privy_council.backends.fake import FakeBackend, Reply
from privy_council.backends.registry import BackendRegistry
from privy_council.compare import estimate
from privy_council.config import Settings, load_settings
from privy_council.engine.budget import Ledger, UsageEntry
from privy_council.engine.research import ResearchMode, ResearchRequiredError, resolve_mode
from privy_council.engine.resume import ResumeError, check_resumable
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import (
    LocalOnlyConfigError,
    apply_local_only,
    local_personas,
    local_settings,
)
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import SessionMeta, SessionState
from tests.conftest import REPO_ROOT, repo_settings
from tests.engine.test_memory import memory_responder
from tests.engine.test_protocol import Harness

SUPER = "nemotron-3.5-super:latest"


@pytest.fixture
def personas() -> dict[str, Persona]:
    """The repo's example personas."""
    loaded, _ = load_personas(REPO_ROOT / "personas")
    return loaded


@pytest.fixture
def settings() -> Settings:
    """The repo's config.toml, seat map included."""
    return repo_settings()


def test_every_claude_seat_moves_to_its_local_model(
    settings: Settings, personas: dict[str, Persona]
) -> None:
    seated = local_personas(settings, personas, {"chair"})
    assert not [p.slug for p in seated.values() if p.meta.backend is BackendKind.CLAUDE_CLI]
    assert seated["chair"].meta.model == SUPER
    assert seated["chair"].meta.synthesis_model is None  # opus belongs to the CLI
    assert seated["forecaster"].meta.model == SUPER  # [local_only.seats]
    assert seated["kindness"].meta.model == "qwen3.6:27b"
    assert seated["compounder"].meta.model == "gpt-oss:120b"  # default_model
    assert seated["rigor"] is personas["rigor"]  # local seats stay as they are
    assert personas["chair"].meta.backend is BackendKind.CLAUDE_CLI  # copies only


@pytest.mark.parametrize(
    ("seats", "problem"),
    [
        ({"ghost": "qwen3.6:27b"}, "[local_only.seats] ghost: no such persona"),
        ({"rigor": "qwen3.6:27b"}, "[local_only.seats] rigor: already a local seat"),
    ],
)
def test_seat_map_errors_name_the_entry(
    settings: Settings, personas: dict[str, Persona], seats: dict[str, str], problem: str
) -> None:
    cfg = settings.local_only.model_copy(update={"seats": seats})
    bad = settings.model_copy(update={"local_only": cfg})
    with pytest.raises(LocalOnlyConfigError, match=problem.replace("[", r"\[")):
        local_personas(bad, personas, {"chair"})


def test_moderator_model_gets_the_chair_context_for_every_seat(settings: Settings) -> None:
    local = local_settings(settings)
    assert local.ollama.models[SUPER].num_ctx == 24576
    assert local.ollama.models[SUPER].think is False  # other options kept
    assert local.ollama.models["qwen3.6:27b"].num_ctx == 12288
    assert settings.ollama.models[SUPER].num_ctx == 12288  # a copy


def test_moderator_model_missing_from_config_gets_an_entry(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[local_only]\nmoderator_model = "new:1b"\nmoderator_num_ctx = 4096\n', encoding="utf-8"
    )
    assert local_settings(load_settings(tmp_path)).ollama.models["new:1b"].num_ctx == 4096


def test_registry_never_hands_out_claude_in_local_only_mode(
    settings: Settings, personas: dict[str, Persona]
) -> None:
    fake = FakeBackend(name="claude")
    registry = BackendRegistry(settings, overrides={BackendKind.CLAUDE_CLI: fake}, local_only=True)
    with pytest.raises(LocalOnlyError):
        registry.backend(BackendKind.CLAUDE_CLI)
    with pytest.raises(LocalOnlyError):
        registry.route(personas["chair"])  # an unmapped Claude persona
    with pytest.raises(LocalOnlyError):
        registry.route(personas["compounder"], claude_exhausted=True)  # no fallback either
    assert registry.route(personas["rigor"]).kind is BackendKind.OLLAMA
    assert not fake.requests


def test_local_only_registry_needs_no_fallback_model(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[claude]\nfallback_to_local = true\nfallback_model = "qwen3:8b"\n', encoding="utf-8"
    )
    settings = load_settings(tmp_path)
    assert BackendRegistry(settings, local_only=True).required_ollama_models([]) == set()
    assert BackendRegistry(settings).required_ollama_models([]) == {"qwen3:8b"}


def test_research_is_off_and_research_flag_refused(settings: Settings) -> None:
    councils, _ = load_councils(REPO_ROOT / "councils", load_personas(REPO_ROOT / "personas")[0])
    finance = councils["finance"]
    assert finance.protocol.require_research
    assert resolve_mode(settings, finance, None, local_only=True) is ResearchMode.OFF
    assert resolve_mode(settings, finance, False, local_only=True) is ResearchMode.OFF
    with pytest.raises(ResearchRequiredError, match="local-only"):
        resolve_mode(settings, finance, True, local_only=True)


@pytest.mark.parametrize(("stored", "registry"), [(True, False), (False, True)])
def test_resume_keeps_the_mode(stored: bool, registry: bool, personas: dict[str, Persona]) -> None:
    meta = SessionMeta(
        id="s", council="finance", topic="t", moderator="chair", members=["rigor"],
        local_only=stored,
    )  # fmt: skip
    with pytest.raises(ResumeError, match="resume it in that mode"):
        check_resumable(meta, personas, local_only=registry)
    check_resumable(meta, personas, local_only=stored)


def everything(req: CompletionRequest) -> Reply:
    """Answer every call purpose a session can make (memory, vote, summary, dossier)."""
    if req.tag.purpose == "vote":
        return {"position": "fund first", "confidence": 70}
    if req.tag.purpose in ("summary", "dossier"):
        return "## Summary\nKeep a buffer."
    return RESPOND(req)


RESPOND = memory_responder()


async def test_a_full_local_only_session_makes_no_claude_call(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        everything,
        local_only=True,
        research=ResearchMode.OFF,
        config={"dossier": {"summarize_over_tokens": 50}},
    )
    notes = tmp_path / "notes.txt"
    notes.write_text("word " * 200, encoding="utf-8")
    await h.engine.attach(str(notes))  # summarized by the moderator
    await h.engine.step()  # opening
    await h.engine.step()  # blind round
    await h.engine.vote()
    await h.engine.summary()
    h.engine.say("I rent a flat in Lyon")
    await h.engine.close()  # synthesis, private notes, shared facts

    assert h.state() is SessionState.CLOSED
    assert not h.claude.requests
    purposes = {r.tag.purpose for r in h.ollama.requests}
    expected = {"dossier", "opening", "turn", "moderate", "vote", "summary", "synthesis"}
    assert expected | {"memory", "shared_memory"} <= purposes
    meta = h.store.load_meta()
    assert meta.local_only
    assert meta.models["chair"] == [f"ollama:{SUPER}"]
    assert meta.models["compounder"] == ["ollama:gpt-oss:120b"]
    system = [m.text for m in h.engine.messages if m.speaker == "system"]
    assert any(t.startswith("Local-only mode:") for t in system)


async def test_normal_session_has_no_local_only_notice(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, RESPOND)
    await h.engine.step()
    assert not h.engine.meta.local_only
    assert not [m for m in h.engine.messages if m.text.startswith("Local-only mode:")]


def test_apply_local_only_returns_both(settings: Settings, personas: dict[str, Persona]) -> None:
    local, seated = apply_local_only(settings, personas, {"chair"})
    assert local.ollama.models[SUPER].num_ctx == 24576
    assert seated["chair"].meta.backend is BackendKind.OLLAMA


def test_local_moderator_estimate_uses_the_local_default(
    settings: Settings, personas: dict[str, Persona], tmp_path: Path
) -> None:
    ledger = Ledger.in_data_dir(tmp_path)
    ledger.record(UsageEntry(model="sonnet", purpose="moderate", latency_s=20.0))
    assert estimate.moderator_history(ledger, personas["chair"]) == 20.0
    seated = local_personas(settings, personas, {"chair"})
    local = estimate.moderator_history(ledger, seated["chair"])
    assert local == estimate.DEFAULT_LOCAL_MODERATOR_S
