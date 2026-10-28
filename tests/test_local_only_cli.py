from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from privy_council import cli
from privy_council.backends.fake import FakeBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.compare import cli as compare_cli
from privy_council.compare.store import COMPARE_DIR, CompareStore, VariantStatus
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.models.persona import BackendKind
from privy_council.storage.sessions import SessionStore
from tests.compare.test_runner import respond
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import standard_responder

QUESTION = "Should I max my PEA before building a bigger emergency fund?"


class Fakes:
    """Separate Claude and Ollama fakes, and the local_only flag of every registry built."""

    def __init__(self) -> None:
        """Start with no call and no registry."""
        self.claude = FakeBackend(responder=respond, name="claude")
        self.ollama = FakeBackend(responder=respond, name="ollama")
        self.modes: list[bool] = []

    def registry(
        self, settings: Settings, ledger: Ledger, local_only: bool = False
    ) -> BackendRegistry:
        """Build a registry on the fakes, noting its mode."""
        self.modes.append(local_only)
        overrides = {BackendKind.CLAUDE_CLI: self.claude, BackendKind.OLLAMA: self.ollama}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Install the fakes in both CLIs."""
    fakes = Fakes()
    monkeypatch.setattr(cli, "BackendRegistry", fakes.registry)
    monkeypatch.setattr(compare_cli, "BackendRegistry", fakes.registry)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return fakes


def make_root(tmp_path: Path, extra: str = "") -> Path:
    """Write a config on the repo's personas and councils."""
    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n{extra}",
        encoding="utf-8",
    )
    return tmp_path


def council(root: Path, *args: str, input: str | None = None) -> Result:
    """Run `council --root <root> ...`."""
    return CliRunner().invoke(cli.app, ["--root", str(root), *args], input=input)


def only_session(root: Path) -> SessionStore:
    """Return the one saved session."""
    base = root / "data" / "sessions"
    [directory] = [d for d in base.iterdir() if not d.is_symlink()]
    return SessionStore.open(directory, {})


def test_new_local_only_then_resume_never_call_claude(tmp_path: Path, fakes: Fakes) -> None:
    fakes.ollama = FakeBackend(responder=standard_responder([]), name="ollama")
    root = make_root(tmp_path)
    result = council(root, "new", "finance", QUESTION, "--no-tui", "--local-only", input="")
    assert result.exit_code == 0, result.output
    assert "[local-only]" in result.output
    assert "Local-only mode:" in result.output
    store = only_session(root)
    assert store.load_meta().local_only

    result = council(root, "resume", "--no-tui", input="/close\n")
    assert result.exit_code == 0, result.output
    assert fakes.modes == [True, True]  # resume took the mode from session.json
    assert not fakes.claude.requests
    assert (store.directory / "synthesis.md").is_file()


def test_config_enabled_is_the_default_and_the_flag_overrides_it(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.ollama = FakeBackend(responder=standard_responder([]), name="ollama")
    fakes.claude = FakeBackend(responder=standard_responder([]), name="claude")
    root = make_root(tmp_path, "[local_only]\nenabled = true\n")
    assert (
        council(root, "new", "salon", "What is a good life?", "--no-tui", input="").exit_code == 0
    )
    args = ["new", "salon", "Is courage a virtue?", "--no-tui", "--no-local-only"]
    assert council(root, *args, input="").exit_code == 0
    assert fakes.modes == [True, False]


@pytest.mark.usefixtures("fakes")
def test_local_only_refuses_research_flag(tmp_path: Path) -> None:
    result = council(make_root(tmp_path), "new", "finance", QUESTION, "--local-only", "--research")
    assert result.exit_code == 1
    assert "--research in local-only mode needs local search" in result.output


@pytest.mark.usefixtures("fakes")
def test_bad_seat_map_is_a_config_error(tmp_path: Path) -> None:
    root = make_root(tmp_path, '[local_only.seats]\nghost = "qwen3.6:27b"\n')
    result = council(root, "new", "finance", QUESTION, "--local-only", "--no-tui")
    assert result.exit_code == 2
    assert "[local_only.seats] ghost: no such persona" in result.output


def test_personas_new_refused_when_local_only_is_enabled(tmp_path: Path, fakes: Fakes) -> None:
    root = make_root(tmp_path, "[local_only]\nenabled = true\n")
    result = council(root, "personas", "new", "a tax adviser")
    assert result.exit_code == 1
    assert "not available with [local_only] enabled" in result.output
    assert not fakes.claude.requests


@pytest.mark.usefixtures("fakes")
def test_compare_local_only_refuses_the_judge(tmp_path: Path) -> None:
    args = ["compare", "new", "finance", QUESTION, "-v", "a", "-v", "b:+skeptic"]
    result = council(make_root(tmp_path), *args, "--local-only", "--judge", "--yes")
    assert result.exit_code == 1
    assert "--judge is one Claude call" in result.output


@pytest.mark.usefixtures("fakes")
def test_compare_local_only_refuses_a_claude_seat(tmp_path: Path) -> None:
    args = ["compare", "new", "finance", QUESTION, "-v", "a", "-v", "b:chair=claude_cli:sonnet"]
    result = council(make_root(tmp_path), *args, "--local-only", "--yes")
    assert result.exit_code == 1
    assert "seats chair on Claude: not in local-only mode" in result.output


def test_compare_local_only_runs_without_claude(tmp_path: Path, fakes: Fakes) -> None:
    root = make_root(tmp_path)
    args = ["compare", "new", "finance", QUESTION, "-v", "base", "-v", "q:rigor=qwen3.6:27b"]
    result = council(root, *args, "--rounds", "1", "--local-only", "--yes")
    assert result.exit_code == 0, result.output
    assert "≤  0 Claude calls" in result.output
    [directory] = list((root / "data" / COMPARE_DIR).iterdir())
    state = CompareStore(directory).load()
    assert state.local_only and state.research == "off"
    assert [v.status for v in state.variants] == [VariantStatus.DONE] * 2
    assert not fakes.claude.requests
    assert set(fakes.modes) == {True}
