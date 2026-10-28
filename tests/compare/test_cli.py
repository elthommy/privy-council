"""`council compare ...` through the CLI, on fake backends."""

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
from tests.compare.test_runner import respond
from tests.conftest import REPO_ROOT

QUESTION = "Should I max my PEA before building a bigger emergency fund?"


@pytest.fixture
def fake() -> FakeBackend:
    """One fake answering every backend."""
    return FakeBackend(responder=respond)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeBackend) -> Path:
    """A project root using the repo's personas and councils, with fake backends."""
    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose backends are both the fake."""
        overrides = {BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)

    monkeypatch.setattr(compare_cli, "BackendRegistry", registry)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


def council(root: Path, *args: str, input: str | None = None) -> Result:
    """Run `council --root <root> ...`."""
    return CliRunner().invoke(cli.app, ["--root", str(root), *args], input=input)


def compares(root: Path) -> list[CompareStore]:
    """Return the saved comparisons."""
    base = root / "data" / COMPARE_DIR
    return [CompareStore(d) for d in base.iterdir()] if base.is_dir() else []


def test_new_runs_judges_and_writes_the_report(root: Path, fake: FakeBackend) -> None:
    args = ["compare", "new", "finance", QUESTION, "-v", "base", "-v", "q:rigor=qwen3.6:27b"]
    result = council(root, *args, "--rounds", "1", "--judge", "--yes")

    assert result.exit_code == 0, result.output
    assert "base" in result.output and "Claude calls" in result.output
    assert "(judge included)" in result.output
    assert "Judge: comparing the syntheses" in result.output
    [store] = compares(root)
    state = store.load()
    assert [v.status for v in state.variants] == [VariantStatus.DONE] * 2
    assert state.judged and store.load_judge()
    assert (store.directory / "report.md").is_file()
    assert (store.directory / "report.html").is_file()
    assert f"Report: {store.directory / 'report.md'}" in result.output
    assert [r.tag.purpose for r in fake.requests][-1] == "compare_judge"

    listed = council(root, "compare", "list")
    assert f"{store.id}  2/2 variants done" in listed.output
    again = council(root, "compare", "resume")
    assert again.exit_code == 0 and "Every variant is done" in again.output
    assert [r.tag.purpose for r in fake.requests].count("compare_judge") == 1


def test_declining_the_estimate_runs_nothing(root: Path, fake: FakeBackend) -> None:
    args = ["compare", "new", "finance", QUESTION, "-v", "a", "-v", "b:-contrarian"]
    result = council(root, *args, input="n\n")
    assert result.exit_code == 0
    assert "Run it?" in result.output
    assert compares(root) == [] and fake.requests == []


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        (["-v", "only"], "at least two --variant"),
        (["-v", "a", "-v", "b:realist=qwen"], "'realist' has no seat"),
        (["-v", "a", "-v", "B"], "must be lowercase"),
    ],
)
def test_bad_variants_exit_before_any_call(
    root: Path, fake: FakeBackend, variants: list[str], message: str
) -> None:
    result = council(root, "compare", "new", "finance", QUESTION, *variants, "--yes")
    assert result.exit_code == 1
    assert message in result.output
    assert fake.requests == []


def test_required_research_refuses_no_research(root: Path) -> None:
    args = ["compare", "new", "finance", QUESTION, "-v", "a", "-v", "b:-contrarian"]
    result = council(root, *args, "--no-research", "--yes")
    assert result.exit_code == 1 and "requires web research" in result.output


def test_lookups_without_comparisons(root: Path) -> None:
    assert "No comparisons yet." in council(root, "compare", "list").output
    result = council(root, "compare", "report")
    assert result.exit_code == 1 and "no comparisons yet" in result.output
