"""`council resume` and `council sessions` through the CLI, on fake backends."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from privy_council import cli
from privy_council.backends.fake import FakeBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.models.persona import BackendKind
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import decision, standard_responder


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root using the repo's personas and councils, with fake backends."""
    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )
    fake = FakeBackend(responder=standard_responder([decision(stop=True, reason="agreed")] * 5))

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose backends are both the fake."""
        overrides = {BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)

    monkeypatch.setattr(cli, "BackendRegistry", registry)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


def council(root: Path, *args: str, input: str | None = None) -> Result:
    """Run `council --root <root> ...`."""
    return CliRunner().invoke(cli.app, ["--root", str(root), *args], input=input)


def session_dirs(root: Path) -> list[Path]:
    """Return the saved session directories."""
    base = root / "data" / "sessions"
    return [d for d in base.iterdir() if not d.is_symlink()]


def test_quit_list_show_resume_close(root: Path) -> None:
    started = council(root, "new", "finance", "Rent or buy?", "--no-tui", input="/quit\n")
    assert started.exit_code == 0, started.output
    [directory] = session_dirs(root)
    assert f"Resume with: council resume {directory.name}" in started.output

    listed = council(root, "sessions", "list")
    assert listed.exit_code == 0, listed.output
    assert f"{directory.name}  " in listed.output
    assert "paused · after round 0" in listed.output  # opening done, round 1 next

    shown = council(root, "sessions", "show", "rent-or-buy")
    assert shown.exit_code == 0, shown.output
    assert "topic: Rent or buy?" in shown.output
    assert f"resume: council resume {directory.name}" in shown.output

    resumed = council(root, "resume", "--no-tui", input="/step\n/close\n")
    assert resumed.exit_code == 0, resumed.output
    assert "<Chair> Restating" in resumed.output  # the replayed opening
    assert "* Resumed after round 0." in resumed.output
    assert "-- Round 1 — blind takes --" in resumed.output
    assert "<Chair> ## Consensus" in resumed.output
    assert "Resume with" not in resumed.output  # closed now

    again = council(root, "resume", directory.name, "--no-tui")
    assert again.exit_code == 1
    assert "is closed; its synthesis is in synthesis.md" in again.output


def test_resume_errors(root: Path) -> None:
    none = council(root, "resume")
    assert none.exit_code == 1
    assert "No saved sessions" in none.output
    council(root, "new", "finance", "Rent or buy?", "--no-tui", input="/quit\n")
    unknown = council(root, "resume", "nope")
    assert unknown.exit_code == 1
    assert "No session matches 'nope'" in unknown.output


def test_prune(root: Path) -> None:
    council(root, "new", "finance", "Rent or buy?", "--no-tui", input="/quit\n")
    council(root, "new", "finance", "PEA first?", "--no-tui", input="/quit\n")
    old = next(d for d in session_dirs(root) if d.name.endswith("rent-or-buy"))
    for path in [old, *old.iterdir()]:
        os.utime(path, (0, 0))

    bad = council(root, "sessions", "prune", "--older-than", "soon")
    assert bad.exit_code == 2
    kept = council(root, "sessions", "prune", "--older-than", "90d", input="n\n")
    assert kept.exit_code == 1 and old.exists()
    assert "Delete these 1 session(s)?" in kept.output

    pruned = council(root, "sessions", "prune", "--older-than", "12w", "--yes")
    assert pruned.exit_code == 0, pruned.output
    assert "Deleted 1 session(s)." in pruned.output
    assert [d.name[11:] for d in session_dirs(root)] == ["finance_pea-first"]
    assert not old.exists()
    assert council(root, "sessions", "prune", "--older-than", "90d").output.startswith(
        "No sessions"
    )


def test_parse_age() -> None:
    assert cli.parse_age("90d") == timedelta(days=90)
    assert cli.parse_age(" 2w ") == timedelta(days=14)
    assert cli.parse_age("3m") is None


def test_new_refuses_no_research_on_a_council_that_requires_it(root: Path) -> None:
    refused = council(root, "new", "geopolitics", "Ceasefire by 2027?", "--no-tui", "--no-research")
    assert refused.exit_code == 1
    assert "council 'geopolitics' requires web research; drop --no-research" in refused.output
    assert not (root / "data" / "sessions").exists() or session_dirs(root) == []


def test_new_with_language(root: Path) -> None:
    bad = council(root, "new", "finance", "Rent or buy?", "--lang", "de", "--no-tui")
    assert bad.exit_code == 1
    assert "unknown language 'de': use en or fr" in bad.output

    detected = council(
        root, "new", "finance", "Dois-je louer ou acheter ?", "--no-tui", input="/quit\n"
    )
    assert detected.exit_code == 0, detected.output
    assert "Debate language: French. /lang en switches it before round 1" in detected.output
    shown = council(root, "sessions", "show", "louer")
    assert "language: French" in shown.output

    forced = council(
        root, "new", "finance", "Dois-je épargner ?", "--lang", "en", "--no-tui", input="/quit\n"
    )
    assert "Debate language: English. /lang fr switches it before round 1" in forced.output


def test_export_command_and_slash_export(root: Path) -> None:
    started = council(
        root, "new", "finance", "Rent or buy?", "--no-tui", input="/export html\n/quit\n"
    )
    assert started.exit_code == 0, started.output
    [directory] = session_dirs(root)
    exports = root / "data" / "exports"
    assert f"Exported: {exports / directory.name}.html" in started.output

    done = council(root, "sessions", "export", "rent-or-buy", "--format", "all")
    assert done.exit_code == 0, done.output
    assert f"wrote {exports / directory.name}.md" in done.output
    report = (exports / f"{directory.name}.md").read_text()
    assert report.startswith("# Rent or buy?") and "### Opening" in report

    usage = council(root, "new", "finance", "Rent?", "--no-tui", input="/export pdf\n/quit\n")
    assert "usage: /export [md|html|all]" in usage.output
    missing = council(root, "sessions", "export", "no-such-session")
    assert missing.exit_code == 1
