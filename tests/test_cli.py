from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from privy_council.cli import app
from privy_council.memory.store import MemoryStore
from tests.conftest import REPO_ROOT, PersonaWriter

runner = CliRunner()


def test_personas_list_shows_badges() -> None:
    result = runner.invoke(app, ["--root", str(REPO_ROOT), "personas", "list"])
    assert result.exit_code == 0, result.output
    assert "@Chair" in result.output
    assert "+Compounder" in result.output
    assert "ollama:qwen3.6:27b" in result.output


def test_validate_repo_examples_ok() -> None:
    result = runner.invoke(app, ["--root", str(REPO_ROOT), "personas", "validate"])
    assert result.exit_code == 0, result.output
    assert "34 personas, 9 councils OK" in result.output


def test_validate_reports_file_and_field(tmp_path: Path, write_persona: PersonaWriter) -> None:
    path = write_persona("alpha", temperature=9)
    result = runner.invoke(app, ["--root", str(tmp_path), "personas", "validate"])
    assert result.exit_code == 1
    assert str(path) in result.output
    assert "temperature:" in result.output


def test_show_persona_and_unknown() -> None:
    ok = runner.invoke(app, ["--root", str(REPO_ROOT), "personas", "show", "rigor"])
    assert ok.exit_code == 0
    assert "role: risk_officer" in ok.output
    assert "You are Rigor" in ok.output
    missing = runner.invoke(app, ["--root", str(REPO_ROOT), "personas", "show", "nobody"])
    assert missing.exit_code == 1


def test_councils_list_and_show() -> None:
    listed = runner.invoke(app, ["--root", str(REPO_ROOT), "councils", "list"])
    assert listed.exit_code == 0
    assert "finance" in listed.output
    shown = runner.invoke(app, ["--root", str(REPO_ROOT), "councils", "show", "finance"])
    assert shown.exit_code == 0
    assert "protocol.rebuttal_style: targeted" in shown.output
    assert "memory: on" in shown.output


def test_bad_config_exits_2(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[nope]\n", encoding="utf-8")
    result = runner.invoke(app, ["--root", str(tmp_path), "personas", "list"])
    assert result.exit_code == 2
    assert "config error" in result.output


def _memory_root(tmp_path: Path) -> MemoryStore:
    """Write a config pointing data at tmp_path and seed some memory."""
    (tmp_path / "config.toml").write_text(f"[paths]\ndata = '{tmp_path / 'data'}'\n")
    store = MemoryStore(tmp_path / "data")
    store.append_private("rigor", ["- 2026-10-17 (finance): rigor note"])
    store.append_private("pragmatist", ["- pragmatist note"])
    store.append_shared(["- 2026-10-17 (finance): The user rents."])
    return store


def test_memory_show(tmp_path: Path) -> None:
    _memory_root(tmp_path)
    shown = runner.invoke(app, ["--root", str(tmp_path), "memory", "show"])
    assert shown.exit_code == 0
    assert "# shared" in shown.output and "The user rents." in shown.output
    assert "# rigor" in shown.output and "pragmatist note" in shown.output
    one = runner.invoke(app, ["--root", str(tmp_path), "memory", "show", "--persona", "rigor"])
    assert "rigor note" in one.output and "pragmatist" not in one.output
    bad = runner.invoke(app, ["--root", str(tmp_path), "memory", "show", "--persona", "../x"])
    assert bad.exit_code == 2


def test_memory_reset(tmp_path: Path) -> None:
    store = _memory_root(tmp_path)
    base = ["--root", str(tmp_path), "memory", "reset"]
    assert runner.invoke(app, base).exit_code == 2
    assert runner.invoke(app, [*base, "--shared", "--all"]).exit_code == 2
    assert runner.invoke(app, [*base, "--persona", "../.."]).exit_code == 2

    declined = runner.invoke(app, [*base, "--persona", "rigor"], input="n\n")
    assert declined.exit_code == 1 and store.read_private("rigor")
    confirmed = runner.invoke(app, [*base, "--persona", "rigor"], input="y\n")
    assert "Deleted rigor's private notes." in confirmed.output
    assert store.read_private("rigor") == ""

    shared = runner.invoke(app, [*base, "--shared", "--yes"])
    assert "Deleted the shared user facts." in shared.output and store.read_shared() == ""
    again = runner.invoke(app, [*base, "--shared", "-y"])
    assert "Nothing to delete" in again.output

    everything = runner.invoke(app, [*base, "--all", "--yes"])
    assert "Deleted all memory." in everything.output and not store.root.exists()
