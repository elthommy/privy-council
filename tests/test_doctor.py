from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock
from typer.testing import CliRunner

from privy_council import cli
from privy_council.backends.base import BackendHealth
from privy_council.config import Settings, load_settings
from privy_council.doctor import check_budget, check_config, check_search, render
from privy_council.engine.budget import Ledger, QuotaSnapshot, UsageEntry
from tests.conftest import REPO_ROOT, PersonaWriter, repo_settings

runner = CliRunner()


def test_check_config_lists_required_models() -> None:
    health, models = check_config(repo_settings())
    assert health.ok
    assert models == {
        "qwen3.6:27b",
        "gemma4:26b",
        "gpt-oss:120b",
        "nemotron-3.5-super:latest",
    }


def test_check_config_local_only_lists_the_seat_map_models() -> None:
    health, models = check_config(repo_settings(), local_only=True)
    assert health.ok
    # Every Claude seat maps onto models the council already uses: no new pull.
    assert models == {
        "qwen3.6:27b",
        "gemma4:26b",
        "gpt-oss:120b",
        "nemotron-3.5-super:latest",
    }
    assert "local-only seat map: 4 model(s)" in health.details


def test_check_config_local_only_reports_a_bad_seat(
    tmp_path: Path, write_persona: PersonaWriter
) -> None:
    write_persona("alpha")
    (tmp_path / "config.toml").write_text(
        '[local_only.seats]\nghost = "qwen3.6:27b"\n', encoding="utf-8"
    )
    health, models = check_config(load_settings(tmp_path), local_only=True)
    assert not health.ok
    assert health.problems == ["[local_only.seats] ghost: no such persona"]
    assert models == set()


def test_check_config_reports_broken_persona(tmp_path: Path, write_persona: PersonaWriter) -> None:
    write_persona("alpha", color="nope")
    health, _ = check_config(load_settings(tmp_path))
    assert not health.ok
    assert "color:" in health.problems[0]


def test_check_budget_warns_at_cap(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[claude]\ndaily_soft_cap_calls = 1\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    Ledger.in_data_dir(settings.data_dir).record(UsageEntry(model="m"))
    health = check_budget(settings)
    assert health.details == ["1/1 Claude calls today"]
    assert health.warnings == ["daily soft cap reached"]


def test_render_marks_failures() -> None:
    lines = render(
        [BackendHealth(backend="ollama", ok=False, problems=["down"], warnings=["slow"])]
    )
    assert lines == [("fail", "✗ ollama: "), ("warn", "    ! slow"), ("fail", "    ✗ down")]


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_doctor_command_exit_code(monkeypatch: pytest.MonkeyPatch, ok: bool, code: int) -> None:
    async def fake_run(settings: Settings, local_only: bool) -> list[BackendHealth]:
        """Return a canned result instead of probing real backends."""
        return [BackendHealth(backend="claude_cli", ok=ok, details=["logged in"])]

    monkeypatch.setattr(cli, "run_doctor", fake_run)
    result = runner.invoke(cli.app, ["--root", str(REPO_ROOT), "doctor"])
    assert result.exit_code == code
    assert "claude_cli: logged in" in result.output


def test_budget_command(tmp_path: Path) -> None:
    Ledger.in_data_dir(tmp_path / "data").record(UsageEntry(model="m", input_tokens=1500))
    result = runner.invoke(cli.app, ["--root", str(tmp_path), "budget"])
    assert result.exit_code == 0
    assert "1/150 calls" in result.output
    assert "1,500 in" in result.output


def test_budget_and_doctor_show_quota(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    quota = QuotaSnapshot(status="allowed", five_hour=0.9, seven_day=0.1)
    Ledger.in_data_dir(settings.data_dir).record(UsageEntry(model="m", quota=quota))
    health = check_budget(settings)
    assert any(d.startswith("Pro quota 5h 90% / 7d 10%") for d in health.details)
    assert health.warnings == ["Claude 5-hour quota at 90%"]
    result = runner.invoke(cli.app, ["--root", str(tmp_path), "budget"])
    assert "Pro quota 5-hour: 90%" in result.output


def test_budget_without_quota(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["--root", str(tmp_path), "budget"])
    assert "Pro quota: unknown" in result.output


async def test_check_search_off_is_not_a_problem(tmp_path: Path) -> None:
    health = await check_search(Settings.model_validate({"root": tmp_path}))
    assert health.ok and health.details == ["off"]


async def test_check_search_on_queries_searxng(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    settings = Settings.model_validate({"root": tmp_path, "research": {"local": {"enabled": True}}})
    health = await check_search(settings)
    assert not health.ok
    assert "scripts/searxng.sh start" in health.problems[0]


async def test_check_search_names_the_provider(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    research = {"provider": "auto", "local": {"enabled": True}}
    health = await check_search(Settings.model_validate({"root": tmp_path, "research": research}))
    assert "used by: fallback ([research] provider = 'auto')" in health.details
