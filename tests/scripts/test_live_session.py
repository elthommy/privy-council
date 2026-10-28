"""scripts/live_session.py: `wait` returns once every line sent got its prompt."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from tests.conftest import REPO_ROOT


def driver() -> Any:
    """Import the script (scripts/ is not a package); Any lets a test set its constants."""
    spec = importlib.util.spec_from_file_location(
        "live_session", REPO_ROOT / "scripts/live_session.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wait_needs_one_prompt_per_line_sent(tmp_path: Path) -> None:
    live = driver()
    live.IDLE_S, live.POLL_S = 0.0, 0.01
    log = tmp_path / "out.log"
    log.write_text("— ★ opening\n> ", encoding="utf-8")  # multi-byte before `after`
    after = log.stat().st_size
    log.write_text(log.read_text(encoding="utf-8") + "[19:00] <Thomas> answer\n> ", "utf-8")
    assert live.wait(tmp_path, after, timeout=0.1, lines=2) == 1  # /close still running
    assert live.wait(tmp_path, after, timeout=0.1, lines=1) == 0
    log.write_text(log.read_text(encoding="utf-8") + "-- Round 2 --\n★ done\n> ", "utf-8")
    assert live.wait(tmp_path, after, timeout=0.1, lines=2) == 0


def test_setup_copies_the_local_config_without_the_discord_token() -> None:
    live = driver()
    text = (
        "[claude]\nsession_soft_cap_calls = 50\n\n"
        '[chat.discord]\ntoken = "secret-token"\nallowed_users = [\n  1,\n  2,\n]\n\n'
        "[research.local]\nenabled = true\n"
    )
    out = live.without_chat(text)
    assert "secret-token" not in out and "allowed_users" not in out
    assert "[claude]" in out and "[research.local]\nenabled = true" in out
