from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.config import ConfigError, Settings, deep_merge, load_settings
from tests.conftest import repo_settings


def test_defaults_without_files(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    assert settings.claude.default_model == "sonnet"
    assert settings.ollama.concurrency == 1
    assert settings.engine.max_output_tokens == 1500
    assert settings.personas_dir == tmp_path.resolve() / "personas"


def test_memory_scope_defaults_to_the_council() -> None:
    memory = Settings().memory
    assert memory.scope == "council" and memory.notes_council("finance") == "finance"
    memory = Settings.model_validate({"memory": {"scope": "global"}}).memory
    assert memory.notes_council("finance") is None


def test_repo_config_loads() -> None:
    settings = repo_settings()
    assert settings.ollama.models["qwen3:8b"].think is False
    assert settings.ollama.models["qwen3.6:latest"].think is False
    assert settings.ollama.models["gpt-oss:120b"].think == "low"


def test_local_override_deep_merges(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[claude]\ndefault_model = "sonnet"\ntimeout_s = 99\n', encoding="utf-8"
    )
    (tmp_path / "config.local.toml").write_text(
        '[claude]\ndefault_model = "haiku"\n[paths]\ndata = "/abs/data"\n', encoding="utf-8"
    )
    settings = load_settings(tmp_path)
    assert settings.claude.default_model == "haiku"
    assert settings.claude.timeout_s == 99
    assert settings.data_dir == Path("/abs/data")


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[claude]\ntimeout = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"claude\.timeout"):
        load_settings(tmp_path)


def test_invalid_toml_names_file(tmp_path: Path) -> None:
    (tmp_path / "config.local.toml").write_text("[claude\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"config\.local\.toml: invalid TOML"):
        load_settings(tmp_path)


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_user_name_from_config_or_login(monkeypatch: pytest.MonkeyPatch) -> None:
    import getpass

    assert Settings.model_validate({"user": {"name": "Ada"}}).user_name == "Ada"
    monkeypatch.setattr(getpass, "getuser", lambda: "thomas")
    assert Settings().user_name == "Thomas"

    def no_login() -> str:
        """Behave like a missing login name."""
        raise OSError("no login")

    monkeypatch.setattr(getpass, "getuser", no_login)
    assert Settings().user_name == "User"
