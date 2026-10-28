"""Shared test fixtures."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from privy_council.config import CONFIG_FILE, Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_settings() -> Settings:
    """The repo's committed config.toml, never the user's config.local.toml.

    `load_settings(REPO_ROOT)` would merge the user's personal overrides, so a
    local setting (e.g. `[research.local] enabled`) could change test results.
    """
    data = tomllib.loads((REPO_ROOT / CONFIG_FILE).read_text(encoding="utf-8"))
    return Settings.model_validate({**data, "root": REPO_ROOT})


PersonaWriter = Callable[..., Path]


def persona_text(body: str = "You are a test persona.", **meta: object) -> str:
    """Render a persona file from frontmatter fields and a body."""
    return f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n\n{body}\n"


@pytest.fixture
def write_persona(tmp_path: Path) -> PersonaWriter:
    """Return a factory writing a valid persona (overridable) into tmp_path/personas."""
    directory = tmp_path / "personas"
    directory.mkdir(exist_ok=True)

    def _write(stem: str, body: str = "You are a test persona.", **overrides: object) -> Path:
        """Write personas/<stem>.md (slug defaults to stem) and return its path."""
        meta: dict[str, object] = {
            "name": stem.capitalize(),
            "slug": stem,
            "backend": "ollama",
            "model": "qwen3:8b",
        }
        meta.update(overrides)
        meta = {k: v for k, v in meta.items() if v is not None}
        path = directory / f"{stem}.md"
        path.write_text(persona_text(body, **meta), encoding="utf-8")
        return path

    return _write
