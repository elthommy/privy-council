from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.loaders.errors import LoadError
from privy_council.loaders.personas import load_persona, load_personas
from privy_council.models.persona import BackendKind, Role
from tests.conftest import REPO_ROOT, PersonaWriter


def test_example_personas_all_load() -> None:
    personas, errors = load_personas(REPO_ROOT / "personas")
    assert errors == []
    assert {"chair", "rigor", "compounder", "pragmatist", "contrarian"} <= set(personas)
    assert personas["chair"].meta.role is Role.MODERATOR


def test_valid_persona_defaults(write_persona: PersonaWriter) -> None:
    persona = load_persona(write_persona("alpha", body="  # Who\nYou are Alpha.  \n"))
    assert persona.slug == "alpha"
    assert persona.name == "Alpha"
    assert persona.meta.role is Role.ADVISOR
    assert persona.meta.backend is BackendKind.OLLAMA
    assert persona.meta.memory is True
    assert persona.meta.max_output_tokens is None  # falls back to [engine] default
    assert persona.body == "# Who\nYou are Alpha."


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"color": "red"}, "color"),
        ({"role": "king"}, "role"),
        ({"backend": "openai"}, "backend"),
        ({"temperature": 3.5}, "temperature"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"model": None}, "model"),
        ({"name": "has space"}, "name"),
        ({"colour": "#ffffff"}, "colour"),
    ],
)
def test_invalid_field_is_named(
    write_persona: PersonaWriter, overrides: dict[str, object], field: str
) -> None:
    path = write_persona("alpha", **overrides)
    with pytest.raises(LoadError) as excinfo:
        load_persona(path)
    assert str(path) in str(excinfo.value)
    assert any(p.startswith(f"{field}:") for p in excinfo.value.problems)


def test_slug_must_match_filename(write_persona: PersonaWriter) -> None:
    path = write_persona("alpha", slug="beta")
    with pytest.raises(LoadError, match=r"slug: 'beta' must match the filename 'alpha'"):
        load_persona(path)


def test_empty_body_rejected(write_persona: PersonaWriter) -> None:
    with pytest.raises(LoadError, match="body:"):
        load_persona(write_persona("alpha", body="   "))


def test_missing_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "alpha.md"
    path.write_text("# Just a body\n", encoding="utf-8")
    with pytest.raises(LoadError, match="missing YAML frontmatter"):
        load_persona(path)


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "alpha.md"
    path.write_text("---\nname: [unclosed\n---\nbody\n", encoding="utf-8")
    with pytest.raises(LoadError, match="invalid YAML frontmatter"):
        load_persona(path)


def test_load_personas_collects_all_errors(write_persona: PersonaWriter) -> None:
    good = write_persona("alpha")
    write_persona("beta", color="nope")
    write_persona("gamma", role="nope")
    personas, errors = load_personas(good.parent)
    assert list(personas) == ["alpha"]
    assert sorted(e.path.stem for e in errors) == ["beta", "gamma"]


def test_duplicate_nick_rejected_case_insensitively(write_persona: PersonaWriter) -> None:
    write_persona("alpha", name="Echo")
    path = write_persona("beta", name="echo")
    personas, errors = load_personas(path.parent)
    assert list(personas) == ["alpha"]
    assert len(errors) == 1
    assert errors[0].path == path
    assert "already used" in errors[0].problems[0]


def test_synthesis_model_only_for_moderator(write_persona: PersonaWriter) -> None:
    path = write_persona("advisor", synthesis_model="opus")
    with pytest.raises(LoadError, match="synthesis_model is only used by a moderator"):
        load_persona(path)
    ok = write_persona(
        "mod", role="moderator", backend="claude_cli", model="sonnet", synthesis_model="opus"
    )
    assert load_persona(ok).meta.synthesis_model == "opus"
