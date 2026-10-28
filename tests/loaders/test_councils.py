from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.loaders.councils import load_council, load_councils
from privy_council.loaders.errors import LoadError
from privy_council.loaders.personas import load_personas
from privy_council.models.council import MemoryMode, RebuttalStyle
from privy_council.models.persona import Persona, Role
from tests.conftest import REPO_ROOT, PersonaWriter


def _write_council(tmp_path: Path, name: str, text: str) -> Path:
    """Write councils/<name>.yaml under tmp_path."""
    directory = tmp_path / "councils"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def roster(write_persona: PersonaWriter) -> dict[str, Persona]:
    """Load a moderator plus two advisors from tmp_path/personas."""
    write_persona("mod", role="moderator")
    write_persona("alpha")
    path = write_persona("beta")
    personas, errors = load_personas(path.parent)
    assert errors == []
    return personas


def test_example_council_loads() -> None:
    personas, _ = load_personas(REPO_ROOT / "personas")
    councils, errors = load_councils(REPO_ROOT / "councils", personas)
    assert errors == []
    finance = councils["finance"]
    assert finance.moderator == "chair"
    assert finance.protocol.rebuttal_style is RebuttalStyle.TARGETED
    assert finance.memory is MemoryMode.ON
    wellbeing = councils["wellbeing"]
    assert wellbeing.memory is MemoryMode.READ_ONLY
    assert wellbeing.members == ["stoic", "coach", "kindness", "vitality", "contrarian"]
    # Memory-free personas, so a later switch to `on` still keeps their notes off disk.
    assert not any(personas[slug].meta.memory for slug in wellbeing.members[:4])
    salon = councils["salon"]
    assert salon.memory is MemoryMode.OFF
    assert salon.members == ["socrates", "plato", "nietzsche", "epicurus"]
    assert salon.protocol.max_rounds == 5
    assert salon.protocol.stall_rounds == 0 and not salon.protocol.auto_stop_on_consensus
    assert all(personas[slug].meta.role is Role.ADVISOR for slug in salon.members)
    purchases = councils["purchases"]
    assert purchases.memory is MemoryMode.ON and purchases.protocol.require_research
    assert purchases.members == [
        "specialist",
        "fixer",
        "value",
        "pragmatist",
        "rigor",
        "contrarian",
    ]


def test_protocol_defaults(tmp_path: Path) -> None:
    council = load_council(_write_council(tmp_path, "c", "name: c\nmoderator: m\nmembers: [a]\n"))
    assert council.protocol.blind_first_round is True
    assert council.protocol.max_rounds == 6


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("name: c\nmoderator: m\nmembers: [a, a]\n", "duplicates"),
        ("name: c\nmoderator: a\nmembers: [a]\n", "must not also be a member"),
        ("name: c\nmoderator: m\nmembers: []\n", "members:"),
        (
            "name: c\nmoderator: m\nmembers: [a]\nprotocol: {max_rounds: 0}\n",
            "protocol.max_rounds:",
        ),
        ("name: c\nmoderator: m\nmembers: [a]\nprotocol: {style: x}\n", "protocol.style:"),
        ("name: c\nmoderator: m\nmembers: [a]\nmemory: maybe\n", "memory:"),
        ("name: other\nmoderator: m\nmembers: [a]\n", "must match the filename"),
        ("- just\n- a list\n", "expected a YAML mapping"),
        ("name: [oops\n", "invalid YAML"),
    ],
)
def test_invalid_council(tmp_path: Path, text: str, fragment: str) -> None:
    path = _write_council(tmp_path, "c", text)
    with pytest.raises(LoadError) as excinfo:
        load_council(path)
    assert str(path) in str(excinfo.value)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    ("moderator", "members", "fragment"),
    [
        ("ghost", "[alpha]", "moderator: unknown persona 'ghost'"),
        ("alpha", "[beta]", "has role 'advisor', expected 'moderator'"),
        ("mod", "[alpha, ghost]", "members.1: unknown persona 'ghost'"),
    ],
)
def test_roster_checked_against_personas(
    tmp_path: Path, roster: dict[str, Persona], moderator: str, members: str, fragment: str
) -> None:
    _write_council(tmp_path, "c", f"name: c\nmoderator: {moderator}\nmembers: {members}\n")
    councils, errors = load_councils(tmp_path / "councils", roster)
    assert councils == {}
    assert fragment in str(errors[0])


def test_moderator_cannot_be_member(
    tmp_path: Path, roster: dict[str, Persona], write_persona: PersonaWriter
) -> None:
    write_persona("mod2", role="moderator")
    personas, _ = load_personas(tmp_path / "personas")
    _write_council(tmp_path, "c", "name: c\nmoderator: mod\nmembers: [alpha, mod2]\n")
    _, errors = load_councils(tmp_path / "councils", personas)
    assert "members.1: 'mod2' is a moderator" in str(errors[0])


@pytest.mark.parametrize(
    ("value", "mode"),
    [("on", MemoryMode.ON), ("off", MemoryMode.OFF), ("read_only", MemoryMode.READ_ONLY)],
)
def test_memory_mode_from_yaml(tmp_path: Path, value: str, mode: MemoryMode) -> None:
    text = f"name: c\nmoderator: m\nmembers: [a]\nmemory: {value}\n"
    assert load_council(_write_council(tmp_path, "c", text)).memory is mode
