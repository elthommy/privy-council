"""Variant parsing and application (M14)."""

from __future__ import annotations

import pytest

from privy_council.compare.spec import (
    SeatChange,
    Variant,
    VariantError,
    apply_variant,
    check_variants,
    parse_variant,
)
from privy_council.models.council import CouncilPreset, MemoryMode, ProtocolSettings
from privy_council.models.persona import BackendKind, Persona
from tests.engine.test_protocol import MEMBERS
from tests.engine.test_protocol import personas as personas


@pytest.fixture
def preset() -> CouncilPreset:
    """The finance-like council the engine tests use."""
    return CouncilPreset(
        name="finance",
        moderator="chair",
        members=MEMBERS,
        protocol=ProtocolSettings(max_rounds=6, auto_stop_on_consensus=True),
    )


def test_bare_name_is_the_council_as_is() -> None:
    variant = parse_variant("base")
    assert variant == Variant(name="base")
    assert variant.label() == "as is"


def test_ollama_model_names_keep_their_colon() -> None:
    variant = parse_variant("qwen:rigor=qwen3.6:27b")
    assert variant.seats == [SeatChange(slug="rigor", model="qwen3.6:27b")]


def test_backend_prefix_switches_backend() -> None:
    variant = parse_variant("mix: rigor=claude_cli:sonnet, compounder=ollama:gemma4:26b")
    assert variant.seats == [
        SeatChange(slug="rigor", model="sonnet", backend=BackendKind.CLAUDE_CLI),
        SeatChange(slug="compounder", model="gemma4:26b", backend=BackendKind.OLLAMA),
    ]
    assert variant.label() == "rigor=claude_cli:sonnet, compounder=ollama:gemma4:26b"


def test_add_and_drop_members() -> None:
    variant = parse_variant("roster:+steelman,-contrarian")
    assert variant.add == ["steelman"] and variant.drop == ["contrarian"]
    assert variant.label() == "+steelman, -contrarian"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("Base", "must be lowercase"),
        (":rigor=x", "must be lowercase"),
        ("v:rigor", "can't read 'rigor'"),
        ("v:rigor=", "can't read"),
        ("v:=qwen", "can't read"),
    ],
)
def test_bad_variants_are_refused(text: str, match: str) -> None:
    with pytest.raises(VariantError, match=match):
        parse_variant(text)


def test_a_comparison_needs_two_distinct_variants() -> None:
    with pytest.raises(VariantError, match="at least two"):
        check_variants([Variant(name="a")])
    with pytest.raises(VariantError, match="distinct"):
        check_variants([Variant(name="a"), Variant(name="a")])
    check_variants([Variant(name="a"), Variant(name="b")])


def test_apply_fixes_rounds_and_turns_memory_off(
    preset: CouncilPreset, personas: dict[str, Persona]
) -> None:
    council, seated = apply_variant(preset, personas, Variant(name="base"), 3)
    assert council.members == MEMBERS
    assert council.protocol.max_rounds == 3
    assert council.protocol.auto_stop_on_consensus is False
    assert council.memory is MemoryMode.OFF
    assert seated == personas
    assert preset.protocol.max_rounds == 6  # the preset itself is untouched


def test_apply_reseats_and_changes_roster(
    preset: CouncilPreset, personas: dict[str, Persona]
) -> None:
    variant = parse_variant("v:rigor=qwen3.6:27b,+steelman,-contrarian")
    council, seated = apply_variant(preset, personas, variant, 2)
    assert council.members == ["rigor", "compounder", "pragmatist", "steelman"]
    assert seated["rigor"].meta.model == "qwen3.6:27b"
    assert seated["rigor"].meta.backend is personas["rigor"].meta.backend
    assert personas["rigor"].meta.model != "qwen3.6:27b"  # the loaded persona is untouched


def test_moderator_backend_switch_drops_its_synthesis_model(
    preset: CouncilPreset, personas: dict[str, Persona]
) -> None:
    assert personas["chair"].meta.synthesis_model is not None
    same = parse_variant("v:chair=haiku")
    _, seated = apply_variant(preset, personas, same, 2)
    assert seated["chair"].meta.synthesis_model == personas["chair"].meta.synthesis_model
    local = parse_variant("v:chair=ollama:gpt-oss:120b")
    _, seated = apply_variant(preset, personas, local, 2)
    assert seated["chair"].meta.backend is BackendKind.OLLAMA
    assert seated["chair"].meta.synthesis_model is None


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("v:realist=qwen", "'realist' has no seat"),
        ("v:contrarian=qwen,-contrarian", "'contrarian' has no seat"),
        ("v:-realist", "-realist is not a member"),
        ("v:+nobody", r"\+nobody is not a persona"),
        ("v:+rigor", r"\+rigor already sits"),
        ("v:+chair", r"\+chair already sits"),
        ("v:-rigor,-compounder,-pragmatist,-contrarian", "leaves no member"),
    ],
)
def test_variants_that_dont_fit_the_council(
    preset: CouncilPreset, personas: dict[str, Persona], text: str, match: str
) -> None:
    with pytest.raises(VariantError, match=match):
        apply_variant(preset, personas, parse_variant(text), 2)
