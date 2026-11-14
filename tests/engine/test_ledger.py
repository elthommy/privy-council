"""M17 estimate lines: parsing, coherence checks, rendering (made-up numbers)."""

from __future__ import annotations

import pytest

from privy_council.engine import ledger, prompts
from privy_council.models.estimate import Estimate, EstimateRecord, Split
from privy_council.models.flaw import FlawKind


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("**Estimate:** 12% (7-20%)", (12, 7, 20)),
        ("**Estimate**: about 12 % (7 to 20 %)", (12, 7, 20)),
        ("Estimate: 12%", (12, None, None)),
        ("**Estimation :** 12,5 % (7 à 20 %)", (12.5, 7, 20)),
        ("**Estimate:** 12% (7–20%)", (12, 7, 20)),  # noqa: RUF001
        ("**Estimate:** 7-20%", (13.5, 7, 20)),
        ("Position: low.\nmy estimate: 30%\n\n**Estimate:** 9% (5-14%)", (9, 5, 14)),
    ],
)
def test_parse_estimate(text: str, want: tuple[float, float | None, float | None]) -> None:
    est = ledger.parse_estimate(text)
    assert est is not None
    assert (est.value, est.low, est.high) == want


@pytest.mark.parametrize(
    "text", ["Position: unlikely, about 12%.", "**Estimate:** low", "**Estimate:** 150%"]
)
def test_unreadable_estimates_are_none(text: str) -> None:
    assert ledger.parse_estimate(text) is None


def test_parse_split_in_both_languages() -> None:
    en = ledger.parse_split("**Split:** P(event) 40%; if it happens 20%; if it doesn't 5%")
    fr = ledger.parse_split("**Décomposition :** P(événement) 40 % ; s'il arrive 20 % ; sinon 5 %")
    assert en == fr == Split(event=40, if_yes=20, if_no=5)
    assert en is not None and en.implied == pytest.approx(11)
    assert ledger.parse_split("**Split:** P(event) 40%; if it happens 20%") is None


def read(text: str, previous: EstimateRecord | None = None, split_set: bool = False) -> list[str]:
    """Return the notes of the flaws a turn gets, kind first."""
    _, flaws = ledger.read_turn(text, member="m", round_=2, previous=previous, split_set=split_set)
    return [f"{f.kind.value}: {f.note}" for f in flaws]


def before(value: float) -> EstimateRecord:
    """A previous record at `value`%."""
    return EstimateRecord(round=1, estimate=Estimate(value=value, low=1, high=50))


def test_a_clean_turn_is_recorded_without_flaws() -> None:
    record, flaws = ledger.read_turn(
        "**Estimate:** 12% (7-20%)", member="m", round_=2, previous=before(13), split_set=False
    )
    assert flaws == []
    assert record == EstimateRecord(round=2, estimate=Estimate(value=12, low=7, high=20))


def test_missing_and_incoherent_lines() -> None:
    assert read("No number today.") == ["missing_estimate: No estimate line."]
    assert read("**Estimate:** 25% (7-20%)") == [
        "estimate_incoherent: 25% lies outside its own range 7-20%."
    ]
    assert read("**Estimate:** 12% (20-7%)") == [
        "estimate_incoherent: The range 20-7% is reversed."
    ]


def test_a_split_that_disagrees_with_the_estimate() -> None:
    text = (
        "**Split:** P(event) 40%; if it happens 20%; if it doesn't 5%\n**Estimate:** 30% (25-35%)"
    )
    assert read(text, split_set=True) == [
        "estimate_incoherent: 40% x 20% + 60% x 5% = 11%, but the estimate says 30%."
    ]
    fitting = text.replace("30% (25-35%)", "11% (8-15%)")
    assert read(fitting, split_set=True) == []
    # Without a range, the implied value must sit within SPLIT_TOLERANCE points.
    assert read(fitting.replace("11% (8-15%)", "13%"), split_set=True) == []
    assert read("**Estimate:** 11% (8-15%)", split_set=True) == [
        "missing_estimate: No split line, though the moderator split the question."
    ]


def test_a_split_line_is_ignored_before_the_moderator_sets_one() -> None:
    text = "**Split:** P(event) 40%; if it happens 20%; if it doesn't 5%\n**Estimate:** 30%"
    record, flaws = ledger.read_turn(text, member="m", round_=1, previous=None, split_set=False)
    assert flaws == [] and record is not None and record.split is None


def test_a_moved_estimate_needs_the_update_marker() -> None:
    assert read("**Estimate:** 7% (3-12%)", previous=before(12)) == [
        'unmarked_change: The estimate moved from 12% to 7% without "I update" or a reason '
        "for the change."
    ]
    assert read("I update: lower.\n**Estimate:** 7% (3-12%)", previous=before(12)) == []
    assert read("**Estimate:** 10% (3-12%)", previous=before(12)) == []  # under 3 points


@pytest.mark.parametrize(
    "prose",
    [
        "I correct myself: that figure was the total, not the share.",
        "My 30% was always the odds of the rule, not of a default.",
        "The longer window pushes my number from 9% to about 12%.",
        "The joint event is not the earlier 30%: 0.4 x 0.5 = 20%, and less with timing.",
        "Je me corrige : ce chiffre était le total.",
        "Je passe de 30 % à 12 % : la règle est moins probable.",
    ],
)
def test_a_change_explained_in_other_words_is_not_flagged(prose: str) -> None:
    assert read(f"{prose}\n\n**Estimate:** 12% (6-20%)", previous=before(30)) == []


def test_a_bare_move_is_still_flagged() -> None:
    # The old number only in the new range doesn't count as naming it.
    text = "The covenant test is the signal to watch.\n\n**Estimate:** 12% (6-30%)"
    assert [n.split(":")[0] for n in read(text, previous=before(30))] == ["unmarked_change"]


def test_no_change_check_after_the_user_spoke() -> None:
    _, flaws = ledger.read_turn(
        "**Estimate:** 7% (3-12%)",
        member="m",
        round_=2,
        previous=before(30),
        split_set=False,
        reframed=True,
    )
    assert flaws == []


def test_blank_lines_hides_only_the_estimate_and_split_numbers() -> None:
    text = "Debt 124% of GDP.\n**Split:** P(event) 40%; if it happens 20%; if it doesn't 5%\n"
    text += "**Estimate:** 22% (12-32%)"
    blanked = ledger.blank_lines(text)
    assert len(blanked) == len(text)
    assert "124%" in blanked
    assert not any(n in blanked for n in ("40%", "22%", "12-32%"))


def test_aggregate() -> None:
    assert ledger.aggregate([]) is None
    assert ledger.aggregate([12, 8, 30]) == (12, 8, 30)


def test_estimate_rules_per_seat_and_split() -> None:
    plain = prompts.estimate_rule("en", devil=False, split=None)
    assert "**Estimate:** N% (low-high%)" in plain
    split = prompts.estimate_rule("fr", devil=False, split="lenders refuse to extend")
    assert "lenders refuse to extend" in split
    assert "**Décomposition :** P(événement) A%; s'il arrive B%; sinon C%" in split
    assert "**Estimation :** N% (low-high%)" in split
    devil = prompts.estimate_rule("en", devil=True, split="x")
    assert "Don't give your own probability" in devil and "Estimate:" not in devil


def test_split_rule() -> None:
    assert prompts.split_rule(False, None) == prompts.SPLIT_RULE_OFF
    assert prompts.split_rule(True, None) == prompts.SPLIT_RULE_ON
    assert "already set (an event)" in prompts.split_rule(True, "an event")


def test_render_estimates_and_synthesis() -> None:
    records = {
        "a": [
            EstimateRecord(round=1, estimate=Estimate(value=12, low=7, high=20)),
            EstimateRecord(
                round=2,
                estimate=Estimate(value=8),
                split=Split(event=40, if_yes=15, if_no=3),
            ),
        ]
    }
    table = prompts.render_estimates(records, ["a", "b", "d"], ["d"], str.upper, "an event")
    assert "Split event: an event" in table
    assert "- @A: 12% (7-20%) -> 8%; split 40% / 15% / 3%, implies 7.8%" in table.replace("→", "->")
    assert "- @B: no estimate line yet" in table
    assert "- @D: devil's advocate: gives no estimate" in table
    synthesis = prompts.synthesis_estimates(records, ["a", "b"], str.upper, (8, 8, 8))
    assert synthesis is not None
    assert "- @A (round 2): 8%; split" in synthesis
    assert "median 8%, members' estimates from 8% to 8%" in synthesis
    assert prompts.synthesis_estimates({}, ["a"], str.upper, None) is None


def test_flaw_kinds_for_estimates_exist() -> None:
    assert {k.value for k in FlawKind} >= {
        "missing_estimate",
        "estimate_incoherent",
        "unmarked_change",
    }
