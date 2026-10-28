"""The engine's deterministic checks on member turns (M11)."""

from __future__ import annotations

import tomllib
from typing import Any

import pytest

from privy_council.engine import checks, context, prompts
from privy_council.models.flaw import FlawKind
from tests.conftest import REPO_ROOT

KNOWN = ["Rigor", "rigor", "Value", "value", "Chair", "chair", "User", "user"]


def _fixture_turns() -> list[dict[str, Any]]:
    """Load the synthetic turns of tests/fixtures/flaw_turns.toml."""
    with (REPO_ROOT / "tests" / "fixtures" / "flaw_turns.toml").open("rb") as fh:
        turns: list[dict[str, Any]] = tomllib.load(fh)["turn"]
    return turns


@pytest.mark.parametrize("turn", _fixture_turns(), ids=lambda t: t["name"])
def test_fixture_turns(turn: dict[str, Any]) -> None:
    flaws = checks.check_turn(
        turn["text"],
        member=turn["nick"].lower(),
        nick=turn["nick"],
        sources=turn["sources"],
        given=turn.get("given", ""),
        known=KNOWN,
    )
    got = [f"{f.kind.value}:{f.note if f.by == 'engine' and f.note else f.excerpt}" for f in flaws]
    assert got == turn["flaws"]


@pytest.mark.parametrize(
    ("raw", "value", "precision"),
    [
        ("1500", 1500, 100),
        ("1,500", 1500, 100),
        ("1 500", 1500, 100),
        ("1\u202f500", 1500, 100),
        ("1.5", 1.5, 0.1),
        ("1,5", 1.5, 0.1),
        ("1,234.56", 1234.56, 0.01),
        ("1.234,5", 1234.5, 0.1),
        ("16", 16, 1),
    ],
)
def test_parse_number(raw: str, value: float, precision: float) -> None:
    assert checks.parse_number(raw) == pytest.approx((value, precision))


def test_figures_read_ranges_multipliers_and_units() -> None:
    figs = checks.figures("€130\u2013200, a €200k pension, 15-20%, 3x faster, 2 years, $1.48tn")
    assert [(f.text, f.values, f.family) for f in figs] == [
        ("€130\u2013200", (130.0, 200.0), "money"),
        ("€200k", (200_000.0,), "money"),
        ("15-20%", (15.0, 20.0), "%"),
        ("3x", (3.0,), "x"),
        ("2 years", (2.0,), "time"),
        ("$1.48tn", (1.48e12,), "money"),
    ]


def test_bare_numbers_are_not_figures() -> None:
    assert checks.figures("Round 2: my top 3 reasons, in 2026.") == []


def test_topic_units_are_the_words_after_the_facts_numbers() -> None:
    facts = "Runs to November 11, 2026 alongside 2 events: 160 pulls, or 25,600 Astrite, 5 years."
    assert checks.topic_units(facts) == {"pull", "astrite", "event"}


def test_estimate_label_belongs_to_the_nearest_figure() -> None:
    figs = checks.figures("Over 3 years that is €60 a year (estimate, from memory).")
    assert [(f.text, f.labelled) for f in figs] == [("3 years", False), ("€60", True)]


def test_a_rounded_figure_matches_its_source() -> None:
    flaws = checks.figure_flaws("about €1,200 and 15%", "costs 1,234 € at 15.3 %", "", "rigor")
    assert flaws == []


def test_rounding_is_not_a_licence_to_drift() -> None:
    flaws = checks.figure_flaws("about €200", "it costs €170", "", "value")
    assert [f.note for f in flaws] == ["€200"]


def test_unsourced_flaw_carries_round_figure_and_sentence() -> None:
    text = "Fine so far. The T490 maxes at 32 GB. Done."
    (flaw,) = checks.figure_flaws(text, "", "", "value", round_=3)
    assert (flaw.kind, flaw.note, flaw.excerpt, flaw.round) == (
        FlawKind.UNSOURCED,
        "32 GB",
        "The T490 maxes at 32 GB.",
        3,
    )


def test_estimate_label_on_a_number_found_only_outside_the_facts() -> None:
    # 233 is in the debate (a member said it), not in the facts: an honest label.
    flaws = checks.figure_flaws("$233 (estimate, from memory)", "<Fixer> $233", "", "rigor")
    assert flaws == []


def test_devil_line_checked_in_either_language() -> None:
    ok = "**Position:** Playing devil's advocate: replace it."
    assert checks.devil_line_flaw(ok, "contrarian", "Playing devil's advocate:") == []
    fr = "**Position :** Je joue l\u2019avocat du diable : non."
    assert checks.devil_line_flaw(fr, "contrarian", "Je joue l'avocat du diable :") == []
    (flaw,) = checks.devil_line_flaw("**Position:** No.", "contrarian", "Playing devil's advocate:")
    assert flaw.kind is FlawKind.MISSING_DEVIL_LINE


def test_group_mentions_and_possessives_are_fine() -> None:
    text = "@all: as @Rigor's numbers show, @User should wait."
    assert checks.format_flaws(text, "value", "Value", KNOWN) == []


def test_each_unknown_mention_is_flagged_once() -> None:
    flaws = checks.format_flaws("@Bob and @bob again", "value", "Value", KNOWN)
    assert [f.excerpt for f in flaws] == ["@Bob"]


def test_sources_drop_the_task_and_given_keeps_facts_and_dossier() -> None:
    user = "\n\n".join(
        [
            prompts.section(prompts.TITLE_QUESTION, "Upgrade for 600 EUR?"),
            prompts.section(prompts.TITLE_DOSSIER, "File a.txt: 16 GB"),
            prompts.section(prompts.TITLE_FACTS, "- $233 a stick"),
            prompts.section(prompts.TITLE_HISTORY, "<Rigor> 40 GB max"),
            prompts.section(prompts.TITLE_TASK, "your top 3, 0-100"),
        ]
    )
    sources = context.sources_text("system", user)
    assert "600 EUR" in sources and "40 GB" in sources and "top 3" not in sources
    given = context.given_text(user)
    assert "16 GB" in given and "$233" in given and "40 GB" not in given


SHARES = "<Thomas> I can put in 25,000 EUR\n<Ana> I can put 40,000 EUR in\n<Ben> only 15,000 EUR"


@pytest.mark.parametrize(
    "text",
    [
        "Thomas (25 k), Ana (40 k) and Ben (15 k) together only cover €80 k.",
        "Ana ≈ 40 k, Thomas ≈ 25 k, Ben ≈ 15 k. The price exceeds the combined cash (€80 k).",
        "Ana's €40,000 and Ben's €15,000 make €55,000.",
    ],
)
def test_totals_of_sourced_figures_are_the_members_math(text: str) -> None:
    assert checks.figure_flaws(text, SHARES, "", "pragmatist") == []


@pytest.mark.parametrize(
    ("text", "note"),
    [
        ("Thomas (25 k), Ana (40 k) and Ben (15 k) cover €95 k.", "€95 k"),  # a wrong total
        ("Costs are €300, €450 and €250, so €1,000 a year.", "€1,000"),  # parts unsourced
        ("Ana and Ben bring €55,000 between them.", "€55,000"),  # parts not in the turn
    ],
)
def test_other_totals_stay_unsourced(text: str, note: str) -> None:
    flaws = checks.figure_flaws(text, SHARES, "", "pragmatist")
    assert note in [f.note for f in flaws]


KILOMETRES = "<Thomas> I drive about 6,000 km a year\n<Ana> about 12,000 km\n<Ben> about 3,000 km"


def test_shares_of_the_turns_distances_are_the_members_math() -> None:
    text = (
        "Thomas drives 6 000 km (≈29 % of the 21 000 km in total), Ana 12 000 km (≈57 %) "
        "and Ben 3 000 km (≈14 %)."
    )
    assert checks.figure_flaws(text, KILOMETRES, "", "pragmatist") == []


def test_an_invented_distance_is_unsourced() -> None:
    flaws = checks.figure_flaws("A typical car does 15,000 km a year.", KILOMETRES, "", "rigor")
    assert [f.note for f in flaws] == ["15,000 km"]


def test_figures_found_needs_every_figure_with_its_unit() -> None:
    note = "- One singular pull costs 160 Astrites (https://x)\n- Hard pity at 80 pulls"
    assert checks.figures_found("A single pull costs 160 Astrite", note)
    assert checks.figures_found("80 pulls to hard pity, 160 Astrite each", note)
    assert not checks.figures_found("a pull costs 180 Astrite", note)
    assert not checks.figures_found("160 Astrite and 25,600 Astrite in all", note)
    assert not checks.figures_found("no figure at all", note)


PULLS = (
    "# Fresh facts\n- A singular pull costs 160 astrites (https://x)\n- Hard pity at 80 pulls\n"
    "<Mika> I'm at 15 pity with 30 pulls saved\n<Thomas> I'm at 40 pity with 60 pulls saved"
)


@pytest.mark.parametrize(
    "text",
    [
        "The worst case is 65 pulls to hard pity (80 \u2212 15).",  # written out, other units
        "65 pulls to hard pity (80 \u2212 15), 30 saved, so 35 more (65 \u2212 30).",  # chained
        "You're at 40 pity with 60 saved pulls (≈9 600 Astrite).",  # count x sourced price
        "He needs 35 more pulls (5,600 Astrite at 160 each).",
    ],
)
def test_written_operations_and_prices_are_the_members_math(text: str) -> None:
    assert checks.figure_flaws(text, PULLS, PULLS, "saver") == []


@pytest.mark.parametrize(
    ("text", "note"),
    [
        ("The worst case is 120 pulls \u00d7 200 = 24,000 Astrite.", "24,000 Astrite"),  # 200
        ("Your 60 saved pulls are about 10,000 Astrite.", "10,000 Astrite"),  # not exact
        ("Expect 7 000 Astrite of income.", "7 000 Astrite"),
    ],
)
def test_wrong_operands_and_rounded_prices_stay_unsourced(text: str, note: str) -> None:
    assert note in [f.note for f in checks.figure_flaws(text, PULLS, PULLS, "saver")]


def test_prices_in_euros_excuse_nothing() -> None:
    sources = "<Thomas> a blade costs €10; I shave 3 times a week"
    flaws = checks.figure_flaws("A starter razor is about €30.", sources, sources, "fixer")
    assert [f.note for f in flaws] == ["€30"]


def test_figures_cleared_for_a_given_figure_or_own_arithmetic() -> None:
    text = "You're at 40 pity with 60 saved pulls (≈9 600 Astrite), and a 4% rate."
    assert checks.figures_cleared("60 saved pulls (≈9 600 Astrite)", text, PULLS, PULLS)
    assert not checks.figures_cleared("a 4% rate", text, PULLS, PULLS)
    assert not checks.figures_cleared("no figure", text, PULLS, PULLS)


@pytest.mark.parametrize(
    ("note", "confirms", "disputes"),
    [
        ("160 pulls x 200 Astrite = 32,000 Astrite, which is correct; but ...", True, False),
        ("This matches Saver's figure, so it is not actually an arithmetic error.", True, False),
        ("5% of $725-730B is about $36B, which is fine, but it assumes a cut.", True, False),
        ("The 65 is wrong: ... but 35 x 160 = 5,600 is correct.", True, True),
        ("At 40 pity Thomas needs 40 pulls to reach hard pity, not 20.", False, True),
        ("Two 8 TB drives would cost about $660, well over 400 EUR.", False, False),
        ("Le total de 2 320 € est correct, mais la marge est faible.", True, False),
    ],
)
def test_moderator_notes_that_confirm_or_dispute(note: str, confirms: bool, disputes: bool) -> None:
    assert (checks.note_confirms(note), checks.note_disputes(note)) == (confirms, disputes)
