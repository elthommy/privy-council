"""M18 fact tags: numbering, rendering, citations (made-up notes and turns)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from privy_council.engine import checks, context, facts, prompts
from privy_council.models.session import ResearchNote

FIRST = (
    "- **Debt:** Acme owes $9.2bn, $5.5bn of it due in 2027. [Wire](https://example.com/a)\n"
    "  - Its main loan carries a 1.35x coverage test.\n"
    "- Missing: the lenders' names"
)
SECOND = "Rates rose to 4.5% in March.\n- Acme cut 1,200 jobs. [Paper](https://example.com/b)"


def notes(*texts: str) -> list[ResearchNote]:
    """Research notes with the given texts, oldest first."""
    ts = datetime(2026, 1, 5, tzinfo=UTC)
    return [ResearchNote(focus=f"focus {i}", text=t, ts=ts) for i, t in enumerate(texts)]


def test_bullets_are_numbered_oldest_first_and_keep_their_tags() -> None:
    one = context.research_facts(notes(FIRST))
    both = context.research_facts(notes(FIRST, SECOND))
    assert [f.id for f in one] == ["F1", "F2"]  # the "Missing:" line is no fact
    assert both[:2] == one  # a new note doesn't renumber the old ones
    assert [(f.id, f.note_index) for f in both[2:]] == [("F3", 1), ("F4", 1)]
    assert both[2].text == "Rates rose to 4.5% in March."


def test_prompts_show_each_bullet_with_its_tag() -> None:
    text = context.facts_text(notes(FIRST, SECOND))
    assert text is not None
    assert "- [F1] **Debt:** Acme owes $9.2bn" in text
    assert "  - [F2] Its main loan" in text
    assert "[F3] Rates rose" in text and "- [F4] Acme cut" in text
    assert text.index("[F3]") < text.index("[F1]")  # newest note first, as before
    shown = facts.in_prompt(prompts.section(prompts.TITLE_FACTS, text))
    assert set(shown) == {"F1", "F2", "F3", "F4"}
    assert shown["F2"] == "Its main loan carries a 1.35x coverage test."


@pytest.mark.parametrize(
    ("sentence", "ids"),
    [
        ("Debt is $9.2bn [F1].", ["F1"]),
        ("Both hold [F1, F3] and [F3][F4].", ["F1", "F3", "F4"]),
        ("A tier comes later [F7, primary].", ["F7"]),
        ("No tag, just [brackets] and F1.", []),
    ],
)
def test_tags(sentence: str, ids: list[str]) -> None:
    assert facts.tags(sentence) == ids


def test_a_tag_after_the_full_stop_stays_with_its_sentence() -> None:
    text = "Acme owes $9.2bn. [F1] Rates are 4.5% [F3].\n- A list item [F4]\nNo tag here."
    spans = facts.tagged_sentences(text)
    assert [(text[s:e].strip(), ids) for s, e, ids in spans] == [
        ("Acme owes $9.2bn. [F1]", ["F1"]),
        ("Rates are 4.5% [F3].", ["F3"]),
        ("- A list item [F4]", ["F4"]),
    ]


def prompt_parts(*texts: str, user: str = "") -> tuple[str, str]:
    """Return (sources, given) as a member's prompt would give them."""
    facts_part = prompts.section(prompts.TITLE_FACTS, context.facts_text(notes(*texts)) or "")
    given = facts_part
    sources = f"{facts_part}\n\n# Recent messages\n{user}"
    return sources, given


def cite(turn: str, user: str = "") -> list[str]:
    """Return the flaws a turn gets, as "kind: note"."""
    sources, given = prompt_parts(FIRST, SECOND, user=user)
    flaws = checks.check_turn(turn, member="m", nick="M", sources=sources, given=given, known=[])
    return [f"{f.kind.value}: {f.note}" for f in flaws]


def test_a_right_citation_passes() -> None:
    assert cite("Acme owes $9.2bn [F1], and its test is 1.35x [F2].") == []


def test_a_figure_from_another_fact_is_miscited() -> None:
    assert cite("Acme cut 1,200 jobs and owes $9.2bn [F4].") == [
        "miscited: $9.2bn is in [F1], not in [F4]."
    ]


def test_a_value_inside_the_cited_range_passes() -> None:
    sources, given = prompt_parts("- Used parts sell for about $40-$75.")
    turn = "A used part is ~$60 [F1], a new one more."
    flaws = checks.check_turn(turn, member="m", nick="M", sources=sources, given=given, known=[])
    assert flaws == []
    outside = turn.replace("$60", "$90")
    flaws = checks.check_turn(outside, member="m", nick="M", sources=sources, given=given, known=[])
    assert [f.note for f in flaws] == ["$90 is not in [F1]."]


def test_a_figure_no_fact_holds_is_miscited_not_unsourced() -> None:
    assert cite("Acme owes $7.1bn [F1].") == ["miscited: $7.1bn is not in [F1]."]


def test_an_unknown_tag() -> None:
    assert cite("Acme owes $9.2bn [F9].") == ["unknown_citation: [F9] is not a fact in the prompt."]


def test_figures_from_the_user_or_own_arithmetic_pass_in_a_tagged_sentence() -> None:
    turn = "With your $40k saved and $9.2bn owed [F1], the $3.7bn left after $5.5bn is due [F1]."
    assert cite(turn, user="<Ana> I have $40k saved") == []


def test_untagged_sentences_keep_the_old_check() -> None:
    assert cite("I think Acme owes $7.1bn.") == ["unsourced: $7.1bn"]


def test_facts_command_lists_the_tags() -> None:
    from privy_council.tui.commands import NO_FACTS, facts_list_text

    assert facts_list_text([]) == NO_FACTS
    text = facts_list_text(notes(FIRST, SECOND))
    assert "- [F1] **Debt:**" in text and "[F3] Rates rose" in text
    assert "Missing" not in text


def test_the_export_shows_tags_and_keeps_the_missing_line() -> None:
    from unittest.mock import MagicMock

    from privy_council.export.report import Report

    research = notes(FIRST, SECOND)
    report = MagicMock()  # only meta.runtime.research is read
    report.meta.runtime.research = research
    text = Report.research_text(report, research[0])
    assert text.startswith("- [F1] **Debt:**")
    assert text.endswith("- Missing: the lenders' names")
    assert Report.research_text(report, research[1]).startswith("[F3] Rates rose")


@pytest.mark.parametrize(
    ("turn", "source", "flagged"),
    [
        ("Acme owes $9.2B.", "Acme owes $9.2bn.", False),
        ("Acme owes $9.2bn.", "Acme owes $9.2 billion.", False),
        ("A $1.2T market.", "a $1.2 trillion market", False),
        ("Acme owes $9.2B.", "Acme owes $7.1bn.", True),
        ("Capex of $40-45B.", "capex of $40-45B", False),
        ("Capex of $45bn.", "capex of $40-45B", False),
        # A bare "70B" is a model's size: it neither counts as a figure nor sources one.
        ("Llama 70B is enough.", "", False),
        ("A $70 billion deal.", "Llama 70B runs locally.", True),
    ],
)
def test_billions_written_with_b(turn: str, source: str, flagged: bool) -> None:
    flaws = checks.figure_flaws(turn, source, source, "m")
    assert bool([f for f in flaws if f.kind.value == "unsourced"]) == flagged


def test_the_tag_vouches_for_the_last_amount_before_it() -> None:
    sources, given = prompt_parts("- A 16GB kit was €310 at the second-cheapest shop.")
    turn = "Path B is ~$470: $230 for a CPU + $150 for a board + $90 for a 16GB kit [F1]."
    flaws = checks.check_turn(turn, member="m", nick="M", sources=sources, given=given, known=[])
    assert [f"{f.kind.value}: {f.note}" for f in flaws if f.kind.value == "miscited"] == [
        "miscited: $90 is not in [F1]."
    ]


def test_an_amount_found_elsewhere_in_the_prompt_is_still_miscited() -> None:
    assert cite("Acme owes $5.5bn [F3].", user="<Ana> what about the $5.5bn?") == [
        "miscited: $5.5bn is in [F1], not in [F3]."
    ]


def test_sizes_before_a_tag_are_not_vouched_for() -> None:
    turn = "I assume you keep 48GB, since 96GB of new memory costs too much [F1]."
    assert cite(turn, user="<Ana> I have 96GB and maybe 48GB") == []


def test_each_tag_vouches_for_its_own_amount() -> None:
    assert cite("Acme owes $9.2bn [F1] while rates are 4.5% [F3].") == []
    assert cite("Acme owes $9.2bn [F3] while rates are 4.5% [F1].") == [
        "miscited: $9.2bn is in [F1], not in [F3].",
        "miscited: 4.5% is in [F3], not in [F1].",
    ]


def test_a_tag_does_not_vouch_across_a_clause_break() -> None:
    assert cite("If rates pass 6% or Acme cuts more jobs [F4], sell.") == []


def test_tags_joined_by_a_slash() -> None:
    assert facts.tags("Acme owes $9.2bn [F3/F1].") == ["F3", "F1"]
    assert cite("Acme owes $9.2bn [F3/F1].") == []


def test_a_tag_does_not_vouch_for_a_figure_many_words_back() -> None:
    turn = (
        "A 30% cut is unlikely, since it rests on the strong cash position the filings report [F1]."
    )
    assert cite(turn) == ["unsourced: 30%"]  # the ordinary check, not a miscitation


def test_figures_quoted_from_the_cited_fact_pass_whatever_word_follows() -> None:
    sources, given = prompt_parts(
        "- Output should reach about 48,000-52,000 blue widgets a month.",
        "- About $6B of deposits is expected in the next five months.",
    )
    turn = (
        "Output is 48,000-52,000 widgets a month [F1]. "
        "Deposits (~$6B expected in 5 months [F2]) give a buffer."
    )
    flaws = checks.check_turn(turn, member="m", nick="M", sources=sources, given=given, known=[])
    assert flaws == []


def test_a_figure_the_cited_fact_lacks_keeps_the_unit_check() -> None:
    sources, given = prompt_parts(
        "- Output should reach about 48,000 blue widgets a month.",
        "- Each tool has a 25 GB cache.",
    )
    turn = "Output is 48,000 widgets, fed by 25 km of track [F1]."
    flaws = checks.check_turn(turn, member="m", nick="M", sources=sources, given=given, known=[])
    assert [f.kind.value for f in flaws] == ["wrong_unit"]


def test_tags_written_as_a_span() -> None:
    assert facts.tags("Parts sell for $40-90 [F12\u2013F15].") == ["F12", "F13", "F14", "F15"]
    assert facts.tags("A 6% note [F21-F22].") == ["F21", "F22"]
    assert facts.tags("A typo [F3-F90].") == ["F3", "F90"]
    assert cite("Acme owes $9.2bn [F1-F2].") == []
    assert cite("Acme owes $9.2bn [F2-F4].") == [
        "miscited: $9.2bn is in [F1], not in [F2], [F3], [F4]."
    ]
