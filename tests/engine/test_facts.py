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
