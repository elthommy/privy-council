"""Research ledger (M20): repeated lookups, facts not found, source tiers."""

from __future__ import annotations

import pytest

from privy_council.config import PRIMARY_DOMAINS
from privy_council.engine import context, facts, prompts
from privy_council.engine.research_ledger import (
    earlier_queries,
    missing_items,
    not_findable,
    page_for_repeat,
    readable_pages,
    repeat_count,
    repeat_focus,
)
from privy_council.models.session import ResearchNote


def note(
    focus: str, *, auto: bool = True, missing: tuple[str, ...] = (), searches: tuple[str, ...] = ()
) -> ResearchNote:
    """A research note with just what the ledger reads."""
    return ResearchNote(
        focus=focus, text="- x", auto=auto, missing=list(missing), searches=list(searches)
    )


def test_missing_items_from_a_claude_note() -> None:
    text = "- Debt $9.1B (https://example.org)\n**Missing:** split by quarter; covenant terms."
    assert missing_items(text) == ["split by quarter", "covenant terms"]
    assert missing_items("- nothing missing here") == []


def test_missing_items_drops_an_item_the_cap_would_cut() -> None:
    long_item = "verification of a report " + "x" * 200
    assert missing_items(f"Missing: the split; {long_item}; covenants") == ["the split"]


def test_third_lookup_of_the_same_thing_repeats() -> None:
    notes = [
        note("Widget Corp filing debt note maturity table 2027"),
        note("Widget Corp filing debt note maturities 2027 by quarter"),
    ]
    focus = "Widget Corp debt note maturity table by quarter 2027"
    assert repeat_count(focus, notes) == 2
    assert repeat_focus(focus, notes)
    assert not repeat_focus(focus, notes[:1])  # a second try often finds it


def test_a_new_date_is_a_new_lookup() -> None:
    notes = [note("Index close Oct 16 2026"), note("Index official close Oct 16 2026")]
    assert repeat_focus("Index close Oct 16 2026 official", notes)
    assert not repeat_focus("Index close Oct 19 2026", notes)


def test_a_page_read_is_not_a_search() -> None:
    read = note("Index close Oct 16 2026").model_copy(update={"source": "document"})
    notes = [note("Index close Oct 16 2026"), read]
    assert repeat_count("Index official close Oct 16 2026", notes) == 1


def test_only_the_chairs_own_lookups_count() -> None:
    notes = [note("Widget Corp price list", auto=False), note("Widget Corp price list")]
    assert repeat_count("Widget Corp price list", notes) == 1


def test_not_findable_counts_and_drops_what_a_later_lookup_found() -> None:
    notes = [
        note("Widget filing", missing=("maturities by quarter", "covenant terms")),
        note("Widget maturities by quarter", missing=("quarterly maturities split",)),
        note("Widget covenant terms"),  # aimed at the covenants, reported nothing missing
    ]
    assert not_findable(notes) == [("quarterly maturities split", 2)]


def test_earlier_queries_without_repeats() -> None:
    notes = [note("a", searches=("q1", "q2")), note("b", searches=("q2", "q3"))]
    assert earlier_queries(notes) == ["q1", "q2", "q3"]


def test_facts_text_lists_what_is_not_findable_first() -> None:
    notes = [note("Widget", missing=("the quarterly split",))]
    text = context.facts_text(notes) or ""
    assert text.startswith("Not findable so far: the quarterly split (missing after one search).")


def test_not_findable_keeps_the_most_often_missing() -> None:
    words = ["harbor", "meadow", "copper", "violet", "granite", "falcon", "willow"]
    notes = [note(f"lookup {w}", missing=(w, "the debt table")) for w in words]
    found = not_findable(notes)
    assert found[0] == ("the debt table", 7)
    assert len(found) == 5
    assert found[1] == ("willow", 1)  # then newest first


def test_tier_from_the_urls() -> None:
    assert facts.tier("- Rate 2% (https://www.sec.gov/x)", PRIMARY_DOMAINS) == "primary"
    assert facts.tier("- Rate 2% [stats](https://www.insee.fr/fr/x)", PRIMARY_DOMAINS) == (
        "primary"
    )
    mixed = "- Rate 2% (https://ecb.europa.eu/a, https://blog.example.com/b)"
    assert facts.tier(mixed, PRIMARY_DOMAINS) == "secondary"
    assert facts.tier("- Rate 2%, no source", PRIMARY_DOMAINS) == "secondary"
    assert facts.tier("- (https://notgov.com/x)", PRIMARY_DOMAINS) == "secondary"


def test_only_primary_facts_carry_the_tier_and_tags_still_parse() -> None:
    notes = ["- Debt $9B (https://www.sec.gov/f)\n- Debt $8B (https://news.example.com/a)"]
    (tagged,) = facts.tagged_notes(notes, PRIMARY_DOMAINS, "primary")
    assert tagged.splitlines() == [
        "- [F1, primary] Debt $9B (https://www.sec.gov/f)",
        "- [F2] Debt $8B (https://news.example.com/a)",
    ]
    section = prompts.section(prompts.TITLE_FACTS, tagged)
    assert facts.in_prompt(section) == {
        "F1": "Debt $9B (https://www.sec.gov/f)",
        "F2": "Debt $8B (https://news.example.com/a)",
    }
    assert facts.tags("debt was $9B [F1, primary]") == ["F1"]


GOV = "https://stats.example.gov/release/current.htm"


def cited(focus: str, text: str, *, source: str = "claude", auto: bool = True) -> ResearchNote:
    """A research note citing pages."""
    return ResearchNote(focus=focus, text=text, auto=auto, source=source)  # type: ignore[arg-type]


def test_readable_pages_are_the_official_html_pages_newest_first() -> None:
    notes = [
        cited("deposits", f"- Deposits 120 ({GOV})\n- Blog view https://blog.example.com/x"),
        cited("loans", "- Loans 80 https://stats.example.gov/release/h8.pdf."),
        cited("rates", "- Rate 3 % (https://bank.example.gov/rates/)"),
        cited("split by type", f"- Table 2: 70 {GOV}", source="document", auto=False),
    ]
    assert readable_pages(notes, PRIMARY_DOMAINS) == [
        (GOV, ["split by type"]),
        ("https://bank.example.gov/rates/", []),
    ]
    assert readable_pages(notes, []) == []
    assert [u for u, _ in readable_pages(notes, PRIMARY_DOMAINS, ["stats.example.gov"])] == [
        "https://bank.example.gov/rates/"
    ]


@pytest.mark.parametrize(
    ("url", "readable"),
    [
        ("https://stats.example.gov/schedule/2027/05_sched.htm", False),
        ("https://stats.example.gov/release-calendar.htm", False),
        ("https://bank.example.gov/calendars/fomc.htm", False),
        ("https://stats.example.gov/news.release/empsit.t01.htm", True),
        ("https://stats.example.gov/tables/rescheduled-loans.htm", True),
    ],
)
def test_a_release_calendar_is_not_readable(url: str, readable: bool) -> None:
    notes = [cited("release", f"- Released on 4 May ({url})")]
    assert bool(readable_pages(notes, PRIMARY_DOMAINS)) is readable


def test_a_repeated_lookup_gets_the_official_page_its_searches_cite() -> None:
    notes = [
        cited("Widget Corp filing debt note maturities 2027", f"- Note cut off ({GOV})"),
        cited("Widget Corp filing debt maturities 2027 principal", "- https://news.example.com/a"),
        cited("unrelated market close", "- https://other.example.gov/close.htm"),
    ]
    focus = "Widget Corp debt note maturities 2027"
    assert page_for_repeat(focus, notes, PRIMARY_DOMAINS) == GOV
    read = cited("maturities", f"- 2027: 2,222 {GOV}", source="document", auto=False)
    assert page_for_repeat(focus, [*notes, read], PRIMARY_DOMAINS) is None
    assert page_for_repeat("gold price today", notes, PRIMARY_DOMAINS) is None
    assert page_for_repeat(focus, notes, PRIMARY_DOMAINS, ["example.gov"]) is None
