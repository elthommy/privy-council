"""Research ledger (M20): repeated lookups, facts not found, source tiers."""

from __future__ import annotations

from privy_council.config import PRIMARY_DOMAINS
from privy_council.engine import context, facts, prompts
from privy_council.engine.research_ledger import (
    earlier_queries,
    missing_items,
    not_findable,
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
    assert text.startswith("Not findable so far: the quarterly split (searched once).")


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
