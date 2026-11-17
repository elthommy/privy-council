from __future__ import annotations

from privy_council.tui.commands import (
    PROPOSAL_FACTS_HINT,
    PROPOSAL_FACTS_TITLE,
    PROPOSAL_REWRITE_HINT,
    PROPOSAL_REWRITE_TITLE,
    Command,
    parse,
    proposal_text,
)


def test_parse() -> None:
    assert parse("   ") is None
    assert parse("hello there") == Command("say", "hello there")
    assert parse("/STEP") == Command("step")
    assert parse("/say  I forgot something ") == Command("say", "I forgot something")
    assert parse("/ask @Rigor worst case?") == Command("ask", "@Rigor worst case?")


def test_parse_a_wrapped_copy_of_a_command() -> None:
    wrapped = "/research\n     oil price\n     forecasts for 2027"
    assert parse(wrapped) == Command("research", "oil price forecasts for 2027")
    assert parse("/read\thttps://example.org  rates") == Command(
        "read", "https://example.org rates"
    )
    assert parse("/say first\nsecond") == Command("say", "first\nsecond")  # a paste stays whole


def test_proposal_text() -> None:
    facts = proposal_text(["- 2026-10-17 (finance): The user rents.", "- b"], None)
    assert facts.splitlines() == [
        PROPOSAL_FACTS_TITLE,
        "  1. 2026-10-17 (finance): The user rents.",
        "  2. b",
        PROPOSAL_FACTS_HINT,
    ]
    merged = proposal_text([], "- merged")
    assert merged.splitlines() == [PROPOSAL_REWRITE_TITLE, "- merged", PROPOSAL_REWRITE_HINT]
