"""Debate language detection and parsing."""

from __future__ import annotations

import pytest

from privy_council.util.language import detect_language, parse_language


@pytest.mark.parametrize(
    "text",
    [
        "Dois-je rembourser mon crédit immobilier avant d'investir ?",
        "Est-ce qu'une vie confortable est une bonne vie ?",
        "Comment mieux dormir avec un emploi du temps chargé ?",
        "L\u2019ambition est-elle une vertu ou un vice ?",  # typographic apostrophe
        "Café ou thé ?",
        "Je gagne 3000 € par mois, que faire de mon épargne ?",
    ],
)
def test_french_questions(text: str) -> None:
    assert detect_language(text) == "fr"


@pytest.mark.parametrize(
    "text",
    [
        "Should I max my PEA before building a bigger emergency fund?",
        "Is a life of comfort a good life?",
        "Should I open a PEA or an assurance-vie?",  # French terms, English question
        "How can I sleep better with a busy schedule?",
        "Should I buy a Renault Clio or a Peugeot 208?",
    ],
)
def test_english_questions(text: str) -> None:
    assert detect_language(text) == "en"


@pytest.mark.parametrize("text", ["", "PEA?", "42", "Lyon", "Paris ou"])
def test_unclear_text_stays_english(text: str) -> None:
    assert detect_language(text) == "en"


def test_parse_language() -> None:
    assert parse_language("fr") == "fr"
    assert parse_language(" FR ") == "fr"
    assert parse_language("French") == "fr"
    assert parse_language("français") == "fr"
    assert parse_language("en") == "en"
    assert parse_language("english") == "en"
    assert parse_language("de") is None
    assert parse_language("") is None
