"""Debate language detection (M10): French or English, from the question. Pure, no I/O.

Counts frequent function words of each language; no dependency, and good enough
for a question of a sentence or two. Anything unclear stays English, the language
of every prompt.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

Language = Literal["en", "fr"]
LANGUAGES: tuple[Language, ...] = get_args(Language)
LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "fr": "French"}

# Short, frequent words that are rare in the other language. Words both share
# ("a", "on", "son") are left out.
_FRENCH_WORDS = (
    "le la les un une des du de au aux et est sont ou mais donc que qui quoi quel quelle "
    "quels quelles je tu il elle nous vous ils elles mon ma mes ton ta tes sa ses notre "
    "votre leur leurs ce cette ces pour avec sans dans sur sous chez par pas ne plus moins "
    "très trop aussi comme comment pourquoi quand combien dois doit faut peut puis suis "
    "être avoir fait faire c'est j'ai qu'il d'un d'une l'on n'est vie argent"
)
_ENGLISH_WORDS = (
    "the an and is are was were or but so that which what who whom whose how why when "
    "i you he she we they my your his her our their this these those for with without "
    "into from about of to not no more less very too also should would could can will "
    "do does did have has had be been being it's i'm don't money life"
)
_FRENCH = frozenset(_FRENCH_WORDS.split())
_ENGLISH = frozenset(_ENGLISH_WORDS.split())
_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+(?:['\u2019][a-zàâäçéèêëîïôöùûüÿœæ]+)?")
_ACCENTED = re.compile(r"[àâçéèêëîïôùûœ]")
# Accents count for a little: French text has many, English almost none.
_ACCENT_WEIGHT = 0.5
_ALIASES: dict[str, Language] = {"francais": "fr", "français": "fr", "anglais": "en"}


def detect_language(text: str) -> Language:
    """Return "fr" when the text is clearly French, else "en"."""
    lowered = text.lower()
    words = [w.replace("\u2019", "'") for w in _WORD.findall(lowered)]
    french = sum(w in _FRENCH for w in words) + _ACCENT_WEIGHT * len(_ACCENTED.findall(lowered))
    english = sum(w in _ENGLISH for w in words)
    return "fr" if french > english and french >= 2 else "en"


def parse_language(text: str) -> Language | None:
    """Parse a language code or name ("fr", "French", "français"); None if unknown."""
    key = text.strip().lower()
    for code in LANGUAGES:
        if key in (code, LANGUAGE_NAMES[code].lower()):
            return code
    return _ALIASES.get(key)
