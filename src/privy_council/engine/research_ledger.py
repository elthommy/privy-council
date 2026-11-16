"""Research ledger (M20): what the Chair already searched for, and what stayed missing.

Pure functions, no I/O. The rule for a repeated lookup was set on the saved sessions
(2026-11-15, 40 hand-labelled lookups whose focus shared half its words with an
earlier one): about half of the second tries found something, every third try or
later found nothing new. So a lookup is skipped only after two earlier ones, and
never when it names a new date or number ("close on Oct 19" after "close on Oct 16").
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from privy_council.engine.facts import URL
from privy_council.engine.local_research import MAX_MISSING_CHARS, missing_line, normalize_url
from privy_council.engine.moderation import _fact_words
from privy_council.models.session import ResearchNote

# Share of the new focus's words an earlier focus must hold to count as the same lookup.
REPEAT_OVERLAP = 0.5
# Earlier lookups of the same thing after which the Chair's own lookup is skipped.
REPEAT_SKIP_AFTER = 2
# The research prompt lists at most this many earlier queries (newest kept).
MAX_EARLIER_QUERIES = 15
# The facts section lists at most this many facts not found, the most often missing
# first: the 2026-11-15 live check reached 8, four of them wordings of one fact.
MAX_NOT_FINDABLE = 5
_NUMBER = re.compile(r"\d+")
_ITEM_SPLIT = re.compile(r"\s*;\s*")


def missing_items(note_text: str) -> list[str]:
    """Return the facts a note's "Missing:" line lists, one per item ("" items dropped).

    Whole items only, up to MAX_MISSING_CHARS in all: a cap cutting an item left
    a dangling "report (" fragment in a live check (2026-11-15).
    """
    items: list[str] = []
    size = 0
    for item in _ITEM_SPLIT.split(missing_line(note_text).rstrip(".")):
        size += len(item)
        if size > MAX_MISSING_CHARS:
            break
        if item:
            items.append(item)
    return items


def _overlap(words: set[str], other: set[str]) -> float:
    """Return the share of `words` found in `other` (0 when `words` is empty)."""
    return len(words & other) / len(words) if words else 0.0


def repeat_count(focus: str, notes: Sequence[ResearchNote]) -> int:
    """Return how many of the Chair's earlier lookups searched for this focus.

    0 when the focus names a date or number none of them did: a new day's close is a
    new lookup, even in the same words.
    """
    words = _fact_words(focus)
    matched = [
        n for n in notes if n.auto and _overlap(words, _fact_words(n.focus)) >= REPEAT_OVERLAP
    ]
    seen = {num for n in matched for num in _NUMBER.findall(n.focus)}
    if set(_NUMBER.findall(focus)) - seen:
        return 0
    return len(matched)


def repeat_focus(focus: str, notes: Sequence[ResearchNote]) -> bool:
    """Tell whether the Chair's lookup repeats two earlier ones and should be skipped."""
    return repeat_count(focus, notes) >= REPEAT_SKIP_AFTER


def _similar(words: set[str], other: set[str]) -> bool:
    """Tell whether two items name the same fact (half the words of either)."""
    return _overlap(words, other) >= REPEAT_OVERLAP or _overlap(other, words) >= REPEAT_OVERLAP


def _found_later(words: set[str], later: Sequence[ResearchNote]) -> bool:
    """Tell whether a later lookup aimed at this fact and didn't report it missing."""
    return any(
        _overlap(words, _fact_words(n.focus)) >= REPEAT_OVERLAP
        and not any(_similar(words, _fact_words(m)) for m in n.missing)
        for n in later
    )


def not_findable(notes: Sequence[ResearchNote]) -> list[tuple[str, int]]:
    """Return the facts still missing after the lookups, most often missing first.

    Each comes with the number of notes that reported it missing (newest wording; at
    most MAX_NOT_FINDABLE). A fact a later lookup aimed at without reporting it
    missing counts as found.
    """
    out: list[tuple[str, int]] = []
    seen: list[set[str]] = []
    for index in reversed(range(len(notes))):
        for item in notes[index].missing:
            words = _fact_words(item)
            if not words or any(_similar(words, s) for s in seen):
                continue
            seen.append(words)
            if _found_later(words, notes[index + 1 :]):
                continue
            count = sum(any(_similar(words, _fact_words(m)) for m in n.missing) for n in notes)
            out.append((item, count))
    return sorted(out, key=lambda pair: -pair[1])[:MAX_NOT_FINDABLE]  # stable: newest first


def cited_url(url: str, notes: Sequence[ResearchNote]) -> bool:
    """Tell whether a URL appears in the session's facts or was read by a search (M21)."""
    wanted = normalize_url(url)
    return any(
        normalize_url(found) == wanted
        for n in notes
        for found in [*URL.findall(n.text), *n.fetched]
    )


def earlier_queries(notes: Sequence[ResearchNote]) -> list[str]:
    """Return the web queries earlier lookups ran, without repeats, newest last (capped)."""
    queries = list(dict.fromkeys(q for n in notes for q in n.searches))
    return queries[-MAX_EARLIER_QUERIES:]
