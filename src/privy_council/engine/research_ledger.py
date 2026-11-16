"""Research ledger (M20): what the Chair already searched for, and what stayed missing.

Pure functions, no I/O. The rule for a repeated lookup was set on the saved sessions
(2026-11-15, 40 hand-labelled lookups whose focus shared half its words with an
earlier one): about half of the second tries found something, every third try or
later found nothing new. So a lookup is skipped only after two earlier ones, and
never when it names a new date or number ("close on Oct 19" after "close on Oct 16").
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from urllib.parse import urlsplit

from privy_council.engine.facts import URL, on_domains, tier
from privy_council.engine.local_research import (
    MAX_MISSING_CHARS,
    URL_TAIL,
    missing_line,
    normalize_url,
)
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
# The moderator's task lists at most this many official pages it may read (newest first).
MAX_READABLE_PAGES = 6
# The document reader takes HTML and plain text only.
_UNREADABLE = (".pdf", ".xls", ".xlsx", ".csv", ".zip", ".doc", ".docx", ".ppt", ".pptx")
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
    return len(repeat_notes(focus, notes))


def repeat_notes(focus: str, notes: Sequence[ResearchNote]) -> list[ResearchNote]:
    """Return the Chair's earlier lookups of this focus (none if it names a new number)."""
    words = _fact_words(focus)
    matched = [
        n for n in notes if n.auto and _overlap(words, _fact_words(n.focus)) >= REPEAT_OVERLAP
    ]
    seen = {num for n in matched for num in _NUMBER.findall(n.focus)}
    if set(_NUMBER.findall(focus)) - seen:
        return []
    return matched


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


def _note_urls(note: ResearchNote) -> dict[str, str]:
    """Return the URLs a note cites or fetched, as written, by normalized form (in order)."""
    out: dict[str, str] = {}
    for url in [*URL.findall(note.text), *note.fetched]:
        out.setdefault(normalize_url(url), url.strip().rstrip(URL_TAIL))
    return out


def _readable(url: str, primary_domains: Collection[str], refusing: Collection[str]) -> bool:
    """Tell whether a URL is an official page the reader can take.

    No PDF or data file, and no site that refuses the reader (`refusing`).
    """
    path = urlsplit(url).path.lower()
    data_file = path.endswith(_UNREADABLE) or path.rstrip("/").endswith("/pdf")
    official = tier(url, primary_domains) == "primary"
    return official and not data_file and not on_domains(url, refusing)


def _read_pages(notes: Sequence[ResearchNote]) -> dict[str, list[str]]:
    """Return what the document reader looked for, by normalized URL of the page read."""
    read: dict[str, list[str]] = {}
    for n in notes:
        if n.source != "document":
            continue
        for key in _note_urls(n):
            if n.focus not in read.setdefault(key, []):
                read[key].append(n.focus)
    return read


def readable_pages(
    notes: Sequence[ResearchNote],
    primary_domains: Collection[str],
    refusing: Collection[str] = (),
) -> list[tuple[str, list[str]]]:
    """Return the official pages the facts cite, each with what the reader looked for there.

    For the moderator's task (M21 live check, 2026-11-16: it searched twice more for a
    table that a page in its facts held). Newest note first, at most MAX_READABLE_PAGES.
    """
    read = _read_pages(notes)
    pages: dict[str, str] = {}
    for n in reversed(notes):
        for key, url in _note_urls(n).items():
            if _readable(url, primary_domains, refusing):
                pages.setdefault(key, url)
    return [(url, read.get(key, [])) for key, url in pages.items()][:MAX_READABLE_PAGES]


def page_for_repeat(
    focus: str,
    notes: Sequence[ResearchNote],
    primary_domains: Collection[str],
    refusing: Collection[str] = (),
) -> str | None:
    """Return an official page the earlier lookups of `focus` cite and nobody read yet.

    A third search for the same thing finds nothing new (see the module docstring);
    reading the page those searches cite is the one thing not tried. Newest lookup
    first; None when they cite no such page or each was read already.
    """
    read = _read_pages(notes)
    for n in reversed(repeat_notes(focus, notes)):
        for key, url in _note_urls(n).items():
            if _readable(url, primary_domains, refusing) and key not in read:
                return url
    return None


def earlier_queries(notes: Sequence[ResearchNote]) -> list[str]:
    """Return the web queries earlier lookups ran, without repeats, newest last (capped)."""
    queries = list(dict.fromkeys(q for n in notes for q in n.searches))
    return queries[-MAX_EARLIER_QUERIES:]
