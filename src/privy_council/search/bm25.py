"""BM25 ranking of a page's passages against what to look for (M21). Pure, no I/O.

Lexical only: no embedding model, so no GPU swap. Numbers are words too, so "2027"
or "Note 10" pull the passage that names them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

K1 = 1.5
B = 0.75
# A statistics release repeats one table for "Small Domestically Chartered" banks,
# "Large" ones, "Not seasonally adjusted"... with the same rows (H.8, 2026-11-16).
# A passage's score is multiplied by 1 + this per distinctive label word the query
# asks for, and divided by 1 + this per one it doesn't: the table wanted is the one
# whose title qualifies it as the query does.
QUALIFIER_PENALTY = 0.3
_WORD = re.compile(r"[^\W_]+")


def words(text: str) -> list[str]:
    """Return a text's lowercased words and numbers."""
    return _WORD.findall(text.lower())


def scores(query: str, passages: Sequence[str]) -> list[float]:
    """Return each passage's BM25 score for the query."""
    docs = [Counter(words(p)) for p in passages]
    if not docs:
        return []
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(docs) or 1.0
    idf = {}
    for term in set(words(query)):
        having = sum(1 for d in docs if term in d)
        idf[term] = math.log(1 + (len(docs) - having + 0.5) / (having + 0.5))
    out: list[float] = []
    for doc, length in zip(docs, lengths, strict=True):
        norm = K1 * (1 - B + B * length / average)
        out.append(sum(w * doc[t] * (K1 + 1) / (doc[t] + norm) for t, w in idf.items() if doc[t]))
    return out


def qualifiers(query: str, labels: Sequence[str]) -> list[tuple[int, int]]:
    """Return, per label, its distinctive words (in under half the labels): (asked, not asked)."""
    sets = [set(words(label)) for label in labels]
    common = Counter(w for s in sets for w in s)
    asked = set(words(query))
    out = []
    for s in sets:
        rare = [w for w in s if common[w] < len(sets) / 2 and not w.isdigit()]
        hits = sum(w in asked for w in rare)
        out.append((hits, len(rare) - hits))
    return out


def top(
    query: str,
    passages: Sequence[str],
    count: int,
    max_chars: int,
    labels: Sequence[str] | None = None,
) -> list[int]:
    """Return the indexes of the best passages, best first, within `max_chars` in all.

    Passages scoring 0 (no word of the query) are never picked. The best one is
    always kept, cut by the caller if it is longer than `max_chars`. `labels` (a
    passage's heading and subtitle) raise a passage for each distinctive word the
    query asks for and lower it for each one the query doesn't.
    """
    ranks = scores(query, passages)
    if labels is not None:
        ranks = [
            s * (1 + QUALIFIER_PENALTY * hits) / (1 + QUALIFIER_PENALTY * misses)
            for s, (hits, misses) in zip(ranks, qualifiers(query, labels), strict=True)
        ]
    ranked = sorted((i for i, s in enumerate(ranks) if s > 0), key=lambda i: -ranks[i])
    picked: list[int] = []
    size = 0
    for index in ranked[:count]:
        if picked and size + len(passages[index]) > max_chars:
            continue
        picked.append(index)
        size += len(passages[index])
    return picked
