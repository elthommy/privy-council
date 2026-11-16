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


def top(query: str, passages: Sequence[str], count: int, max_chars: int) -> list[int]:
    """Return the indexes of the best passages, best first, within `max_chars` in all.

    Passages scoring 0 (no word of the query) are never picked. The best one is
    always kept, cut by the caller if it is longer than `max_chars`.
    """
    ranks = scores(query, passages)
    ranked = sorted((i for i, s in enumerate(ranks) if s > 0), key=lambda i: -ranks[i])
    picked: list[int] = []
    size = 0
    for index in ranked[:count]:
        if picked and size + len(passages[index]) > max_chars:
            continue
        picked.append(index)
        size += len(passages[index])
    return picked
