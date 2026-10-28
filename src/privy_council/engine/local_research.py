"""Local web search (SearXNG): the moderator's queries, the merged hits, the summary check.

Pure functions; `ResearchFlow` runs the calls. The summary's figures and URLs are
checked against the hits, because research counts as "sourced" for the members'
flaw checks: a figure a local model got wrong while summarizing would otherwise
become a trusted fact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from privy_council.backends.base import BackendOutputError, CompletionResult
from privy_council.engine import prompts
from privy_council.engine.checks import figure_flaws
from privy_council.engine.moderation import DecisionError, json_object
from privy_council.models.flaw import FlawKind
from privy_council.models.research import ResearchQueries
from privy_council.search.base import SearchResult

MAX_QUERY_CHARS = 150
SHORT_QUERY_WORDS = 4
# Words that narrow a query without naming its subject: dropped for the retry.
_FILLER_WORDS = (
    "official officially site website latest current recent report data statistics "
    "release figures rates the of a an and for in on to"
)
_FILLER = frozenset(_FILLER_WORDS.split())
_YEAR = re.compile(r"^(?:19|20)\d{2}$")


_MISSING = re.compile(r"^\s*(?:[-*•]\s*)?\**missing\**\s*:\s*\**\s*(?P<facts>.+?)\s*$", re.I | re.M)
MAX_MISSING_CHARS = 200


def missing_facts(note: str) -> str:
    """Return the facts a summary says the results lack ("Missing: …" line), or ""."""
    found = _MISSING.findall(note)
    return " ".join(found[-1].split())[:MAX_MISSING_CHARS] if found else ""


def without_missing(note: str) -> str:
    """Return a note without its "Missing: …" lines, for prompts.

    Each line is judged against its own search's hits only, so a later note can say
    a fact is missing that an earlier note gives (2026-10-26 gacha run).
    """
    return "\n".join(line for line in note.splitlines() if not _MISSING.match(line))


class NoSearchHits(BackendOutputError):
    """Every query came back empty, even shortened: the debate goes on without it."""


def shorter_query(query: str) -> str:
    """Return a query without filler words or years, cut to its first few words.

    "ESM lending capacity firepower 2026 official" found nothing where "ESM lending
    capacity" finds plenty: the engines treat every word as required.
    """
    words = [w for w in query.split() if w.lower() not in _FILLER and not _YEAR.match(w)]
    return " ".join(words[:SHORT_QUERY_WORDS])


def shorter_queries(queries: Sequence[str]) -> list[str]:
    """Return the shortened queries that differ from what was already searched."""
    out: list[str] = []
    seen = {q.lower() for q in queries}
    for query in queries:
        short = shorter_query(query)
        if short and short.lower() not in seen:
            seen.add(short.lower())
            out.append(short)
    return out


QUERIES_SCHEMA: dict[str, Any] = ResearchQueries.model_json_schema()
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_URL_TAIL = ".,;:!?*_`"


def parse_queries(result: CompletionResult, limit: int) -> list[str]:
    """Return the moderator's queries, cleaned and deduped; raise DecisionError if none."""
    try:
        reply = ResearchQueries.model_validate(json_object(result))
    except ValidationError as exc:
        raise DecisionError("; ".join(e["msg"] for e in exc.errors())) from exc
    out: list[str] = []
    for query in reply.queries:
        text = " ".join(query.split())[:MAX_QUERY_CHARS]
        if text and text.lower() not in {q.lower() for q in out}:
            out.append(text)
    if not out:
        raise DecisionError("every query was empty")
    return out[:limit]


def fallback_query(focus: str, topic: str, standing: str = "") -> str:
    """Return the one query used when the moderator gives none: the focus, else the topic.

    `standing`: the council's own first-search focus, a description like the default
    one rather than a query.
    """
    standard = focus == prompts.RESEARCH_DEFAULT_FOCUS or (standing and focus == standing)
    text = topic if standard else focus
    return " ".join(text.split())[:MAX_QUERY_CHARS]


def normalize_url(url: str) -> str:
    """Return a URL in the form used to compare them (no trailing slash or punctuation)."""
    return url.strip().rstrip(_URL_TAIL).rstrip("/")


def merge_results(batches: Sequence[Sequence[SearchResult]], limit: int) -> list[SearchResult]:
    """Interleave the queries' hits (each query's best first), dropping repeated URLs."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for rank in range(max((len(b) for b in batches), default=0)):
        for batch in batches:
            if rank >= len(batch):
                continue
            key = normalize_url(batch[rank].url)
            if key not in seen:
                seen.add(key)
                out.append(batch[rank])
            if len(out) >= limit:
                return out
    return out


def render_results(results: Sequence[SearchResult]) -> str:
    """Render the hits, numbered, for the summary prompt."""
    return "\n\n".join(
        prompts.LOCAL_RESULT_ENTRY.format(
            index=i,
            title=r.title or r.url,
            date=f" ({r.published})" if r.published else "",
            url=r.url,
            snippet=r.snippet or "(no snippet)",
        )
        for i, r in enumerate(results, start=1)
    )


def _blocks(text: str) -> list[str]:
    """Split a summary into bullets, each with its continuation lines."""
    blocks: list[str] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if _BULLET.match(line) or not blocks:
            blocks.append(line)
        else:
            blocks[-1] += "\n" + line
    return blocks


def _backed(block: str, urls: set[str], sources: str) -> bool:
    """Whether a bullet cites only known URLs and only figures found in the sources."""
    if any(normalize_url(u) not in urls for u in _URL.findall(block)):
        return False
    body = _URL.sub(" ", block)  # a URL's digits aren't figures
    flaws = figure_flaws(body, sources, "", "moderator")
    return not any(f.kind is FlawKind.UNSOURCED for f in flaws)


def check_summary(
    text: str, results: Sequence[SearchResult], context: str = ""
) -> tuple[str, list[str]]:
    """Keep the bullets backed by the hits; return the kept text and the dropped bullets.

    `context` (the question and the focus) also counts as a source: a bullet may
    restate the user's own figure.
    """
    urls = {normalize_url(r.url) for r in results}
    sources = "\n".join([context, *(f"{r.title}\n{r.snippet}" for r in results)])
    kept: list[str] = []
    dropped: list[str] = []
    for block in _with_missing_prefix(_blocks(text)):
        if _copies_example(block):
            continue
        missing = _MISSING.match(block) is not None  # not a fact: never dropped
        (kept if missing or _backed(block, urls, sources) else dropped).append(block)
    return "\n".join(kept), dropped


def _copies_example(block: str) -> bool:
    """Whether a "Missing:" line copies the summary prompt's example word for word."""
    match = _MISSING.match(block)
    folded = " ".join(match["facts"].lower().rstrip(".").split()) if match else ""
    return folded == prompts.LOCAL_MISSING_EXAMPLE.lower()


def _with_missing_prefix(blocks: list[str]) -> list[str]:
    """Restore the "Missing:" prefix of a last bullet that cites no result.

    Every fact bullet cites a URL; Super once ended with the missing facts as a plain
    bullet ("- Wuthering Waves pity and 50/50 rules …", 2026-10-27 gacha run), so the
    follow-up search never ran and the line stayed in prompts as a fact.
    """
    if len(blocks) < 2 or any(_MISSING.match(b) for b in blocks) or _URL.search(blocks[-1]):
        return blocks
    facts = " ".join(_BULLET.sub("", blocks[-1]).split()).rstrip(".")
    return [*blocks[:-1], f"- Missing: {facts}"]
