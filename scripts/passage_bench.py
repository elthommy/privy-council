"""Score ways to pick a long page's passages for the document reader (M21), offline.

No model call. Pages are saved once with `fetch` (a real request through the URL
guard, only the URL leaves the PC), then `score` runs each option on a queries file:

    uv run python scripts/passage_bench.py fetch URL NAME      # -> PAGES/NAME.html
    uv run python scripts/passage_bench.py score [--queries FILE] [--pages DIR] [-v]

The queries file (TOML) is a list of `[[query]]` with `page` (a file in PAGES),
`look_for` (as the Chair would write it) and `expect`: strings one picked passage must
hold together, e.g. a table's title and one of its rows. A query passes when a
passage given to the summary holds them all; "first" is when the best passage
alone does.

Options:
  bm25          BM25 on the passage text alone
  labels        + the label rule of `search/bm25.py` (what the reader does now)
  labels-16k    the same with 12 passages in 16,000 characters
  fields        the label scored as a field of its own, added to the text's score
  toc-ceiling   not a ranking: whether a table-of-contents step *could* pick the
                expected passage, i.e. its label is one no other passage shares
"""

from __future__ import annotations

import argparse
import asyncio
import tomllib
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from privy_council.config import DocumentsConfig
from privy_council.search import bm25
from privy_council.search.extract import Chunk, chunks, html_blocks, text_blocks
from privy_council.search.fetch import fetch_page

PAGES = Path("data/bench/m21/pages")
QUERIES = Path("data/bench/m21/queries.toml")
# Weight of the label's own BM25 score in the "fields" option.
FIELD_WEIGHT = 2.0

Picker = Callable[[str, Sequence[Chunk]], list[int]]


def _cut(path: Path) -> list[Chunk]:
    """Return a saved page's passages, as the reader cuts them."""
    text = path.read_text(encoding="utf-8")
    return chunks(text_blocks(text) if path.suffix == ".txt" else html_blocks(text))


def _budget(ranked: Sequence[int], sizes: Sequence[int], count: int, max_chars: int) -> list[int]:
    """Keep the best passages within the budget, as `bm25.top` does."""
    picked: list[int] = []
    size = 0
    for index in ranked[:count]:
        if picked and size + sizes[index] > max_chars:
            continue
        picked.append(index)
        size += sizes[index]
    return picked


def pick_bm25(query: str, cut: Sequence[Chunk]) -> list[int]:
    """Plain BM25 on the rendered passages."""
    return bm25.top(query, [c.render() for c in cut], 6, 8000)


def pick_labels(query: str, cut: Sequence[Chunk]) -> list[int]:
    """BM25 with the label rule: the reader's current choice."""
    return bm25.top(query, [c.render() for c in cut], 6, 8000, [c.label() for c in cut])


def pick_labels_wide(query: str, cut: Sequence[Chunk]) -> list[int]:
    """The label rule with twice the budget."""
    return bm25.top(query, [c.render() for c in cut], 12, 16000, [c.label() for c in cut])


def pick_fields(query: str, cut: Sequence[Chunk]) -> list[int]:
    """BM25 on the text plus a weighted BM25 on the labels, no penalty."""
    rendered = [c.render() for c in cut]
    body = bm25.scores(query, rendered)
    label = bm25.scores(query, [c.label() for c in cut])
    total = [b + FIELD_WEIGHT * s if b > 0 else 0.0 for b, s in zip(body, label, strict=True)]
    ranked = sorted((i for i, s in enumerate(total) if s > 0), key=lambda i: -total[i])
    return _budget(ranked, [len(r) for r in rendered], 6, 8000)


OPTIONS: dict[str, Picker] = {
    "bm25": pick_bm25,
    "labels": pick_labels,
    "labels-16k": pick_labels_wide,
    "fields": pick_fields,
}


def _holds(texts: Sequence[str], expect: Sequence[str]) -> bool:
    """Tell whether one of the texts holds every expected string."""
    return any(all(e in text for e in expect) for text in texts)


def _toc_ceiling(cut: Sequence[Chunk], expect: Sequence[str]) -> bool:
    """Tell whether some passage holding `expect` has a label no other passage shares."""
    counts = Counter(c.label() for c in cut)
    return any(counts[c.label()] == 1 for c in cut if _holds([c.render()], expect))


def score(queries: Path, pages: Path, verbose: bool) -> None:
    """Print each option's passes over the queries file."""
    items = tomllib.loads(queries.read_text(encoding="utf-8"))["query"]
    passed: Counter[str] = Counter()
    first: Counter[str] = Counter()
    cuts: dict[str, list[Chunk]] = {}
    for item in items:
        cut = cuts.setdefault(item["page"], _cut(pages / item["page"]))
        marks = []
        for name, picker in OPTIONS.items():
            picked = [cut[i].render()[:8000] for i in picker(item["look_for"], cut)]
            ok, top = _holds(picked, item["expect"]), _holds(picked[:1], item["expect"])
            passed[name] += ok
            first[name] += top
            marks.append(f"{name}={'1st' if top else 'ok' if ok else 'MISS'}")
        ceiling = _toc_ceiling(cut, item["expect"])
        passed["toc-ceiling"] += ceiling
        if verbose:
            print(f"{item['page']}: {item['look_for']}\n   {' '.join(marks)} toc={ceiling}")
    print(f"\n{len(items)} queries on {len(cuts)} pages")
    for name in [*OPTIONS, "toc-ceiling"]:
        best = f", best passage alone {first[name]}" if name in OPTIONS else ""
        print(f"  {name:12} {passed[name]}/{len(items)}{best}")
    for page, cut in cuts.items():
        toc = sum(len(c.label()) + 6 for c in cut)
        print(f"  {page}: {len(cut)} passages, table of contents {toc:,} chars")


async def fetch(url: str, name: str, pages: Path) -> None:
    """Save a public page through the reader's URL guard."""
    page = await fetch_page(url, DocumentsConfig(enabled=True))
    suffix = ".txt" if page.content_type == "text/plain" else ".html"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / f"{name}{suffix}").write_text(page.text, encoding="utf-8")
    print(f"{page.url}: {len(page.text):,} chars -> {pages / (name + suffix)}")


def main() -> None:
    """Run `fetch` or `score`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("fetch", help="save a page")
    get.add_argument("url")
    get.add_argument("name")
    run = sub.add_parser("score", help="score the options")
    run.add_argument("--queries", type=Path, default=QUERIES)
    run.add_argument("-v", "--verbose", action="store_true")
    for cmd in (get, run):
        cmd.add_argument("--pages", type=Path, default=PAGES)
    args = parser.parse_args()
    if args.command == "fetch":
        asyncio.run(fetch(args.url, args.name, args.pages))
    else:
        score(args.queries, args.pages, args.verbose)


if __name__ == "__main__":
    main()
