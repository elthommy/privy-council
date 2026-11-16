"""Page text for the document reader (M21): HTML to blocks, blocks to chunks.

Pure functions, stdlib only. Scripts and styles are dropped, headings kept, and a
table row becomes its cells joined by " | ", so a maturity table stays readable.
A chunk is about CHUNK_CHARS on paragraph borders, with its heading on top; a table
stays whole up to TABLE_CHARS (plus a short lead-in paragraph), and a longer one is
cut by rows with its first row repeated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal

CHUNK_CHARS = 1500
TABLE_CHARS = 4000
# A paragraph this short right before a table is its lead-in ("The following table
# shows principal maturities by year"): it joins the table's passage, or it outranks
# the table, whose rows hold the figures but few of the words.
LEAD_CHARS = 500
# A first line this short is a table's subtitle ("Seasonally adjusted, billions of
# dollars."), part of its label; a longer one is a paragraph.
SUBTITLE_CHARS = 80
_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe"})
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "caption"})
_BREAKS = frozenset(
    {"p", "div", "li", "br", "section", "article", "blockquote", "pre", "dd", "dt", "ul", "ol"}
)
_SPACES = re.compile(r"\s+")

Kind = Literal["heading", "text", "table"]


@dataclass(frozen=True)
class Block:
    """One piece of a page: a heading, a paragraph or a whole table."""

    kind: Kind
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage for retrieval: the nearest heading and the text under it."""

    heading: str
    text: str

    def label(self) -> str:
        """Return the heading and, if the text opens with a short line, that subtitle."""
        first = self.text.split("\n", 1)[0]
        return f"{self.heading}\n{first}" if len(first) <= SUBTITLE_CHARS else self.heading

    def render(self) -> str:
        """Return the passage as the summary reads it, heading first."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class _Table:
    """A table being parsed: finished rows, and the current row's cells."""

    rows: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    cell: list[str] | None = None


class _PageParser(HTMLParser):
    """Collect a page's blocks; nested tables are read as text inside their cell."""

    def __init__(self) -> None:
        """Start with no blocks and nothing skipped."""
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._text: list[str] = []
        self._skip = 0
        self._heading = False
        self._tables: list[_Table] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a skipped element, a heading, a table, a row or a cell."""
        if tag in _SKIP:
            self._skip += 1
        elif tag == "table":
            self._flush()
            self._tables.append(_Table())
        elif self._tables and tag in ("td", "th"):
            self._tables[-1].cell = []
        elif tag in _HEADINGS or (tag in _BREAKS and not self._tables):
            self._flush()
            self._heading = tag in _HEADINGS

    def handle_endtag(self, tag: str) -> None:
        """Close what `handle_starttag` opened."""
        if tag in _SKIP:
            self._skip = max(self._skip - 1, 0)
        elif tag == "table" and self._tables:
            self._end_table()
        elif self._tables and tag in ("td", "th"):
            table = self._tables[-1]
            if table.cell is not None:
                table.cells.append(_clean("".join(table.cell)))
                table.cell = None
        elif self._tables and tag == "tr":
            self._end_row(self._tables[-1])
        elif tag in _HEADINGS or (tag in _BREAKS and not self._tables):
            self._flush()

    def handle_data(self, data: str) -> None:
        """Add text to the open cell, or to the current paragraph."""
        if self._skip:
            return
        if self._tables and self._tables[-1].cell is not None:
            self._tables[-1].cell.append(data)
        elif not self._tables or self._heading:  # a table's caption is its heading
            self._text.append(data)

    def _end_row(self, table: _Table) -> None:
        """Finish a row: its non-empty cells joined by " | "."""
        if table.cell is not None:
            table.cells.append(_clean("".join(table.cell)))
            table.cell = None
        cells = [c for c in table.cells if c]
        if cells:
            table.rows.append(" | ".join(cells))
        table.cells = []

    def _end_table(self) -> None:
        """Finish a table: a block, or text in the outer table's open cell."""
        table = self._tables.pop()
        self._end_row(table)
        text = "\n".join(table.rows)
        if self._tables and self._tables[-1].cell is not None:
            self._tables[-1].cell.append(" " + text.replace("\n", "; "))
        elif text:
            self.blocks.append(Block("table", text))

    def _flush(self) -> None:
        """Turn the collected text into a block (a heading if inside one)."""
        text = _clean("".join(self._text))
        self._text = []
        if text:
            self.blocks.append(Block("heading" if self._heading else "text", text))
        self._heading = False

    def close(self) -> None:
        """Finish parsing, keeping any open table and the last text."""
        super().close()
        while self._tables:
            self._end_table()
        self._flush()


def _clean(text: str) -> str:
    """Fold runs of whitespace to one space."""
    return _SPACES.sub(" ", text).strip()


def html_blocks(html: str) -> list[Block]:
    """Return an HTML page's blocks in reading order."""
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return parser.blocks


def text_blocks(text: str) -> list[Block]:
    """Return a plain-text page's paragraphs (split on blank lines)."""
    return [Block("text", _clean(p)) for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunks(blocks: list[Block]) -> list[Chunk]:
    """Group blocks into passages under their nearest heading."""
    out: list[Chunk] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        """Close the passage being built."""
        if body:
            out.append(Chunk(heading, "\n".join(body)))
            body.clear()

    for block in blocks:
        if block.kind == "heading":
            flush()
            heading = block.text
        elif block.kind == "table":
            lead = body.pop() if body and len(body[-1]) <= LEAD_CHARS else ""
            flush()
            parts = _table_parts(block.text)
            out.append(Chunk(heading, f"{lead}\n{parts[0]}" if lead else parts[0]))
            out += [Chunk(heading, part) for part in parts[1:]]
        else:
            if body and sum(len(b) for b in body) + len(block.text) > CHUNK_CHARS:
                flush()
            body.append(block.text[: CHUNK_CHARS * 4])
    flush()
    return out


def _table_parts(table: str) -> list[str]:
    """Cut a long table by rows, each part starting with the first row (the header)."""
    if len(table) <= TABLE_CHARS:
        return [table]
    header, *rows = table.split("\n")
    parts: list[str] = []
    current: list[str] = [header]
    for row in rows:
        if len(current) > 1 and sum(len(r) + 1 for r in current) + len(row) > TABLE_CHARS:
            parts.append("\n".join(current))
            current = [header]
        current.append(row[:TABLE_CHARS])
    parts.append("\n".join(current))
    return parts
