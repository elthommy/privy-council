"""Checks and text helpers for dossier files (`/attach`). Pure, except reading the file."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path

from privy_council.engine import prompts
from privy_council.engine.notices import (
    NOTICE_ATTACH_EMPTY,
    NOTICE_ATTACH_MISSING,
    NOTICE_ATTACH_NOT_TEXT,
    NOTICE_ATTACH_TOO_BIG,
    NOTICE_ATTACH_TYPE,
)

# Plain-text formats only.
ALLOWED_SUFFIXES = (".txt", ".json", ".md", ".csv")
ALLOWED_TEXT = ", ".join(ALLOWED_SUFFIXES)


# CSV figures: exact sums the models can't reliably compute.
STATS_MAX_CHARS = 1500
MAX_GROUPS = 20  # a text column with more distinct values isn't a category
MAX_MONTHS = 24
# Summing these is meaningless even when every value is a number.
_NOT_SUMMED = re.compile(r"(^|[\s_-])(id|year|month|day|zip|postcode|phone)$", re.IGNORECASE)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# Accepted number shapes; anything else (or ambiguous) makes the column text.
_NUMBER_DOT = re.compile(r"^-?(\d+|\d{1,3}(,\d{3})+)(\.\d+)?$")  # 1234.5 or 1,234.5
_NUMBER_COMMA = re.compile(r"^-?(\d+|\d{1,3}([ .\u00a0]\d{3})+)(,\d+)?$")  # 1 234,5 (";" files)
_CURRENCY = "€$£ "


class DossierError(Exception):
    """A file can't be attached; the message says why."""


def attach_path(arg: str) -> Path | None:
    """Turn an `/attach` argument into a path (quotes from drag-and-drop removed)."""
    text = arg.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return Path(text).expanduser() if text else None


def read_attachable(path: Path, max_bytes: int) -> str:
    """Return the text of a file that may be attached, or raise DossierError."""
    if not path.is_file():
        raise DossierError(NOTICE_ATTACH_MISSING.format(path=path))
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise DossierError(NOTICE_ATTACH_TYPE.format(name=path.name, allowed=ALLOWED_TEXT))
    size = path.stat().st_size
    if size > max_bytes:
        raise DossierError(NOTICE_ATTACH_TOO_BIG.format(name=path.name, size=size, limit=max_bytes))
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DossierError(NOTICE_ATTACH_NOT_TEXT.format(name=path.name)) from exc
    if not text.strip():
        raise DossierError(NOTICE_ATTACH_EMPTY.format(name=path.name))
    return text


def csv_head(text: str, rows: int) -> str | None:
    """Return a CSV's header and first `rows` lines, or None when `rows` is 0."""
    if rows <= 0:
        return None
    return "\n".join(text.strip().splitlines()[: rows + 1])


def _number_parser(delimiter: str) -> Callable[[str], Decimal | None]:
    """Return a parser for one cell: comma decimals in ";" files, dot decimals otherwise."""
    pattern, thousands, decimal = (
        (_NUMBER_COMMA, " .\u00a0", ",") if delimiter == ";" else (_NUMBER_DOT, ",", ".")
    )

    def parse(cell: str) -> Decimal | None:
        """Return the cell's value, or None if it isn't an unambiguous number."""
        text = cell.strip().strip(_CURRENCY)
        if not pattern.fullmatch(text):
            return None
        for sep in thousands:
            text = text.replace(sep, "")
        try:
            return Decimal(text.replace(decimal, "."))
        except InvalidOperation:
            return None

    return parse


def _read_csv(text: str) -> tuple[list[str], list[list[str]], str]:
    """Return the header, the non-blank rows and the delimiter (sniffed: , ; or tab)."""
    try:
        delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    if not rows:
        return [], [], delimiter
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    return [h.strip() for h in rows[0]], body, delimiter


def _formatter(values: list[Decimal | None]) -> Callable[[Decimal], str]:
    """Return a column's number format: two decimals if any value has some, else none."""
    decimals = any(v is not None and v != v.to_integral_value() for v in values)
    return lambda value: f"{value:,.2f}" if decimals else f"{value:,.0f}"


def _columns(
    header: list[str], body: list[list[str]], parse: Callable[[str], Decimal | None]
) -> tuple[dict[str, list[Decimal | None]], str | None, list[str]]:
    """Classify columns: summable numbers (None for blanks), a date column, category columns."""
    numbers: dict[str, list[Decimal | None]] = {}
    date, groups = None, []
    for i, name in enumerate(header):
        cells = [row[i].strip() if i < len(row) else "" for row in body]
        filled = [c for c in cells if c]
        if not filled:
            continue
        values = [parse(c) if c else None for c in cells]
        if all(v is not None for v, c in zip(values, cells, strict=True) if c):
            if not _NOT_SUMMED.search(name):
                numbers[name] = values
        elif date is None and all(_DATE.match(c) for c in filled):
            date = name
        elif 2 <= len(set(filled)) <= MAX_GROUPS:
            groups.append(name)
    return numbers, date, groups


def _sums_by(keys: list[str], values: list[Decimal | None]) -> list[tuple[str, Decimal, int]]:
    """Sum values per key, in key order; return (key, sum, rows) triples."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    counts: dict[str, int] = defaultdict(int)
    for key, value in zip(keys, values, strict=True):
        if key and value is not None:
            totals[key] += value
            counts[key] += 1
    return [(k, totals[k], counts[k]) for k in sorted(totals)]


def csv_stats(text: str) -> str | None:
    """Return exact figures for a CSV (totals, per month, per category), or None.

    Only columns where every value is a number are summed; ambiguous number
    formats count as text rather than being guessed.
    """
    header, body, delimiter = _read_csv(text)
    if not header or not body:
        return None
    numbers, date, groups = _columns(header, body, _number_parser(delimiter))
    lines = [prompts.CSV_STATS_ROWS.format(rows=len(body), columns=", ".join(header))]
    for name, values in numbers.items():
        present = [v for v in values if v is not None]
        fmt = _formatter(values)
        lines.append(
            prompts.CSV_STATS_NUMBER.format(
                column=name,
                count=len(present),
                total=fmt(sum(present, Decimal(0))),
                low=fmt(min(present)),
                high=fmt(max(present)),
            )
        )
    lines += _breakdowns(header, body, numbers, date, groups)
    return _cap([prompts.CSV_STATS_TITLE, *lines])


def _breakdowns(
    header: list[str],
    body: list[list[str]],
    numbers: dict[str, list[Decimal | None]],
    date: str | None,
    groups: list[str],
) -> list[str]:
    """Return the date range, per-month sums and per-category sums lines."""

    def column(name: str) -> list[str]:
        """Return a column's cells."""
        i = header.index(name)
        return [row[i].strip() if i < len(row) else "" for row in body]

    lines = []
    if date is not None:
        days = sorted(d for d in column(date) if d)
        lines.append(
            prompts.CSV_STATS_DATES.format(column=date, first=days[0][:10], last=days[-1][:10])
        )
        months = [d[:7] for d in column(date)]
        for value, values in numbers.items():
            fmt = _formatter(values)
            sums = _sums_by(months, values)[:MAX_MONTHS]
            items = "; ".join(
                prompts.CSV_STATS_ITEM.format(key=k, total=fmt(t)) for k, t, _ in sums
            )
            lines.append(prompts.CSV_STATS_BY_MONTH.format(value=value, column=date, items=items))
    for group in groups:
        keys = column(group)
        for value, values in numbers.items():
            fmt = _formatter(values)
            items = "; ".join(
                prompts.CSV_STATS_GROUP_ITEM.format(key=k, total=fmt(t), count=n)
                for k, t, n in _sums_by(keys, values)
            )
            lines.append(prompts.CSV_STATS_BY_GROUP.format(value=value, column=group, items=items))
    return lines


def _cap(lines: list[str]) -> str:
    """Join lines, dropping the last ones (with a note) past STATS_MAX_CHARS."""
    kept: list[str] = []
    used = len(prompts.CSV_STATS_CUT) + 1
    for line in lines:
        if used + len(line) + 1 > STATS_MAX_CHARS:
            return "\n".join([*kept, prompts.CSV_STATS_CUT])
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)
