"""The dates a question names, with the time left from today, for the moderator's opening.

A local Chair called a six-month window "18 months", then "16 months" with today's
date in its task (2026-11-19 runs), and members repeated it: small models don't do
date arithmetic, so the engine does it for them.
"""

from __future__ import annotations

import calendar
import re
from datetime import date

_MONTH_NAMES = {
    1: ("january", "jan", "janvier", "janv"),
    2: ("february", "feb", "février", "fevrier", "févr", "fevr"),
    3: ("march", "mar", "mars"),
    4: ("april", "apr", "avril", "avr"),
    5: ("may", "mai"),
    6: ("june", "jun", "juin"),
    7: ("july", "jul", "juillet", "juil"),
    8: ("august", "aug", "août", "aout"),
    9: ("september", "sep", "sept", "septembre"),
    10: ("october", "oct", "octobre"),
    11: ("november", "nov", "novembre"),
    12: ("december", "dec", "décembre", "decembre", "déc"),
}
_MONTHS = {name: number for number, names in _MONTH_NAMES.items() for name in names}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY = r"(?:1er|\d{1,2})(?:st|nd|rd|th)?"
_DATE = re.compile(
    rf"\b(?:(?P<d1>{_DAY})\s+)?(?P<month>{_MONTH})\.?(?:\s+(?P<d2>{_DAY}),?)?\s+(?P<year>20\d\d)\b"
    r"|\b(?P<iso>20\d\d-\d\d-\d\d)\b"
    r"|\b[QT](?P<quarter>[1-4])\s+(?P<qyear>20\d\d)\b",
    re.IGNORECASE,
)
_DAYS_A_MONTH = 30.44
MAX_DATES = 3


def _day(text: str | None) -> int | None:
    """Return a day number from "5", "5th" or "1er", or None."""
    if not text:
        return None
    digits = re.match(r"\d+", text)
    return int(digits.group()) if digits else 1


def _span(match: re.Match[str]) -> tuple[date, date] | None:
    """Return the first and last day a matched date covers, or None if invalid."""
    try:
        if match["iso"]:
            day = date.fromisoformat(match["iso"])
            return day, day
        if match["quarter"]:
            year, first = int(match["qyear"]), 3 * int(match["quarter"]) - 2
            last = first + 2
            return date(year, first, 1), date(year, last, calendar.monthrange(year, last)[1])
        year, month = int(match["year"]), _MONTHS[match["month"].lower()]
        number = _day(match["d1"]) or _day(match["d2"])
        if number is not None:
            return date(year, month, number), date(year, month, number)
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    except ValueError:
        return None


def months(days: int) -> str:
    """Return a number of days as months, to the half month ("6.5")."""
    halves = round(days / _DAYS_A_MONTH * 2)
    return f"{halves / 2:g}"


def time_left(question: str, today: date) -> list[tuple[str, int, int]]:
    """Return the dates the question names, with the days from today to their first and last day.

    A month or a quarter spans several days ("May 2027": its 1st to its 31st); a
    past date gives negative days. At most MAX_DATES, in the question's order.
    """
    out: list[tuple[str, int, int]] = []
    for match in _DATE.finditer(question):
        span = _span(match)
        if span is None:
            continue
        text = " ".join(match.group().split())
        if any(text == t for t, _, _ in out):
            continue
        out.append((text, (span[0] - today).days, (span[1] - today).days))
    return out[:MAX_DATES]
