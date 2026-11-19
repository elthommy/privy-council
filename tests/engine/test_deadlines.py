"""The dates a question names, counted from today for the moderator's opening."""

from __future__ import annotations

from datetime import date

import pytest

from privy_council.engine import prompts
from privy_council.engine.deadlines import MAX_DATES, months, time_left

TODAY = date(2026, 11, 19)


@pytest.mark.parametrize(
    ("question", "found"),
    [
        ("Will the rate in the release for May 2027 be above 5%?", [("May 2027", 163, 193)]),
        ("Will it happen before 30 June 2027?", [("30 June 2027", 223, 223)]),
        ("by June 30th, 2027", [("June 30th, 2027", 223, 223)]),
        ("in Q2 2027", [("Q2 2027", 133, 223)]),
        ("avant le 1er mars 2027 ?", [("1er mars 2027", 102, 102)]),
        ("due 2027-01-04", [("2027-01-04", 46, 46)]),
        ("since March 2026", [("March 2026", -263, -233)]),
        ("May I ask about 2027?", []),
        ("on 31 February 2027", []),
    ],
)
def test_time_left(question: str, found: list[tuple[str, int, int]]) -> None:
    assert time_left(question, TODAY) == found


def test_time_left_keeps_the_first_dates_once() -> None:
    question = "May 2027, May 2027, June 2027, July 2027, August 2027"
    assert [t for t, _, _ in time_left(question, TODAY)] == ["May 2027", "June 2027", "July 2027"]
    assert len(time_left(question, TODAY)) == MAX_DATES


def test_months_to_the_half() -> None:
    assert [months(d) for d in (0, 15, 183, 223)] == ["0", "0.5", "6", "7.5"]


def test_rendered_dates() -> None:
    text = prompts.render_dates(time_left("by 30 June 2027 or in Q2 2027, after May 2025", TODAY))
    assert "- 30 June 2027: 223 days, about 7.5 months from today" in text
    assert "- Q2 2027: 133-223 days, about 4.5-7.5 months from today" in text
    assert "- May 2025: already past" in text
    assert prompts.render_dates([]) == ""
