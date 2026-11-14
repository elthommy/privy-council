"""Members' estimate lines (M17): parsing, coherence checks and the council aggregate.

Pure functions, no I/O. In councils with `estimate: probability` every member ends a
turn with "**Estimate:** N% (low-high%)" and, once the moderator split the question
on one event, a split line before it. The engine reads the numbers from these lines,
so incoherent or silently moved numbers are caught by arithmetic, not by a model.
Both languages' labels always count: a model that drifts language still answers.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

from privy_council.engine import prompts
from privy_council.engine.dynamics import marks_update
from privy_council.models.estimate import Estimate, EstimateRecord, Split
from privy_council.models.flaw import Flaw, FlawKind

# A moved estimate under this many points is rounding or noise, not a change.
CHANGE_POINTS = 3.0
# How far a split's implied value may sit from the estimate when no range is given.
SPLIT_TOLERANCE = 3.0
# Slack on a range's ends, for rounding ("12%" stated for an implied 12.4%).
RANGE_SLACK = 0.5
_EXCERPT_CHARS = 160


def _label_words(key: str) -> str:
    """Return a regex alternation of a label in every language, colon dropped."""
    words = {labels[key].rstrip(" :") for labels in prompts.LABELS.values()}
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# "**Estimate:** 12%", "**Estimate**: 12%", "Estimation : 12 %".
_LINE = r"(?:{words})\s*\**\s*:\s*\**(?P<rest>.*)$"
_ESTIMATE = re.compile(_LINE.format(words=_label_words("estimate")), re.IGNORECASE | re.MULTILINE)
_SPLIT = re.compile(_LINE.format(words=_label_words("split")), re.IGNORECASE | re.MULTILINE)
_NUM = r"(\d{1,3}(?:[.,]\d+)?)"
_PERCENT = re.compile(_NUM + r"\s*%")
_RANGE = re.compile(_NUM + r"\s*%?\s*(?:-|\u2013|\u2014|to|\u00e0)\s*" + _NUM + r"\s*%")
# A change stated in other words than the update label, English and French.
_STATED_CHANGE = re.compile(
    r"\bfrom (?:about |roughly |around )?\d[\d.,]*\s*%? to (?:about |roughly |around )?\d"
    r"|\bde \d[\d.,]*\s*%? \u00e0 \d"
    r"|\bI (?:correct|revise|lower|raise|cut|move) (?:myself|my\b)"
    r"|\bje (?:me corrige|corrige|r\u00e9vise|baisse|rel\u00e8ve) "
    r"|\bmy (?:earlier |previous |first )?\d[\d.,]*\s*% (?:was|is|answered|meant)\b"
    r"|\bmon \d[\d.,]*\s*% (?:\u00e9tait|est|valait)\b",
    re.IGNORECASE,
)


def _number(raw: str) -> float:
    """Parse "12", "12.5" or "12,5"."""
    return float(raw.replace(",", "."))


def _last_line(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Return the last line matching a label pattern: the closing line, not a mention."""
    found = list(pattern.finditer(text))
    return found[-1] if found else None


def _range_only(rest: str) -> Estimate | None:
    """Read "7-20%" (a range with no point value) as its midpoint and range."""
    first_range, first_value = _RANGE.search(rest), _PERCENT.search(rest)
    if first_range is None or first_value is None or first_range.start() > first_value.start():
        return None
    low, high = _number(first_range.group(1)), _number(first_range.group(2))
    return Estimate(value=(low + high) / 2, low=low, high=high)


def parse_estimate(text: str) -> Estimate | None:
    """Return the turn's estimate line as numbers, or None if there is none or it is unreadable."""
    line = _last_line(_ESTIMATE, text)
    if line is None:
        return None
    rest = line.group("rest")
    try:
        whole = _range_only(rest)
        if whole is not None:
            return whole
        value = _PERCENT.search(rest)
        if value is None:
            return None
        span = _RANGE.search(rest, value.end())
        if span is None:
            return Estimate(value=_number(value.group(1)))
        low, high = _number(span.group(1)), _number(span.group(2))
        return Estimate(value=_number(value.group(1)), low=low, high=high)
    except ValueError:  # a number over 100: not a probability
        return None


def parse_split(text: str) -> Split | None:
    """Return the turn's split line (event, if yes, if no), or None."""
    line = _last_line(_SPLIT, text)
    if line is None:
        return None
    values = [_number(m.group(1)) for m in _PERCENT.finditer(line.group("rest"))]
    if len(values) < 3:
        return None
    try:
        return Split(event=values[0], if_yes=values[1], if_no=values[2])
    except ValueError:
        return None


def _excerpt(pattern: re.Pattern[str], text: str) -> str:
    """Return the labelled line, or the turn's start, as a flaw's excerpt."""
    line = _last_line(pattern, text)
    raw = line.group(0) if line else text
    return " ".join(raw.replace("*", "").split())[:_EXCERPT_CHARS]


def _flaw(kind: FlawKind, excerpt: str, note: str, member: str, round_: int) -> Flaw:
    """Build an engine flaw on an estimate line."""
    return Flaw(member=member, kind=kind, excerpt=excerpt, note=note, round=round_)


def _range_flaw(est: Estimate) -> str | None:
    """Return the note on a reversed range or a value outside its range, or None."""
    if est.low is None or est.high is None:
        return None
    low, high = prompts.percent(est.low), prompts.percent(est.high)
    if est.low > est.high:
        return prompts.ESTIMATE_NOTE_REVERSED.format(low=low, high=high)
    if not est.low - RANGE_SLACK <= est.value <= est.high + RANGE_SLACK:
        return prompts.ESTIMATE_NOTE_OUTSIDE.format(
            value=prompts.percent(est.value), low=low, high=high
        )
    return None


def _split_flaw(est: Estimate, split: Split) -> str | None:
    """Return the note when the split's implied value disagrees with the estimate, or None."""
    implied = split.implied
    if est.low is not None and est.high is not None and est.low <= est.high:
        fits = est.low - RANGE_SLACK <= implied <= est.high + RANGE_SLACK
    else:
        fits = abs(implied - est.value) <= SPLIT_TOLERANCE
    if fits:
        return None
    return prompts.ESTIMATE_NOTE_SPLIT.format(
        event=prompts.percent(split.event),
        if_yes=prompts.percent(split.if_yes),
        not_event=prompts.percent(100 - split.event),
        if_no=prompts.percent(split.if_no),
        implied=prompts.percent(implied),
        value=prompts.percent(est.value),
    )


def _mentions(text: str, value: float) -> bool:
    """Whether the prose (estimate and split lines left out) names `value` as a percent."""
    number = re.escape(prompts.percent(value).rstrip("%")).replace(r"\.", "[.,]")
    pattern = rf"(?<![\d.,]){number}(?:[.,]0+)?\s*%"
    return re.search(pattern, blank_lines(text)) is not None


def _explained(text: str, previous: float) -> bool:
    """Whether the turn says its number changed, in the marker's words or its own.

    Members explain a move without the marker: "I correct myself", "my 45% was the
    ban's probability", "from 13% to 16%", or a worked chain that sets the old
    number aside (all four seen 2026-11-14, every one flagged). Naming the old
    number shows the member addressed it.
    """
    return marks_update(text) or bool(_STATED_CHANGE.search(text)) or _mentions(text, previous)


def _unmarked(
    text: str, est: Estimate, previous: EstimateRecord | None, reframed: bool
) -> str | None:
    """Return the note when the value moved with no stated reason, or None.

    `reframed`: the user spoke since the member's last estimate, so the question
    itself may have changed (an answer to the moderator often redefines it).
    """
    if previous is None or reframed:
        return None
    if abs(est.value - previous.estimate.value) < CHANGE_POINTS:
        return None
    if _explained(text, previous.estimate.value):
        return None
    return prompts.ESTIMATE_NOTE_UNMARKED.format(
        before=prompts.percent(previous.estimate.value),
        after=prompts.percent(est.value),
        update=prompts.LABELS["en"]["update"].rstrip(" :"),
    )


def read_turn(
    text: str,
    *,
    member: str,
    round_: int,
    previous: EstimateRecord | None,
    split_set: bool,
    reframed: bool = False,
) -> tuple[EstimateRecord | None, list[Flaw]]:
    """Parse a member turn's estimate lines and check them.

    Returns the record to keep (None when the turn has no readable estimate) and the
    flaws: a missing line, a range or split that doesn't fit the estimate, or a value
    moved by CHANGE_POINTS or more since `previous` with no stated reason (not
    checked when `reframed`: the user spoke since that estimate).
    """
    est = parse_estimate(text)
    if est is None:
        excerpt = _excerpt(_ESTIMATE, text)
        note = prompts.ESTIMATE_NOTE_MISSING
        return None, [_flaw(FlawKind.MISSING_ESTIMATE, excerpt, note, member, round_)]
    split = parse_split(text) if split_set else None
    flaws: list[Flaw] = []
    est_line, split_line = _excerpt(_ESTIMATE, text), _excerpt(_SPLIT, text)
    if split_set and split is None:
        note = prompts.ESTIMATE_NOTE_NO_SPLIT
        flaws.append(_flaw(FlawKind.MISSING_ESTIMATE, est_line, note, member, round_))
    checks = [
        (_range_flaw(est), FlawKind.ESTIMATE_INCOHERENT, est_line),
        (split and _split_flaw(est, split), FlawKind.ESTIMATE_INCOHERENT, split_line),
        (_unmarked(text, est, previous, reframed), FlawKind.UNMARKED_CHANGE, est_line),
    ]
    flaws += [_flaw(kind, line, note, member, round_) for note, kind, line in checks if note]
    return EstimateRecord(round=round_, estimate=est, split=split), flaws


def blank_lines(text: str) -> str:
    """Return `text` with its estimate and split lines blanked, offsets kept.

    Their numbers are the member's own judgment, checked here by arithmetic: the
    figure check would flag a range like "12-32%" as unsourced (2026-11-14 run).
    """
    chars = list(text)
    for pattern in (_ESTIMATE, _SPLIT):
        for match in pattern.finditer(text):
            chars[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def aggregate(values: Sequence[float]) -> tuple[float, float, float] | None:
    """Return the median, lowest and highest of the members' estimates, or None if none."""
    if not values:
        return None
    return statistics.median(values), min(values), max(values)
