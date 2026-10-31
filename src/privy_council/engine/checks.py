"""Deterministic checks on member turns (M11). Pure functions, no I/O, no model calls.

Every local model makes the same kinds of slips: figures that come from nowhere,
"(estimate, from memory)" on a figure it was given, its own "Nick:" prefix,
@names of people who aren't in the channel. The engine flags them (flag
only: the text is never changed) so the moderator can press on the ones
that matter.

Figure provenance compares the numbers with units in a turn against every number
in the member's prompt, which is exactly what it could have read. A figure counts
as sourced if it matches one of them up to rounding, or is a sum, difference,
product or ratio of two numbers of the same turn (the member's own arithmetic,
which the moderator checks). Anything else is unsourced, whatever its label.
Units are the usual ones plus the topic's own, read from the facts ("25,600
Astrite", "160 pulls"), and a conversion at a rate the facts state counts too.
A number found in the sources only with another unit is the wrong unit: "60
Astrite" when the user wrote "60 pulls" (2026-10-26 gacha run).

Arithmetic stays narrow on purpose: a sum or difference of two distinct figures with
the same unit, a percentage or multiple of two such figures, a cost spread over a
period of the turn ("$175 over 3 years: $58/year"), a percentage of the turn applied
to an amount ("15 % off $108: $92"), or a figure divided or multiplied by a number
the turn writes as an operation ("€750 / 3"). A total of two or three sourced
figures of the turn counts too, a bare "40 k" joining a money total ("25 k, 40 k
and 15 k: €80 k"): invented numbers rarely add up from quoted facts. Wider rules
(any pair of numbers) excuse almost every small number.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from itertools import combinations, permutations

from privy_council.models.flaw import Flaw, FlawKind

# A number: 1500, 1,500, 1 500 (also with the French thin spaces), 1.5, 1,5.
_NUM = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)|\d+)(?:[.,]\d+)*"
_MULT = r"(?:k|K|M|bn|tn|thousand|million|billion|trillion|milliards?|millions?)"
_CURRENCY = r"[€$£]"
# Units that make a number a figure worth checking (M11).
_UNIT = (
    r"(?:%|€|\$|£|\u00d7"
    r"|(?:EUR|USD|euros?|dollars?|GB|Go|TB|To|MB|Mo"
    r"|years?|yrs?|ans|années?|months?|mois|km|miles?|x)(?!\w))"
)
_FIGURE = re.compile(
    rf"(?<![\w.,/-])(?P<cur>{_CURRENCY}\s?)?(?P<a>{_NUM})(?:\s?(?P<am>{_MULT})(?!\w))?"
    rf"(?:\s?(?:\u2013|\u2014|-|to|à)\s?{_CURRENCY}?\s?(?P<b>{_NUM})(?:\s?(?P<bm>{_MULT})(?!\w))?)?"
    rf"(?:\s?(?P<unit>{_UNIT}))?"
)
_ANY_NUMBER = re.compile(rf"(?<![\w.,])(?P<n>{_NUM})(?:\s?(?P<m>{_MULT})(?!\w))?")
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9}
_MULTIPLIERS |= {"billion": 1e9, "milliard": 1e9, "tn": 1e12, "trillion": 1e12}
# The members' estimate label (prompts.MEMBER_PROTOCOL), also translated; it may
# follow a few words later ("€60 a year (estimate, from memory)").
_LABEL = re.compile(
    r"[^()\n.;]{0,25}\((?:estimate|estimation|estimé|est\.)[^)]{0,40}\)", re.IGNORECASE
)
_TIMES = "\u00d7"  # the multiplication sign, as in "3\u00d7"
# A number written as the right side of an operation: "/ 3", "* 12", "divided by 455".
_OPERAND = re.compile(
    r"(?:[/\u00f7*\u00d7]|\b(?:divided by|times|divisés? par|fois)\b)"
    rf"\s?(?P<n>{_NUM})(?![\d.,]*\s?%)",
    re.IGNORECASE,
)
# An operation written out between two numbers, whatever their units: "(80 - 15)",
# "35 x 160". A hyphen counts only between spaces: "40-80" is a range.
_EXPRESSION = re.compile(
    rf"(?<![\w.,])(?P<a>{_NUM})\s?(?:[^\W\d_]+\s?)?"
    rf"(?P<op>[+\u2212\u00d7*/\u00f7]|\s[-\u2013]\s)\s?(?P<b>{_NUM})(?![\d.,])"
)
# A moderator's note that says the quote's arithmetic holds ("which is correct", "no
# flaw found"), and one that also disputes something ("is wrong", "not 20"); 6 and 2
# of the 22 arithmetic flags saved by 2026-10-27.
_CONFIRMS = re.compile(
    r"\b(?:is|are|was|were) (?:\w+ )?(?:correct|fine|right|accurate)\b"
    r"|\bno (?:flaw|error)s?\b|\bnot (?:actually |really )?an? (?:\w+ )?error\b"
    r"|\b(?:est|sont) (?:correcte?s?|justes?|exacte?s?)\b|\baucune erreur\b|\bpas une erreur\b",
    re.IGNORECASE,
)
_DISPUTES = re.compile(
    r"\b(?:wrong|incorrect|mistaken|faux|fausse|erron\w*)\b|\bnot (?:about |roughly |~)?[€$£]?\d",
    re.IGNORECASE,
)
# Units of the same kind, for the members' own sums ("€330 - €170 = €160").
_FAMILIES = {
    "money": ("€", "$", "£", "eur", "usd", "euro", "dollar"),
    "storage": ("gb", "go", "tb", "to", "mb", "mo"),
    "time": ("year", "yr", "an", "année", "month", "moi"),
    "distance": ("km", "mile"),
}
_CURRENCY_WORDS = {"eur": "€", "euro": "€", "usd": "$", "dollar": "$"}
# Lines whose figures are the member's own judgment, not claims: "Confidence: 70%".
_OWN_LINE = re.compile(r"\b(?:confidence|confiance)\b[^\n]*", re.IGNORECASE)
# Percentages that are figures of speech ("100% sure").
_TRIVIAL_PERCENT = {0.0, 100.0}
# Rounding: a figure may round its source, by at most this share of its value.
_ROUNDING_SHARE = 0.1
_MENTION = re.compile(r"(?<![\w@.])@([^\W_][\w-]*)")
# @-addresses that aren't names.
_GROUP_MENTIONS = frozenset({"all", "everyone", "here", "tous", "toutes", "chacun"})
_NICK_LITERAL = "nick"
_EXCERPT_CHARS = 60  # around a figure, each side, within its sentence
_SENTENCE_END = re.compile(r"[!?\n]|\.(?!\d)")
# A topic's own units ("25,600 Astrite", "160 pulls"): the word after a number in the
# facts. Words that only join or compare are no unit ("2026 alongside", "80 more").
_WORD_AFTER = re.compile(rf"(?<![\w.,])(?:{_NUM})(?:\s?{_MULT}(?!\w))?\s(?P<w>[^\W\d_]{{3,}})")
# In a turn the unit may also follow a hyphen: "a 50-pull pity" (also U+2010/U+2011).
_NEXT_WORD = re.compile(r"[\s\-\u2010\u2011]?(?P<w>[^\W\d_]{3,})")
_NOT_UNIT_WORDS = (
    "and the for with per from than are was were has have had alongside between into "
    "over under after before about around each while which that this these those then "
    "but not more less most least other times out off its their his her you your our "
    "les des pour avec par sur sous dans entre plus moins que qui une aux"
)
_NOT_UNITS = frozenset(_NOT_UNIT_WORDS.split())
# Small numbers written as words, read before a named unit only ("one Astrite", not
# "one-time" or "two boards"): "one" and "two" are everywhere else in prose.
_SPELLED_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "dix": 10, "douze": 12,
}  # fmt: skip
_SPACED_WORD = re.compile(r"\s(?P<w>[^\W\d_]{3,})")
_YEARS = re.compile(r"(?:19|20)\d\d")
_SPELLED = re.compile(rf"\b(?P<n>{'|'.join(_SPELLED_VALUES)})\b", re.IGNORECASE)
# Units whose numbers can't be in the wrong unit: shares, multiples, bare multipliers.
_UNITLESS_FAMILIES = frozenset({"", "%", "x", _TIMES})


@dataclass(frozen=True)
class Figure:
    """A number with a unit in a turn (a range has two values)."""

    text: str
    values: tuple[float, ...]
    precisions: tuple[float, ...]  # the place of each value's last significant digit
    start: int
    end: int
    labelled: bool  # followed by "(estimate, from memory)"
    family: str  # the unit's kind: money, storage, time, %, x or the unit itself
    unit: str = ""  # the unit itself, currencies folded ("EUR" is "€")
    spelled: bool = False  # written as a word ("one Astrite")


def parse_number(raw: str) -> tuple[float, float]:
    """Return a number's value and precision (1,500 / 1 500 / 1.5 / 1,5 all read).

    A lone comma before exactly three digits is a thousands separator (English
    first); otherwise it is a decimal comma.
    """
    text = re.sub(r"[ \u00a0\u202f]", "", raw)
    seps = re.findall(r"[.,]", text)
    if len(seps) > 1:
        decimal = seps[-1] if len(set(seps)) > 1 else None
    elif seps == [","]:
        decimal = None if re.search(r",\d{3}$", text) else ","
    else:
        decimal = "." if seps else None
    if decimal is None:
        digits = re.sub(r"[.,]", "", text)
        stripped = digits.rstrip("0")
        precision = 10.0 ** (len(digits) - len(stripped)) if stripped else 1.0
        return float(digits), precision
    whole, _, frac = text.rpartition(decimal)
    whole = re.sub(r"[.,]", "", whole)
    return float(f"{whole or '0'}.{frac}"), 10.0 ** -len(frac)


def _family(currency: str | None, unit: str | None) -> str:
    """Return the kind of a figure's unit ("money" for €, EUR, $ ...)."""
    key = (currency or unit or "").strip().lower().rstrip("s")
    for family, units in _FAMILIES.items():
        if key in units:
            return family
    return key


def _unit(currency: str | None, unit: str | None) -> str:
    """Return a figure's unit, with the currency words folded into their symbols."""
    key = (currency or unit or "").strip().lower().rstrip("s")
    return _CURRENCY_WORDS.get(key, key)


def _model_number(m: re.Match[str]) -> bool:
    """Whether a "k" figure is a product name ("10700K"): nobody writes 10700 thousand."""
    return not m["cur"] and not m["unit"] and parse_number(m["a"])[0] >= 1000


def _multiplier(word: str | None) -> float:
    """Return the factor of a k / M / billion suffix (1 without one)."""
    if not word:
        return 1.0
    key = word.lower().rstrip("s")
    return _MULTIPLIERS.get(key, _MULTIPLIERS.get(key[:1], 1.0))


def topic_units(text: str) -> frozenset[str]:
    """Return the units a text puts after its numbers beyond the usual ones ("astrite")."""
    words = {_fold(m["w"]) for m in _WORD_AFTER.finditer(text)}
    usual = {w for w in words if re.fullmatch(_UNIT, w) or re.fullmatch(_MULT, w)}
    return frozenset(words - usual - _NOT_UNITS)


def _fold(word: str) -> str:
    """Return a unit word lowercased without its plural s ("Pulls" is "pull")."""
    return word.lower().rstrip("s")


def turn_units(text: str, given: str) -> frozenset[str]:
    """Return the units a turn puts after its numbers that the facts write as names.

    "40-80 Astrite" makes Astrite a unit even when the facts never give an amount of
    it. Only names (capitalized, never in lowercase): a game's currency, not "average".
    """
    words = set(re.findall(r"[^\W\d_]{3,}", given))
    return frozenset(
        u
        for u in topic_units(text)
        if not words & {u, f"{u}s"} and words & {u.capitalize(), f"{u.capitalize()}s"}
    )


def _extra_unit(text: str, end: int, extra: Collection[str]) -> tuple[str, int] | None:
    """Return the topic unit right after a number, and where it ends, if there is one."""
    m = _NEXT_WORD.match(text, end)
    return (_fold(m["w"]), m.end()) if m and _fold(m["w"]) in extra else None


def figures(text: str, extra: Collection[str] = (), named: Collection[str] = ()) -> list[Figure]:
    """Return the figures (numbers with a currency, unit or multiplier) in a turn.

    `extra`: the topic's own units (`topic_units` of the facts), counted as units too.
    `named`: units written as names (`turn_units`), also read after a number in words.
    """
    blanked = _OWN_LINE.sub(lambda m: " " * len(m.group()), text)
    out: list[Figure] = []
    for m in _FIGURE.finditer(blanked):
        word = None if m["cur"] or m["unit"] else _extra_unit(blanked, m.end(), extra)
        if not (m["cur"] or m["unit"] or m["am"] or m["bm"] or word):
            continue
        if word and not (m["am"] or m["b"]) and _YEARS.fullmatch(m["a"]):
            continue  # "a 2027 build"
        a, pa = parse_number(m["a"])
        values, precisions = [a], [pa]
        if m["b"]:
            b, pb = parse_number(m["b"])
            values.append(b)
            precisions.append(pb)
        # "€130-200k": the second end's multiplier applies to both.
        factors = [_multiplier(m["am"] or m["bm"]), _multiplier(m["bm"] or m["am"])]
        values = [v * f for v, f in zip(values, factors, strict=False)]
        precisions = [p * f for p, f in zip(precisions, factors, strict=False)]
        if not any(values) or (m["unit"] == "%" and set(values) <= _TRIVIAL_PERCENT):
            continue  # "€0", "100% sure": nothing to source
        if _model_number(m) and not word:
            continue
        unit, end = word if word else (m["unit"], m.end())
        out.append(
            Figure(
                text=blanked[m.start() : end].strip(),
                values=tuple(values),
                precisions=tuple(precisions),
                start=m.start(),
                end=end,
                labelled=False,
                family=_family(m["cur"], unit),
                unit=_unit(m["cur"], unit),
            )
        )
    out += _spelled_figures(blanked, named)
    return _with_labels(text, sorted(out, key=lambda f: f.start))


def _spelled_figures(text: str, named: Collection[str]) -> list[Figure]:
    """Return the small numbers written as words before a named unit ("one Astrite")."""
    out: list[Figure] = []
    for m in _SPELLED.finditer(text):
        word = _SPACED_WORD.match(text, m.end())
        if word is None or _fold(word["w"]) not in named:
            continue
        unit, end = _fold(word["w"]), word.end()
        value = float(_SPELLED_VALUES[m["n"].lower()])
        out.append(
            Figure(
                text=text[m.start() : end],
                values=(value,),
                precisions=(1.0,),
                start=m.start(),
                end=end,
                labelled=False,
                family=unit,
                unit=unit,
                spelled=True,
            )
        )
    return out


def _with_labels(text: str, figs: list[Figure]) -> list[Figure]:
    """Mark the figures an estimate label follows (the nearest one before the label).

    A size names the product, not the estimate: in "€82-120 for a 1 TB NVMe (estimate,
    from memory)" the label belongs to the price before it, in the same sentence.
    """
    labelled: set[int] = set()
    for i, fig in enumerate(figs):
        label = _LABEL.match(text, fig.end)
        nearest = i + 1 == len(figs) or (label is not None and label.end() <= figs[i + 1].start)
        if label is None or not nearest:
            continue
        prev = figs[i - 1] if i else None
        if (
            fig.family == "storage"
            and prev is not None
            and prev.family != "storage"
            and not _SENTENCE_END.search(text, prev.end, fig.start)
        ):
            i -= 1
        labelled.add(i)
    return [replace(fig, labelled=i in labelled) for i, fig in enumerate(figs)]


def numbers(text: str) -> list[float]:
    """Return every number in a text, also scaled by its k / M suffix if it has one.

    In a range the second end's suffix scales the first too ("$860-940M" gives 860M),
    as in `figures`: else a member quoting the range was flagged (2026-10-31 run).
    """
    out: list[float] = []
    for m in _ANY_NUMBER.finditer(text):
        value, _ = parse_number(m["n"])
        out.append(value)
        if m["m"]:
            out.append(value * _multiplier(m["m"]))
    for m in _FIGURE.finditer(text):
        if m["b"] and m["bm"] and not m["am"]:
            out.append(parse_number(m["a"])[0] * _multiplier(m["bm"]))
    return out


def _matches(value: float, precision: float, known: Iterable[float]) -> bool:
    """Whether a value equals a known number, up to its own rounding or truncation.

    Truncation: "€858" for "€858.90" drops the cents rather than rounding them.
    """
    tolerance = max(min(precision / 2, abs(value) * _ROUNDING_SHARE), abs(value) * 1e-6)
    cut = min(precision, abs(value) * _ROUNDING_SHARE)
    return any(abs(k - value) <= tolerance + 1e-9 or 0 <= abs(k) - abs(value) < cut for k in known)


def _in_range(value: float, precision: float, ranges: Iterable[tuple[float, float]]) -> bool:
    """Whether a value, up to its own rounding, falls in one of the ranges."""
    return any(lo - precision / 2 <= value <= hi + precision / 2 for lo, hi in ranges)


def _derived(figs: list[Figure], text: str, skip: Figure | None = None) -> dict[str, set[float]]:
    """Return what a turn's own arithmetic can yield, per unit family.

    `skip`: the figure being checked, left out, operand included: "8,000 Astrite"
    isn't "8,000 + 200 Astrite" up to rounding.

    Sums and differences of two distinct figures of one family keep its unit; their
    ratio gives a percentage or a multiple ("3x"). Money spread over one of the turn's
    periods, and any figure through one of its written operations, keep their unit;
    so does a sum of two figures put through them ("(€450 + €550) / 3").
    """
    out: dict[str, set[float]] = {}
    families = _by_family(f for f in figs if f is not skip)
    totals: dict[str, set[float]] = {}
    for family, values in families.items():
        totals[family] = set(values)
        for a, b in permutations(sorted(values), 2):
            out.setdefault(family, set()).update({a + b, a - b})
            totals[family].add(a + b)
            if b:
                out.setdefault("%", set()).update({a / b * 100, (a - b) / b * 100})
                out.setdefault(_TIMES, set()).add(a / b)
    periods = {t for t in families.get("time", set()) if t}
    shares = {p / 100 for p in families.get("%", set()) if 0 < p < 100}
    money = totals.get("money", set())
    out.setdefault("money", set()).update(m / t for m in money for t in periods)
    # A written percentage applied to a price: "a 15 % reduction (of $108) gives $92".
    out["money"].update(m * k for m in money for s in shares for k in (s, 1 - s, 1 + s))
    operands = {
        v
        for m in _OPERAND.finditer(text)
        if not (skip and m.start("n") < skip.end and skip.start < m.end("n"))
        if (v := parse_number(m["n"])[0])
    }
    for family, values in totals.items():
        out.setdefault(family, set()).update(v / n for v in values for n in operands)
        out[family].update(v * n for v in values for n in operands)
    out["x"] = out.get(_TIMES, set())
    return out


def _written(text: str, known: Collection[float]) -> set[float]:
    """Return the results of the operations a turn writes out ("(80 - 15)" gives 65).

    Any unit: "80 - 15" may take 15 pity from 80 pulls. Each operand must be a
    number of the sources or an earlier result ("65 - 30", then "35 x 160"): a
    wrong price ("120 pulls x 200" for 160) keeps its total unsourced.
    """
    out: set[float] = set()
    for m in _EXPRESSION.finditer(text):
        (a, pa), (b, pb) = parse_number(m["a"]), parse_number(m["b"])
        pool = [*known, *out]
        if not (_matches(a, min(pa, 1.0), pool) and _matches(b, min(pb, 1.0), pool)):
            continue
        op = m["op"].strip()
        if op == "+":
            out.add(a + b)
        elif op in ("\u2212", "-", "\u2013"):
            out.add(a - b)
        elif op in ("\u00d7", "*"):
            out.add(a * b)
        elif b:
            out.add(a / b)
    return out


def _priced(text: str, prices: dict[str, set[float]]) -> dict[str, set[float]]:
    """Return the numbers a turn writes times the sources' prices, per price family.

    "60 saved pulls (≈9,600 Astrite)" is 60 x 160 when the sources give "160
    Astrite": a price the facts state without a number before its unit ("a pull
    costs 160 Astrite"), so `_rates` can't read it. Any number counts ("saved"
    keeps "60 pulls" from being a figure); the product must then be exact. Topic
    units only: euros and months are too common for "3 x €10" not to excuse "€30".
    """
    counts = set(numbers(text))
    plain = _UNITLESS_FAMILIES | set(_FAMILIES)
    return {
        family: {c * p for c in counts for p in values}
        for family, values in prices.items()
        if family not in plain
    }


def _sourced_totals(figs: list[Figure], known: Collection[float]) -> dict[str, set[float]]:
    """Return the sums of two or three of a turn's sourced figures, per unit family.

    A bare figure (no unit, "40 k") joins every family's sums: people's amounts are
    often written without the currency their total gets.
    """
    sourced = [
        (fig.family, v)
        for fig in figs
        for v, p in zip(fig.values, fig.precisions, strict=True)
        if _matches(v, p, known)
    ]
    bare = {v for family, v in sourced if not family}
    out: dict[str, set[float]] = {}
    for family in {fig.family for fig in figs if fig.family}:
        values = sorted({v for f, v in sourced if f == family} | bare)
        sums = {sum(c) for n in (2, 3) for c in combinations(values, n)}
        out[family] = sums
    return out


def _chained(figs: list[Figure]) -> list[tuple[float, float]]:
    """Return the ranges two of a turn's percentages multiply into ("12% x 12% = 1.5%").

    A chained probability. Each factor keeps its own rounding ("12%" is 11.5-12.5%),
    so the product is a range. Each written percentage counts once: a lone "12%"
    doesn't explain "1.44%".
    """
    written = [
        (v, p)
        for fig in figs
        if fig.family == "%"
        for v, p in zip(fig.values, fig.precisions, strict=True)
        if 0 < v < 100
    ]
    return [
        ((a - pa / 2) * (b - pb / 2) / 100, (a + pa / 2) * (b + pb / 2) / 100)
        for (a, pa), (b, pb) in combinations(written, 2)
    ]


def _rates(given: str, extra: Collection[str]) -> dict[str, dict[str, set[float]]]:
    """Return the rates the facts state, as {from family: {to family: factors}}.

    Two figures of different units in one sentence of the facts are one price: "160
    pulls, or 25,600 Astrite" is 160 Astrite a pull, so "120 pulls (19,200 Astrite)"
    is the member's own arithmetic, and "200 Astrite a pull" is not.
    """
    out: dict[str, dict[str, set[float]]] = {}
    figs = figures(given, extra)
    for a, b in permutations(figs, 2):
        if a.family == b.family or "%" in (a.family, b.family) or not (a.family and b.family):
            continue
        if _SENTENCE_END.search(given, min(a.end, b.end), max(a.start, b.start)):
            continue
        for va, vb in zip(a.values, b.values, strict=False):
            if va:
                out.setdefault(a.family, {}).setdefault(b.family, set()).add(vb / va)
    return out


def _converted(
    figs: list[Figure], rates: dict[str, dict[str, set[float]]]
) -> dict[str, set[float]]:
    """Return the turn's figures converted at the facts' rates, per target family."""
    out: dict[str, set[float]] = {}
    for fig in figs:
        for family, factors in rates.get(fig.family, {}).items():
            out.setdefault(family, set()).update(v * f for v in fig.values for f in factors)
    return out


def _excerpt(text: str, start: int, end: int) -> str:
    """Return the passage around a span, within its sentence, whitespace folded."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text, 0, start)]
    left = ends[-1] if ends else 0
    after = _SENTENCE_END.search(text, end)
    right = after.end() if after else len(text)
    lo = max(left, start - _EXCERPT_CHARS)
    hi = min(right, end + _EXCERPT_CHARS)
    snippet = " ".join(text[lo:hi].split()).strip(" *_-")
    return ("…" if lo > left else "") + snippet + ("…" if hi < right else "")


def _by_family(figs: Iterable[Figure]) -> dict[str, set[float]]:
    """Return the figures' values grouped by unit family."""
    out: dict[str, set[float]] = {}
    for fig in figs:
        out.setdefault(fig.family, set()).update(fig.values)
    return out


def _by_unit(figs: Iterable[Figure]) -> dict[str, set[float]]:
    """Return the figures' values grouped by unit (€ apart from $)."""
    out: dict[str, set[float]] = {}
    for fig in figs:
        out.setdefault(fig.unit, set()).update(fig.values)
    return out


def figures_found(excerpt: str, text: str) -> bool:
    """Whether an excerpt has figures and each is in the text, with the same unit."""
    extra = topic_units(text)
    figs = figures(excerpt, extra)
    known = _by_unit(figures(text, extra))
    return bool(figs) and all(
        _matches(value, precision, known.get(fig.unit, ()))
        for fig in figs
        for value, precision in zip(fig.values, fig.precisions, strict=True)
    )


def _occurrences(sources: str, extra: Collection[str]) -> tuple[dict[str, set[float]], set[float]]:
    """Return the sources' numbers per unit family, and those written without a unit.

    Confidence lines don't count: "Confidence: 60" doesn't make "60 Astrite" a number
    the member could have meant bare.
    """
    blanked = _OWN_LINE.sub(lambda m: " " * len(m.group()), sources)
    figs = figures(blanked, extra)
    bare = {
        parse_number(m["n"])[0]
        for m in _ANY_NUMBER.finditer(blanked)
        if not any(f.start <= m.start() < f.end for f in figs)
    }
    return _by_family(figs), bare


def _wrong_unit(
    fig: Figure, occurrences: tuple[dict[str, set[float]], set[float]], own: set[float]
) -> bool:
    """Whether a figure's number is in the sources only with another unit.

    A number the sources also write bare, or the turn's own arithmetic can give, may
    be the member's: only a number every source gives in another unit is flagged.
    Units digits, not rounding: up to rounding, "80" is anything from 75 to 85.
    """
    if fig.family in _UNITLESS_FAMILIES:
        return False
    by_family, bare = occurrences
    others = {v for family, vs in by_family.items() if family != fig.family for v in vs}
    return any(
        not _matches(v, p, by_family.get(fig.family, ()))
        and v not in bare
        and not _matches(v, min(p, 1.0), own)
        and v in others
        for v, p in zip(fig.values, fig.precisions, strict=True)
    )


def _sourced(fig: Figure, known: Collection[float], by_family: dict[str, set[float]]) -> bool:
    """Whether each of a figure's values is in the sources.

    A number written as a word must be there with its unit: "one" is in every prompt.
    """
    pool = by_family.get(fig.family, set()) if fig.spelled else known
    return all(_matches(v, p, pool) for v, p in zip(fig.values, fig.precisions, strict=True))


def figure_flaws(text: str, sources: str, given: str, member: str, round_: int = 0) -> list[Flaw]:
    """Flag unsourced figures, figures in the wrong unit, and misplaced estimate labels.

    Unsourced: a value found nowhere in `sources`, even as a bare number.
    Wrong unit: a value the sources give only with another unit (`_wrong_unit`).
    Mislabelled: labelled, though `given` (the fresh facts and the dossier) has the
    same value in the same unit ("$233" for "$233", not just "233" or "€233"): a
    small number is almost always somewhere in a long prompt, and an estimate in
    another currency is the member's own conversion. The facts' own units count too
    ("Astrite", "pulls"), and so do conversions at the rates they state.
    """
    known = numbers(sources)
    named = turn_units(text, given)
    extra = topic_units(given) | named
    quoted = _by_unit(figures(given, extra))
    occurrences = _occurrences(sources, extra)
    figs = figures(text, extra, named)
    totals = _sourced_totals(figs, known)
    converted = _converted(figs, _rates(given, extra))
    priced = _priced(text, occurrences[0])
    written = _written(text, known)
    chained = _chained(figs)
    out: list[Flaw] = []
    for fig in figs:
        pairs = list(zip(fig.values, fig.precisions, strict=True))
        sourced = _sourced(fig, known, occurrences[0])
        own = (
            _derived(figs, text, fig).get(fig.family, set())
            | totals.get(fig.family, set())
            | converted.get(fig.family, set())
            | written
        )
        exact = priced.get(fig.family, set())
        ranges = chained if fig.family == "%" else []
        if (
            not sourced
            and not fig.spelled
            and all(
                _matches(v, p, known)
                or _matches(v, p, own)
                or _matches(v, min(p, 1.0), exact)
                or _in_range(v, p, ranges)
                for v, p in pairs
            )
        ):
            continue  # the member's own arithmetic: the moderator judges it
        kind = _kind(fig, sourced, quoted, occurrences, own)
        if kind is None:
            continue
        excerpt = _excerpt(text, fig.start, fig.end)
        flaw = Flaw(member=member, kind=kind, excerpt=excerpt, note=fig.text, round=round_)
        if flaw not in out:
            out.append(flaw)
    return out


def figures_cleared(quote: str, text: str, sources: str, given: str) -> bool:
    """Whether a quote's figures are in the turn and none is flagged by `figure_flaws`.

    For the moderator's "unsourced" flags: a figure the member's own prompt gives,
    or its own arithmetic (60 x 160 = 9,600), is not unsourced (2026-10-27 Discord
    debate). False when the quote has no figure or misquotes the turn.
    """
    if not figures_found(quote, text):
        return False
    flagged = figure_flaws(text, sources, given, member="")
    return not any(f.note in quote for f in flagged)


def note_confirms(note: str) -> bool:
    """Whether a moderator's note says the quote's arithmetic holds."""
    return bool(_CONFIRMS.search(note))


def note_disputes(note: str) -> bool:
    """Whether a moderator's note also calls something wrong ("is wrong", "not 20")."""
    return bool(_DISPUTES.search(note))


def _kind(
    fig: Figure,
    sourced: bool,
    quoted: dict[str, set[float]],
    occurrences: tuple[dict[str, set[float]], set[float]],
    own: set[float],
) -> FlawKind | None:
    """Return what is wrong with a figure past the arithmetic check, or None."""
    if not sourced:
        return FlawKind.UNSOURCED
    if _wrong_unit(fig, occurrences, own):
        return FlawKind.WRONG_UNIT
    same = quoted.get(fig.unit, set())
    pairs = zip(fig.values, fig.precisions, strict=True)
    if fig.labelled and all(_matches(v, p, same) for v, p in pairs):
        return FlawKind.MISLABELLED
    return None


def format_flaws(
    text: str, member: str, nick: str, known: Collection[str], round_: int = 0
) -> list[Flaw]:
    """Flag a "Nick:" self-prefix, @self, @nick literally and @names not in the channel.

    `known`: the nicks and slugs that may be addressed (members, moderator, user).
    """
    out: list[Flaw] = []
    head = re.match(rf"[\s*_<\[]*{re.escape(nick)}[\s*_>\]]*[:>\]][*_]*", text, re.IGNORECASE)
    if head:
        out.append(
            Flaw(member=member, kind=FlawKind.SELF_PREFIX, excerpt=head.group(), round=round_)
        )
    allowed = {k.lower() for k in known} | _GROUP_MENTIONS
    seen: set[str] = set()
    for m in _MENTION.finditer(text):
        name = m.group(1).rstrip("-").lower()
        if name in seen:
            continue
        seen.add(name)
        if name == _NICK_LITERAL:
            kind = FlawKind.NICK_LITERAL
        elif name in (nick.lower(), member.lower()):
            kind = FlawKind.SELF_MENTION
        elif name in allowed:
            continue
        else:
            kind = FlawKind.UNKNOWN_MENTION
        out.append(Flaw(member=member, kind=kind, excerpt=f"@{m.group(1)}", round=round_))
    return out


def devil_line_flaw(text: str, member: str, line: str, round_: int = 0) -> list[Flaw]:
    """Flag a devil's advocate turn that doesn't say "Playing devil's advocate:"."""
    label = line.rstrip(" :").lower()
    if label in text.lower().replace("\u2019", "'"):
        return []
    first = " ".join(text.split())[:_EXCERPT_CHARS]
    return [Flaw(member=member, kind=FlawKind.MISSING_DEVIL_LINE, excerpt=first, round=round_)]


def check_turn(
    text: str,
    *,
    member: str,
    nick: str,
    sources: str,
    given: str,
    known: Collection[str],
    devil_line: str | None = None,
    round_: int = 0,
) -> list[Flaw]:
    """Run every check on one member turn.

    `sources`: the member's prompt minus its task (context.sources_text); `given`:
    its fresh facts and dossier (context.given_text). `devil_line`: the line a
    devil's advocate must write this turn (round 1), or None.
    """
    flaws = format_flaws(text, member, nick, known, round_)
    if devil_line is not None:
        flaws += devil_line_flaw(text, member, devil_line, round_)
    return flaws + figure_flaws(text, sources, given, member, round_)
