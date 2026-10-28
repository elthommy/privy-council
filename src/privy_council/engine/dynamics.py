"""Debate dynamics the engine counts for the moderator (M7). Pure functions, no I/O.

The moderator judges consensus and novelty itself, and models misjudge both. These
signals are facts it can't misread: who has spoken, who said "I update:" (or
"Je change d'avis :"), and who is repeating an earlier message almost word for word.
Loop detection combines the moderator's `new_arguments` and `decision_open` with them.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from privy_council.models.session import Message

# Members are told to start a changed position with this (prompts.LABELS "update").
# Both languages always count: a model that drifts language still updates.
_UPDATE_MARK = re.compile(r"\bI update\s*:|\bje change d['\u2019]avis\s*:", re.IGNORECASE)
# Small models also use the marker as a heading and then deny any change ("I update:
# no change", "Je change d'avis : non"), or only say what would move them ("I update:
# if ..."). Replays (2026-10-18): 6 of 8 nemotron markers were false. A marker
# directly followed by one of these doesn't count.
_NOT_AN_UPDATE = re.compile(
    r"[\W_]*(?:no change|nothing (?:has )?changed|unchanged|not changed|none\b|if\b"
    r"|my (?:position|view|estimate|range|answer|number)s? (?:is unchanged|stays|remains"
    r"|holds|stands|is the same|(?:does not|doesn['\u2019]t) change)"
    r"|i (?:still|maintain|hold|keep|stand)\b|same (?:position|view)"
    r"|non\b|pas de changement|aucun changement|inchang|je maintiens|je (?:garde|conserve)\b"
    r"|ma position (?:reste|ne change pas|est inchang))",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+")
# Word n-grams compared between messages. Paraphrases share few 3-grams; a member
# looping on its own talking points shares many.
_SHINGLE = 3
# Overlap from which a message counts as repeating an earlier one.
REPEAT_THRESHOLD = 0.5


@dataclass(frozen=True)
class MemberStats:
    """One member's speaking record, as of the end of a round."""

    slug: str
    turns: int  # messages in the channel so far, `/ask` answers included
    silent_rounds: int  # rounds since the member last spoke (0: spoke in the last one)
    updated: bool  # said "I update:" in the last round
    repeat: float  # highest overlap of a last-round message with an earlier one (0-1)


def _shingles(text: str) -> set[tuple[str, ...]]:
    """Return the word n-grams of a text, lowercased."""
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def overlap(a: str, b: str) -> float:
    """Return the Jaccard overlap of two texts' word n-grams (0: none, 1: same)."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def marks_update(text: str) -> bool:
    """Whether a message announces a position change, ignoring markers that deny one."""
    return any(not _NOT_AN_UPDATE.match(text, mark.end()) for mark in _UPDATE_MARK.finditer(text))


def _member(slug: str, messages: Sequence[Message], round_: int) -> MemberStats:
    """Compute one member's stats from the transcript."""
    own = [m for m in messages if m.speaker == slug]
    latest = [m for m in own if m.round == round_]
    earlier = [m for m in own if m.round < round_]
    spoken = [m.round for m in own if m.round > 0]
    return MemberStats(
        slug=slug,
        turns=len(own),
        silent_rounds=round_ - max(spoken, default=0),
        updated=any(marks_update(m.text) for m in latest),
        repeat=max((overlap(m.text, e.text) for m in latest for e in earlier), default=0.0),
    )


def member_stats(
    messages: Sequence[Message], members: Sequence[str], round_: int
) -> list[MemberStats]:
    """Return each member's stats after round `round_`, in roster order."""
    return [_member(slug, messages, round_) for slug in members]


def stalled(
    new_arguments: Sequence[str] | None,
    decision_open: bool,
    stats: Sequence[MemberStats],
    position_changes: Collection[str] | None = None,
) -> bool:
    """Return whether a round made no progress.

    That is: what is still disputed wouldn't change the user's decision, or the
    round brought nothing new and nobody changed position. A position change in a
    settled debate is a step toward the agreed course (live check, M7), so it
    doesn't count as progress. `None` means not judged.

    A change counts only if the member marked it and the moderator, when it judged
    `position_changes`, confirms it: the moderator can veto a false marker but not
    invent a change.
    """
    if not decision_open:
        return True
    nothing_new = new_arguments is not None and not new_arguments
    changed = any(
        s.updated and (position_changes is None or s.slug in position_changes) for s in stats
    )
    return nothing_new and not changed


def silent_members(stats: Sequence[MemberStats], planned: Collection[str], limit: int) -> list[str]:
    """Return members left out for `limit` rounds or more and not planned (0: none)."""
    if limit <= 0:
        return []
    return [s.slug for s in stats if s.silent_rounds >= limit and s.slug not in planned]
