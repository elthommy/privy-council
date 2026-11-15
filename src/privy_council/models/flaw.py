"""Flaws in member turns (M11): found by the engine's checks or reported by the moderator."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FlawKind(StrEnum):
    """What is wrong with a passage of a member turn."""

    # Found by the engine (engine/checks.py).
    UNSOURCED = "unsourced"  # a figure found nowhere in the member's context
    MISLABELLED = "mislabelled"  # "(estimate, from memory)" on a figure from the context
    WRONG_UNIT = "wrong_unit"  # a number from the context with another unit ("60 Astrite")
    SELF_PREFIX = "self_prefix"  # the turn starts with the member's own "Nick:"
    SELF_MENTION = "self_mention"  # @own-nick
    UNKNOWN_MENTION = "unknown_mention"  # @name of someone not in the channel
    NICK_LITERAL = "nick_literal"  # a literal "@nick", copied from the task wording
    MISSING_DEVIL_LINE = "missing_devil_line"  # a devil's advocate's round 1 without its line
    # Estimate lines in councils with `estimate: probability` (M17, engine/ledger.py).
    MISSING_ESTIMATE = "missing_estimate"  # no parsable estimate (or split) line
    ESTIMATE_INCOHERENT = "estimate_incoherent"  # value outside its range, or split disagrees
    UNMARKED_CHANGE = "unmarked_change"  # the number moved without the update marker
    # Fact tags (M18, engine/facts.py): a claim checked against the fact it cites.
    UNKNOWN_CITATION = "unknown_citation"  # a tag that is no fact in the member's prompt
    MISCITED = "miscited"  # a figure the cited facts don't hold
    # Reported by the moderator (ModeratorDecision.flaws); it may also report UNSOURCED.
    ARITHMETIC = "arithmetic"
    CONTRADICTION = "contradiction"
    LOGIC = "logic"
    OTHER = "other"


# Engine kinds about fact tags (M18): the bench passed its gate on 2026-11-15
# (12 of 13 MISCITED real), so they go to the moderator with FIGURE_KINDS.
CITATION_KINDS = frozenset({FlawKind.UNKNOWN_CITATION, FlawKind.MISCITED})
# Engine kinds about figures: shown to the moderator. The formatting ones are only logged.
FIGURE_KINDS = (
    frozenset({FlawKind.UNSOURCED, FlawKind.MISLABELLED, FlawKind.WRONG_UNIT}) | CITATION_KINDS
)
# Engine kinds about estimate lines: shown to the moderator only when
# `[engine] estimate_flaws_to_moderator` is on (off until the M17 bench passes its gate).
ESTIMATE_KINDS = frozenset({FlawKind.ESTIMATE_INCOHERENT, FlawKind.UNMARKED_CHANGE})
# Engine kinds a local member may fix before its turn is posted (M19, `repair_turns`):
# only checks precise enough that a false flag rarely rewrites a right message. Not
# UNSOURCED or WRONG_UNIT (about a quarter real), UNMARKED_CHANGE (too little data),
# UNKNOWN_MENTION (may be a right name).
REPAIR_KINDS = CITATION_KINDS | {
    FlawKind.MISSING_ESTIMATE,
    FlawKind.ESTIMATE_INCOHERENT,
    FlawKind.MISSING_DEVIL_LINE,
    FlawKind.SELF_PREFIX,
    FlawKind.SELF_MENTION,
    FlawKind.NICK_LITERAL,
}
# Engine kinds whose note is a sentence, not just the figure in question.
NOTE_KINDS = ESTIMATE_KINDS | CITATION_KINDS | {FlawKind.MISSING_ESTIMATE}


class Flaw(BaseModel):
    """One flawed passage of a member turn. Flag only: the turn's text is never changed."""

    model_config = ConfigDict(frozen=True)

    member: str  # persona slug
    kind: FlawKind
    excerpt: str  # the passage, quoted from the turn (by the moderator: its quote)
    note: str = ""  # the moderator's: what is wrong; the engine's: the figure in question
    round: int = Field(default=0, ge=0)
    by: Literal["engine", "moderator"] = "engine"
