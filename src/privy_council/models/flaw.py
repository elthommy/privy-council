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
    # Reported by the moderator (ModeratorDecision.flaws); it may also report UNSOURCED.
    ARITHMETIC = "arithmetic"
    CONTRADICTION = "contradiction"
    LOGIC = "logic"
    OTHER = "other"


# Engine kinds about figures: shown to the moderator. The formatting ones are only logged.
FIGURE_KINDS = frozenset({FlawKind.UNSOURCED, FlawKind.MISLABELLED, FlawKind.WRONG_UNIT})


class Flaw(BaseModel):
    """One flawed passage of a member turn. Flag only: the turn's text is never changed."""

    model_config = ConfigDict(frozen=True)

    member: str  # persona slug
    kind: FlawKind
    excerpt: str  # the passage, quoted from the turn (by the moderator: its quote)
    note: str = ""  # the moderator's: what is wrong; the engine's: the figure in question
    round: int = Field(default=0, ge=0)
    by: Literal["engine", "moderator"] = "engine"
