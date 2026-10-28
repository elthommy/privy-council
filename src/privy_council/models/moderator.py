"""Structured output schema for moderator decisions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from privy_council.models.session import Consensus, DebateState


class NextTurn(BaseModel):
    """One turn the moderator assigns: who speaks and on what."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class SummonProposal(BaseModel):
    """A persona outside the channel the moderator wants to bring in, and why."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReportedFlaw(BaseModel):
    """A clear error the moderator found in a member's message this round (M11)."""

    model_config = ConfigDict(extra="forbid")

    member: str = Field(min_length=1)
    # One of FLAW_KINDS; anything else is read as "other" rather than failing the reply.
    kind: str
    quote: str = Field(min_length=1)
    note: str = ""


# The kinds the moderator may report (the schema's enum).
FLAW_KINDS = ("arithmetic", "contradiction", "logic", "unsourced", "other")


class ModeratorDecision(BaseModel):
    """The moderator's JSON output after each round, parsed strictly."""

    model_config = ConfigDict(extra="forbid")

    debate_state: DebateState
    next_turns: list[NextTurn] = Field(default_factory=list)
    stop: bool = False
    stop_reason: str | None = None
    # Forwarded to the user by the moderator; the engine keeps at most two.
    questions_for_user: list[str] = Field(default_factory=list)
    # Web searches the moderator wants before the next round (search focuses).
    research: list[str] = Field(default_factory=list)
    consensus: Consensus = Consensus.NONE
    # What the round added, one line each ("@Nick: ..."); [] means nothing new and
    # None (a reply without the field) means not judged. For loop detection.
    new_arguments: list[str] | None = None
    # Would a remaining disagreement change what the user should do?
    decision_open: bool = True
    # Slugs of members whose position really changed this round; None (a reply
    # without the field) means not judged. Confirms the members' "I update:" markers.
    position_changes: list[str] | None = None
    summon: SummonProposal | None = None
    # Clear errors in this round's messages, most serious first (M11); a reply
    # without the field reports none.
    flaws: list[ReportedFlaw] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_stop(self) -> ModeratorDecision:
        """Require turns when not stopping.

        A stop without a reason is accepted: local models often leave it null (M15
        bench), and the engine supplies a default reason instead of failing the round.
        """
        if not self.stop and not self.next_turns:
            raise ValueError("next_turns must not be empty when stop is false")
        return self
