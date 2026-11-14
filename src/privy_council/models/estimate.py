"""Members' probability estimates, parsed by the engine from their estimate lines (M17)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Estimate(BaseModel):
    """A probability in percent, with the member's range when given."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0, le=100)
    low: float | None = Field(default=None, ge=0, le=100)
    high: float | None = Field(default=None, ge=0, le=100)


class Split(BaseModel):
    """A member's numbers for the moderator's split event, in percent."""

    model_config = ConfigDict(frozen=True)

    event: float = Field(ge=0, le=100)  # P(event)
    if_yes: float = Field(ge=0, le=100)  # P(outcome | event)
    if_no: float = Field(ge=0, le=100)  # P(outcome | no event)

    @property
    def implied(self) -> float:
        """Return the outcome probability these numbers imply, in percent."""
        p = self.event / 100
        return p * self.if_yes + (1 - p) * self.if_no


class EstimateRecord(BaseModel):
    """One member turn's parsed estimate (and split, when the question had one)."""

    model_config = ConfigDict(frozen=True)

    round: int = Field(ge=0)
    estimate: Estimate
    split: Split | None = None
