"""`/vote`: each member's final position and confidence."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VoteReply(BaseModel):
    """What a member returns when the user calls a vote (strict JSON)."""

    model_config = ConfigDict(extra="forbid")

    position: str = Field(min_length=1)
    confidence: int = Field(ge=0, le=100)  # percent


class Vote(BaseModel):
    """One row of a vote: a member's reply, or why it has none."""

    speaker: str  # member slug
    position: str = ""
    confidence: int | None = Field(default=None, ge=0, le=100)
    error: str | None = None  # set when the member gave no usable vote
    fallback: bool = False  # answered by the local fallback model instead of Claude
