"""Council preset model: a reusable roster of personas plus protocol settings."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RebuttalStyle(StrEnum):
    """How speakers are picked in cross-examination rounds."""

    TARGETED = "targeted"
    ROUND_ROBIN = "round_robin"


class MemoryMode(StrEnum):
    """What a council's sessions do with memory (personas' own `memory` flag still applies)."""

    ON = "on"
    READ_ONLY = "read_only"  # notes and shared facts reach prompts; nothing written at close
    OFF = "off"


class ProtocolSettings(BaseModel):
    """Default debate protocol settings for a council."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_first_round: bool = True
    max_rounds: int = Field(default=6, ge=1, le=50)
    auto_stop_on_consensus: bool = True
    rebuttal_style: RebuttalStyle = RebuttalStyle.TARGETED
    # Moderator searches the web before its opening (needs a Claude moderator).
    auto_research: bool = False
    # Like auto_research, but mandatory: no --no-research, and a failed first search
    # pauses the session (/step retries) instead of debating without facts.
    require_research: bool = False
    # Stop early after this many rounds in a row with no new arguments (0: never).
    stall_rounds: int = Field(default=2, ge=0, le=20)
    # Give the floor to a member left out of this many rounds in a row (0: never).
    max_silent_rounds: int = Field(default=2, ge=0, le=20)


class CouncilPreset(BaseModel):
    """A named council: moderator slug, member slugs and protocol defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    moderator: str = Field(min_length=1)
    members: list[str] = Field(min_length=1)
    protocol: ProtocolSettings = Field(default_factory=ProtocolSettings)
    # Council-wide, so it also covers Chair, which moderates every council.
    memory: MemoryMode = MemoryMode.ON
    # The council's own context for its moderator, so Chair stays one persona: what
    # this domain needs from the user, added to its system prompt ...
    moderator_brief: str = ""
    # ... and what the first web search looks for ("" is the engine's default focus).
    research_focus: str = ""
    path: Path | None = Field(default=None, exclude=True)

    @field_validator("memory", mode="before")
    @classmethod
    def _yaml_switch(cls, value: object) -> object:
        """Map bare `on`/`off`, which YAML reads as booleans, back to the mode names."""
        if isinstance(value, bool):
            return MemoryMode.ON if value else MemoryMode.OFF
        return value

    @model_validator(mode="after")
    def _check_roster(self) -> CouncilPreset:
        """Reject duplicate members and a moderator listed as a member."""
        if len(set(self.members)) != len(self.members):
            raise ValueError("members contains duplicates")
        if self.moderator in self.members:
            raise ValueError(f"moderator {self.moderator!r} must not also be a member")
        return self
