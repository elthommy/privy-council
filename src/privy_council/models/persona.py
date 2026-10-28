"""Persona domain model: the YAML frontmatter schema plus the Markdown body."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    """Structural mandate a persona holds in the debate, independent of personality."""

    ADVISOR = "advisor"
    DEVIL_ADVOCATE = "devil_advocate"
    RISK_OFFICER = "risk_officer"
    MODERATOR = "moderator"
    SCRIBE = "scribe"


class BackendKind(StrEnum):
    """Which backend runs a persona's model."""

    OLLAMA = "ollama"
    CLAUDE_CLI = "claude_cli"


class PersonaFrontmatter(BaseModel):
    """Validated YAML frontmatter of a persona file."""

    # Typos in persona files should fail loudly rather than be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    color: str = Field(default="#abb2bf", pattern=r"^#[0-9a-fA-F]{6}$")
    role: Role = Role.ADVISOR
    backend: BackendKind
    model: str = Field(min_length=1)
    # Ollama only: the claude CLI has no temperature flag.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Hard cap that stops a runaway turn; None means [engine] max_output_tokens.
    # Concision itself is asked for in the prompts, not enforced here.
    max_output_tokens: int | None = Field(default=None, gt=0, le=32000)
    # Moderator only: a stronger model for the final synthesis (one call per session),
    # same backend as `model`. Ignored when the moderator runs on the local fallback.
    synthesis_model: str | None = Field(default=None, min_length=1)
    expertise: list[str] = Field(default_factory=list)
    memory: bool = True
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_synthesis_model(self) -> PersonaFrontmatter:
        """Only the moderator (or scribe) writes the synthesis."""
        if self.synthesis_model and self.role not in (Role.MODERATOR, Role.SCRIBE):
            raise ValueError("synthesis_model is only used by a moderator or scribe persona")
        return self


class Persona(BaseModel):
    """A loaded persona: frontmatter metadata, system-prompt body and source file."""

    model_config = ConfigDict(frozen=True)

    meta: PersonaFrontmatter
    body: str = Field(min_length=1)
    path: Path

    @property
    def slug(self) -> str:
        """Return the persona's unique slug."""
        return self.meta.slug

    @property
    def name(self) -> str:
        """Return the persona's display nick."""
        return self.meta.name

    def reseated(self, model: str, backend: BackendKind | None = None) -> Persona:
        """Return a copy on another model (and backend; None keeps the current one).

        A synthesis_model belongs to the persona's own backend, so a backend switch drops it.
        """
        backend = backend or self.meta.backend
        update: dict[str, object] = {"backend": backend, "model": model}
        if backend is not self.meta.backend:
            update["synthesis_model"] = None
        return self.model_copy(update={"meta": self.meta.model_copy(update=update)})
