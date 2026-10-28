"""Persona generator (M13): one Claude call drafts a persona file from a one-line description."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from privy_council.backends.base import Backend, CallTag, ChatMessage, CompletionRequest
from privy_council.engine import prompts
from privy_council.loaders.personas import parse_persona
from privy_council.memory.store import SLUG_PATTERN
from privy_council.models.persona import BackendKind, Persona

DEFAULT_SEAT_MODEL = "qwen3.6:27b"  # the general-purpose local seat
EXAMPLE_SLUG = "rigor"  # its "never makes up a figure" line is the M12 lesson
DRAFT_PURPOSE = "persona_draft"
DRAFT_MAX_TOKENS = 2000


class DraftError(Exception):
    """The model's draft is unusable; `raw` keeps what it returned, to save the call."""

    def __init__(self, message: str, raw: str) -> None:
        """Store the user-facing message and the raw draft."""
        super().__init__(message)
        self.raw = raw


class PersonaDraft(BaseModel):
    """What the model returns: the creative fields only (the seat comes from the user)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    slug: str
    role: Literal["advisor", "devil_advocate", "risk_officer"]
    color: str
    expertise: list[str] = Field(min_length=1, max_length=4)
    tags: list[str] = Field(max_length=4)
    who: str = Field(min_length=1)
    think: list[str] = Field(min_length=1, max_length=6)
    biases: list[str] = Field(min_length=1, max_length=3)
    speak: str = Field(min_length=1)

    @field_validator("slug")
    @classmethod
    def _plain_slug(cls, slug: str) -> str:
        """Refuse a slug that isn't a plain name: it becomes a file name (e.g. "../x").

        A validator, not `Field(pattern=...)`, keeps the JSON schema sent to claude as is.
        """
        if not SLUG_PATTERN.fullmatch(slug):
            raise ValueError("slug must be lowercase letters, digits, '-' or '_'")
        return slug


class Seat(BaseModel):
    """Where the new persona runs, chosen by the user, not the model."""

    backend: BackendKind = BackendKind.OLLAMA
    model: str = DEFAULT_SEAT_MODEL


def draft_schema() -> dict[str, Any]:
    """Return the JSON schema of the draft, every field required."""
    schema = PersonaDraft.model_json_schema()
    schema["required"] = sorted(schema["properties"])
    return schema


def draft_request(
    description: str, personas: Mapping[str, Persona], councils: Sequence[str], model: str
) -> CompletionRequest:
    """Build the drafting call; the description travels in the prompt (stdin), never argv."""
    example = personas.get(EXAMPLE_SLUG) or next(iter(personas.values()), None)
    task = prompts.PERSONA_DRAFT_TASK.format(
        description=description.strip(),
        names=", ".join(sorted(p.name for p in personas.values())) or "none",
        slugs=", ".join(sorted(personas)) or "none",
        colors=", ".join(sorted({p.meta.color for p in personas.values()})) or "none",
        councils=", ".join(councils) or "none",
        example=example.path.read_text(encoding="utf-8").strip() if example else "(none yet)",
    )
    return CompletionRequest(
        system=prompts.PERSONA_DRAFT_SYSTEM,
        messages=[ChatMessage(role="user", content=task)],
        model=model,
        max_output_tokens=DRAFT_MAX_TOKENS,
        json_schema=draft_schema(),
        tag=CallTag(purpose=DRAFT_PURPOSE),
    )


async def draft_persona(backend: Backend, request: CompletionRequest) -> PersonaDraft:
    """Run the call and parse the draft; raises DraftError or a BackendError."""
    result = await backend.complete(request)
    raw = json.dumps(result.structured, indent=2) if result.structured else result.text
    try:
        if result.structured is not None:
            return PersonaDraft.model_validate(result.structured)
        return PersonaDraft.model_validate_json(result.text)
    except ValidationError as exc:
        raise DraftError(f"the draft doesn't match the persona fields: {exc}", raw) from exc


def render_persona(draft: PersonaDraft, seat: Seat, tags: list[str] | None = None) -> str:
    """Assemble the persona file: frontmatter, then the four usual body sections."""
    meta = {
        "name": draft.name,
        "slug": draft.slug,
        "color": draft.color,
        "role": draft.role,
        "backend": seat.backend.value,
        "model": seat.model,
        "expertise": draft.expertise,
        "memory": True,
        "tags": tags if tags is not None else draft.tags,
    }
    head = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=None)
    body = [
        "# Who you are",
        draft.who.strip(),
        "",
        "# How you think",
        *(f"- {line.strip().removeprefix('- ')}" for line in draft.think),
        "",
        "# Biases you own",
        *(f"- {line.strip().removeprefix('- ')}" for line in draft.biases),
        "",
        "# How you speak",
        draft.speak.strip(),
    ]
    return f"---\n{head}---\n\n" + "\n".join(body) + "\n"


def check_persona(
    text: str, slug: str, personas_dir: Path, personas: Mapping[str, Persona]
) -> Persona:
    """Validate the text of `<slug>.md`; raises LoadError, or ValueError if a name is taken."""
    persona = parse_persona(text, personas_dir / f"{slug}.md")
    if persona.slug in personas or (personas_dir / f"{persona.slug}.md").exists():
        raise ValueError(f"a persona with slug {persona.slug!r} already exists (try --slug)")
    taken = {p.name.lower() for p in personas.values()}
    if persona.name.lower() in taken:
        raise ValueError(f"the nick {persona.name!r} is already used by another persona")
    return persona


def write_persona(text: str, path: Path) -> None:
    """Write a new persona file; never overwrite one (raises FileExistsError)."""
    with path.open("x", encoding="utf-8") as fh:
        fh.write(text)
