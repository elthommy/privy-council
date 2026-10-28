"""What a comparison runs: named variants of one council (M14).

A variant changes seats (`rigor=ollama:qwen3.6:27b`), adds members (`+steelman`) or
drops them (`-insider`); a variant with no changes is the council as it is. Persona
files and council presets are never edited: a variant is applied to copies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from privy_council.models.council import CouncilPreset, MemoryMode
from privy_council.models.persona import BackendKind, Persona

NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_BACKENDS = {kind.value: kind for kind in BackendKind}


class VariantError(ValueError):
    """A variant that can't be parsed or doesn't fit the council."""


class SeatChange(BaseModel):
    """A persona run on another model; `backend` None keeps the persona's backend."""

    model_config = ConfigDict(frozen=True)

    slug: str
    model: str = Field(min_length=1)
    backend: BackendKind | None = None

    def label(self) -> str:
        """Return the change as typed, e.g. `rigor=ollama:qwen3.6:27b`."""
        prefix = f"{self.backend.value}:" if self.backend else ""
        return f"{self.slug}={prefix}{self.model}"


class Variant(BaseModel):
    """One composition of the council under comparison."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=NAME_PATTERN)
    seats: list[SeatChange] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)

    def label(self) -> str:
        """Return the changes as typed, or "as is" for the unchanged council."""
        parts = [s.label() for s in self.seats]
        parts += [f"+{slug}" for slug in self.add] + [f"-{slug}" for slug in self.drop]
        return ", ".join(parts) or "as is"


def parse_variant(text: str) -> Variant:
    """Parse `NAME[:CHANGE,...]`; a CHANGE is `slug=[backend:]model`, `+slug` or `-slug`.

    The backend prefix is needed only to switch backends: Ollama model names
    contain colons themselves (`qwen3.6:27b`), so only a known backend counts.
    """
    name, _, rest = text.strip().partition(":")
    if not re.match(NAME_PATTERN, name):
        raise VariantError(f"variant {text!r}: name {name!r} must be lowercase letters/digits/-/_")
    seats: list[SeatChange] = []
    add: list[str] = []
    drop: list[str] = []
    for part in (p.strip() for p in rest.split(",") if p.strip()):
        if part[0] in "+-":
            (add if part[0] == "+" else drop).append(part[1:].strip())
        elif "=" in part:
            seats.append(_seat_change(part))
        else:
            raise VariantError(
                f"variant {name!r}: can't read {part!r} (use slug=model, +slug or -slug)"
            )
    return Variant(name=name, seats=seats, add=add, drop=drop)


def _seat_change(part: str) -> SeatChange:
    """Parse `slug=[backend:]model`."""
    slug, _, value = (s.strip() for s in part.partition("="))
    head, sep, tail = value.partition(":")
    backend = _BACKENDS.get(head) if sep else None
    model = tail if backend else value
    if not slug or not model:
        raise VariantError(f"can't read {part!r} (use slug=model or slug=backend:model)")
    return SeatChange(slug=slug, model=model, backend=backend)


def check_variants(variants: list[Variant]) -> None:
    """Require two or more variants with distinct names."""
    if len(variants) < 2:
        raise VariantError("a comparison needs at least two --variant")
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise VariantError("variant names must be distinct")


def apply_variant(
    preset: CouncilPreset,
    personas: Mapping[str, Persona],
    variant: Variant,
    rounds: int,
) -> tuple[CouncilPreset, dict[str, Persona]]:
    """Return the council and personas a variant's session runs with.

    Every variant runs `rounds` rounds with no early stop and memory off, so
    their costs and transcripts stay comparable and nothing is remembered.
    """
    members = _roster(preset, personas, variant)
    protocol = preset.protocol.model_copy(
        update={"max_rounds": rounds, "auto_stop_on_consensus": False}
    )
    council = preset.model_copy(
        update={"members": members, "protocol": protocol, "memory": MemoryMode.OFF}
    )
    seated = dict(personas)
    for change in variant.seats:
        if change.slug != preset.moderator and change.slug not in members:
            raise VariantError(f"variant {variant.name!r}: {change.slug!r} has no seat")
        seated[change.slug] = reseat(personas[change.slug], change)
    return council, seated


def _roster(preset: CouncilPreset, personas: Mapping[str, Persona], variant: Variant) -> list[str]:
    """Return the variant's members: the council's, minus drops, plus additions."""
    for slug in variant.drop:
        if slug not in preset.members:
            raise VariantError(f"variant {variant.name!r}: -{slug} is not a member")
    for slug in variant.add:
        if slug not in personas:
            raise VariantError(f"variant {variant.name!r}: +{slug} is not a persona")
        if slug in preset.members or slug == preset.moderator:
            raise VariantError(f"variant {variant.name!r}: +{slug} already sits")
    members = [s for s in preset.members if s not in variant.drop] + list(variant.add)
    if not members:
        raise VariantError(f"variant {variant.name!r} leaves no member")
    return members


def reseat(persona: Persona, change: SeatChange) -> Persona:
    """Return the persona on the variant's model."""
    return persona.reseated(change.model, change.backend)
