"""Parse and validate persona Markdown files (YAML frontmatter + body)."""

from __future__ import annotations

from pathlib import Path

import frontmatter
import yaml
from pydantic import ValidationError

from privy_council.loaders.errors import LoadError, describe_validation_error
from privy_council.models.persona import Persona, PersonaFrontmatter


def load_persona(path: Path) -> Persona:
    """Load one persona file, raising LoadError naming the file and field."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LoadError(path, [f"cannot read file: {exc.strerror}"]) from exc
    return parse_persona(text, path)


def parse_persona(text: str, path: Path) -> Persona:
    """Validate persona file text meant for `path` (which need not exist yet)."""
    try:
        post = frontmatter.loads(text)
    except yaml.YAMLError as exc:
        raise LoadError(path, [f"invalid YAML frontmatter: {exc}"]) from exc
    if not post.metadata:
        raise LoadError(path, ["missing YAML frontmatter (expected a leading '---' block)"])
    try:
        meta = PersonaFrontmatter.model_validate(post.metadata)
    except ValidationError as exc:
        raise LoadError(path, describe_validation_error(exc)) from exc
    if meta.slug != path.stem:
        raise LoadError(path, [f"slug: {meta.slug!r} must match the filename {path.stem!r}"])
    body = post.content.strip()
    if not body:
        raise LoadError(
            path, ["body: the persona prompt (Markdown after the frontmatter) is empty"]
        )
    return Persona(meta=meta, body=body, path=path)


def load_personas(directory: Path) -> tuple[dict[str, Persona], list[LoadError]]:
    """Load every `*.md` persona in a directory.

    Returns personas keyed by slug, plus all errors found, so that one broken
    file doesn't hide the others. Duplicate nicks are reported as errors.
    """
    personas: dict[str, Persona] = {}
    errors: list[LoadError] = []
    nicks: dict[str, Path] = {}
    for path in sorted(directory.glob("*.md")):
        try:
            persona = load_persona(path)
        except LoadError as exc:
            errors.append(exc)
            continue
        nick = persona.name.lower()
        if nick in nicks:
            errors.append(
                LoadError(path, [f"name: nick {persona.name!r} already used by {nicks[nick]}"])
            )
            continue
        nicks[nick] = path
        personas[persona.slug] = persona
    return personas, errors
