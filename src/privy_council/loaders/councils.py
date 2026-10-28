"""Parse and validate council preset YAML files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from privy_council.loaders.errors import LoadError, describe_validation_error
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import Persona, Role


def load_council(path: Path) -> CouncilPreset:
    """Load one council file, raising LoadError naming the file and field."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LoadError(path, [f"cannot read file: {exc.strerror}"]) from exc
    except yaml.YAMLError as exc:
        raise LoadError(path, [f"invalid YAML: {exc}"]) from exc
    if not isinstance(data, dict):
        raise LoadError(path, ["(root): expected a YAML mapping"])
    try:
        council = CouncilPreset.model_validate({**data, "path": path})
    except ValidationError as exc:
        raise LoadError(path, describe_validation_error(exc)) from exc
    if council.name != path.stem:
        raise LoadError(path, [f"name: {council.name!r} must match the filename {path.stem!r}"])
    return council


def check_roster(council: CouncilPreset, personas: Mapping[str, Persona]) -> list[str]:
    """Return problems with a council's roster against the available personas."""
    problems = []
    mod = personas.get(council.moderator)
    if mod is None:
        problems.append(f"moderator: unknown persona {council.moderator!r}")
    elif mod.meta.role is not Role.MODERATOR:
        problems.append(
            f"moderator: persona {council.moderator!r} has role {mod.meta.role.value!r}, "
            "expected 'moderator'"
        )
    for i, slug in enumerate(council.members):
        member = personas.get(slug)
        if member is None:
            problems.append(f"members.{i}: unknown persona {slug!r}")
        elif member.meta.role is Role.MODERATOR:
            problems.append(f"members.{i}: {slug!r} is a moderator and can't be a member")
    return problems


def load_councils(
    directory: Path, personas: Mapping[str, Persona]
) -> tuple[dict[str, CouncilPreset], list[LoadError]]:
    """Load and roster-check every `*.yaml`/`*.yml` council; collect all errors."""
    councils: dict[str, CouncilPreset] = {}
    errors: list[LoadError] = []
    paths = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    for path in paths:
        try:
            council = load_council(path)
        except LoadError as exc:
            errors.append(exc)
            continue
        problems = check_roster(council, personas)
        if problems:
            errors.append(LoadError(path, problems))
            continue
        councils[council.name] = council
    return councils, errors
