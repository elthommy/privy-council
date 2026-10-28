"""Loader errors that always name the offending file and field."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError


class LoadError(Exception):
    """A persona or council file failed to load; message names file and field."""

    def __init__(self, path: Path, problems: list[str]) -> None:
        """Build the error from a file path and one line per problem."""
        self.path = path
        self.problems = problems
        super().__init__(f"{path}: " + "; ".join(problems))


def describe_validation_error(exc: ValidationError) -> list[str]:
    """Turn a Pydantic ValidationError into 'field: message' lines."""
    lines = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"{field}: {err['msg']}")
    return lines
