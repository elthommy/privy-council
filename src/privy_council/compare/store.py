"""A comparison on disk: `data/compare/<id>/compare.json`, its judge and reports.

Each variant's debate is an ordinary session under `data/sessions/`, so it can
be shown and exported like any other; compare.json links them.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from privy_council.compare.spec import Variant
from privy_council.models.session import ResearchNote
from privy_council.storage.sessions import topic_slug
from privy_council.util.language import Language

COMPARE_DIR = "compare"  # under the data dir, gitignored: comparisons hold the user's questions
COMPARE_JSON = "compare.json"
JUDGE_MD = "judge.md"
STOP_FILE = "STOP"  # created by the user to stop at the next turn boundary


class CompareLookupError(Exception):
    """No comparison matches the reference, or several do."""


class VariantStatus(StrEnum):
    """Where one variant's debate stands."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # stopped on an error; `council compare resume` retries it


class VariantRun(BaseModel):
    """A variant and its session."""

    variant: Variant
    session: str | None = None  # session directory name, once created
    status: VariantStatus = VariantStatus.PENDING
    error: str | None = None


class CompareState(BaseModel):
    """Contents of compare.json."""

    id: str
    council: str
    topic: str
    rounds: int = Field(ge=1)
    lang: Language | None = None
    research: str  # ResearchMode value of the first variant, the one that searches
    judge: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    variants: list[VariantRun]
    # The first variant's research, given to the others; None until its opening ran.
    shared_research: list[ResearchNote] | None = None
    judged: bool = False
    # Local-only (M15): every variant runs with no Claude call; resume keeps it.
    local_only: bool = False

    @property
    def done(self) -> bool:
        """Return whether every variant's debate is over."""
        return all(v.status is VariantStatus.DONE for v in self.variants)


class CompareStore:
    """Reads and writes one comparison directory."""

    def __init__(self, directory: Path) -> None:
        """Use an existing comparison directory."""
        self.directory = directory

    @property
    def id(self) -> str:
        """Return the comparison id (its directory name)."""
        return self.directory.name

    @property
    def stop_file(self) -> Path:
        """Return the file whose presence asks a running comparison to stop."""
        return self.directory / STOP_FILE

    @classmethod
    def create(cls, data_dir: Path, council: str, topic: str, now: datetime) -> CompareStore:
        """Make a new `<date>_<council>_<topic-slug>` directory (`-N` if taken)."""
        base = data_dir / COMPARE_DIR
        name = f"{now.astimezone():%Y-%m-%d}_{council}_{topic_slug(topic)}"
        directory, n = base / name, 1
        while directory.exists():
            n += 1
            directory = base / f"{name}-{n}"
        directory.mkdir(parents=True)
        return cls(directory)

    def load(self) -> CompareState:
        """Read compare.json (raises OSError or ValueError)."""
        text = (self.directory / COMPARE_JSON).read_text(encoding="utf-8")
        return CompareState.model_validate_json(text)

    def save(self, state: CompareState) -> None:
        """Write compare.json atomically."""
        path = self.directory / COMPARE_JSON
        tmp = path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def save_judge(self, text: str) -> None:
        """Write the judge's verdict."""
        (self.directory / JUDGE_MD).write_text(text.strip() + "\n", encoding="utf-8")

    def load_judge(self) -> str | None:
        """Return the judge's verdict, if any."""
        path = self.directory / JUDGE_MD
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def list_compares(data_dir: Path) -> list[CompareStore]:
    """Return the saved comparisons, newest first."""
    base = data_dir / COMPARE_DIR
    if not base.is_dir():
        return []
    stores = [CompareStore(d) for d in base.iterdir() if (d / COMPARE_JSON).is_file()]
    return sorted(stores, key=lambda s: (s.directory / COMPARE_JSON).stat().st_mtime, reverse=True)


def find_compare(data_dir: Path, ref: str | None) -> CompareStore:
    """Return the comparison whose id contains `ref` (None: the latest)."""
    stores = list_compares(data_dir)
    if not stores:
        raise CompareLookupError("no comparisons yet")
    if ref is None:
        return stores[0]
    exact = [s for s in stores if s.id == ref]
    matches = exact or [s for s in stores if ref in s.id]
    if not matches:
        raise CompareLookupError(f"no comparison matches {ref!r}")
    if len(matches) > 1:
        names = ", ".join(s.id for s in matches)
        raise CompareLookupError(f"{ref!r} matches several comparisons: {names}")
    return matches[0]
