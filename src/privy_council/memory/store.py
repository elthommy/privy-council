"""Memory files under `data/memory/`: private notes per persona, shared user facts.

Plain Markdown bullets (`- 2026-10-17 (finance): ...`) the user can read, edit
or delete (brief G5). A rewrite (consolidation) first copies the old file to
`<name>.bak`; a reset deletes both, since forgetting must really forget.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import date
from pathlib import Path

MEMORY_DIR = "memory"
PERSONAS_DIR = "personas"
SHARED_DIR = "shared"
NOTES_FILE = "notes.md"
SHARED_FILE = "user.md"
BACKUP_SUFFIX = ".bak"
BULLET_PREFIXES = ("- ", "* ")
# Same rule as persona slugs; also keeps a slug from escaping the memory directory.
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
# A bullet written by `bullet()`; group 1 is its council.
TAGGED_BULLET = re.compile(r"[-*] \d{4}-\d{2}-\d{2} \(([^)]*)\):")


class InvalidSlugError(ValueError):
    """A persona slug that can't name a memory directory."""


def size_bytes(text: str) -> int:
    """Return the UTF-8 size of a memory text (caps are in bytes)."""
    return len(text.encode("utf-8"))


def bullet(day: date, council: str, text: str) -> str:
    """Render one dated memory bullet; the text is folded onto a single line."""
    one_line = " ".join(text.split())
    return f"- {day:%Y-%m-%d} ({council}): {one_line}"


def scoped(text: str, council: str) -> str:
    """Keep the lines of a memory text that belong to `council`.

    Lines without a council tag (hand-written ones) are kept: there's no way to
    tell where they belong, and dropping them would hide the user's own edits.
    """
    kept = []
    for line in text.splitlines():
        tag = TAGGED_BULLET.match(line.strip())
        if tag is None or tag.group(1) == council:
            kept.append(line)
    return "\n".join(kept)


def bullet_lines(text: str) -> list[str]:
    """Return the Markdown bullet lines of a text, without their marker."""
    lines = (line.strip() for line in text.splitlines())
    return [line[2:].strip() for line in lines if line.startswith(BULLET_PREFIXES)]


class MemoryStore:
    """Reads and writes the memory files of one data directory."""

    def __init__(self, data_dir: Path) -> None:
        """Use `data_dir/memory`; directories are created on first write."""
        self.root = data_dir / MEMORY_DIR

    def private_path(self, slug: str) -> Path:
        """Return a persona's private notes file; raise InvalidSlugError on a bad slug."""
        if not SLUG_PATTERN.fullmatch(slug):
            raise InvalidSlugError(f"invalid persona slug {slug!r}")
        return self.root / PERSONAS_DIR / slug / NOTES_FILE

    @property
    def shared_path(self) -> Path:
        """Return the shared user facts file."""
        return self.root / SHARED_DIR / SHARED_FILE

    def read_private(self, slug: str, council: str | None = None) -> str:
        """Return a persona's notes ("" if none), only `council`'s when given."""
        text = _read(self.private_path(slug))
        return scoped(text, council) if council is not None else text

    def read_shared(self) -> str:
        """Return the shared user facts, or "" if there are none."""
        return _read(self.shared_path)

    def append_private(self, slug: str, bullets: list[str]) -> None:
        """Append rendered bullets to a persona's notes."""
        _append(self.private_path(slug), bullets)

    def append_shared(self, bullets: list[str]) -> None:
        """Append rendered bullets to the shared user facts."""
        _append(self.shared_path, bullets)

    def rewrite_private(self, slug: str, text: str) -> None:
        """Replace a persona's notes (consolidation), keeping a `.bak` copy."""
        _rewrite(self.private_path(slug), text)

    def rewrite_shared(self, text: str) -> None:
        """Replace the shared user facts (approved consolidation), keeping a `.bak` copy."""
        _rewrite(self.shared_path, text)

    def personas_with_notes(self) -> list[str]:
        """Return the slugs that have a private notes file, sorted."""
        base = self.root / PERSONAS_DIR
        if not base.is_dir():
            return []
        return sorted(
            d.name
            for d in base.iterdir()
            if SLUG_PATTERN.fullmatch(d.name) and (d / NOTES_FILE).is_file()
        )

    def reset_private(self, slug: str) -> bool:
        """Delete a persona's notes (and backup); return whether there was anything."""
        directory = self.private_path(slug).parent
        existed = directory.exists()
        shutil.rmtree(directory, ignore_errors=True)
        return existed

    def reset_shared(self) -> bool:
        """Delete the shared user facts (and backup); return whether there was anything."""
        return _delete(self.shared_path)

    def reset_all(self) -> bool:
        """Delete every memory file; return whether there was anything."""
        existed = self.root.exists() and any(self.root.iterdir())
        shutil.rmtree(self.root, ignore_errors=True)
        return existed


def _read(path: Path) -> str:
    """Read a memory file, or return "" if it doesn't exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _ensure_dir(path: Path) -> None:
    """Create the parent directories of a memory file, private to the user."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)


def _append(path: Path, bullets: list[str]) -> None:
    """Append bullets to a file, starting on a fresh line."""
    if not bullets:
        return
    _ensure_dir(path)
    existing = _read(path)
    lead = "\n" if existing and not existing.endswith("\n") else ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lead + "\n".join(bullets) + "\n")


def _rewrite(path: Path, text: str) -> None:
    """Back the file up, then replace it atomically."""
    _ensure_dir(path)
    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _delete(path: Path) -> bool:
    """Delete a memory file and its backup; return whether the file existed."""
    existed = path.is_file()
    path.unlink(missing_ok=True)
    path.with_name(path.name + BACKUP_SUFFIX).unlink(missing_ok=True)
    return existed
