from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from privy_council.memory.store import (
    InvalidSlugError,
    MemoryStore,
    bullet,
    bullet_lines,
    scoped,
    size_bytes,
)


def test_bullet_is_dated_and_single_line() -> None:
    assert bullet(date(2026, 10, 17), "finance", "The user\n  rents.") == (
        "- 2026-10-17 (finance): The user rents."
    )


def test_bullet_lines_and_size() -> None:
    assert bullet_lines("# title\n- one\n* two\nprose\n  - three") == ["one", "two", "three"]
    assert size_bytes("é") == 2


def test_scoped_keeps_the_council_and_untagged_lines() -> None:
    text = "# notes\n- 2026-10-01 (finance): a\n- 2026-10-02 (career): b\n* hand-written\n"
    assert scoped(text, "finance") == "# notes\n- 2026-10-01 (finance): a\n* hand-written"
    assert scoped(text, "health") == "# notes\n* hand-written"


def test_read_private_by_council(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.append_private("rigor", ["- 2026-10-01 (finance): a", "- 2026-10-02 (career): b"])
    assert store.read_private("rigor", "career") == "- 2026-10-02 (career): b"
    assert store.read_private("rigor").count("\n") == 2


def test_missing_files_read_empty(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    assert store.read_private("rigor") == ""
    assert store.read_shared() == ""
    assert store.personas_with_notes() == []


def test_append_and_rewrite_keeps_a_backup(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.append_private("rigor", ["- a"])
    store.append_private("rigor", ["- b", "- c"])
    store.append_private("rigor", [])
    assert store.read_private("rigor") == "- a\n- b\n- c\n"
    store.rewrite_private("rigor", "- merged")
    assert store.read_private("rigor") == "- merged\n"
    backup = store.private_path("rigor").with_name("notes.md.bak")
    assert backup.read_text() == "- a\n- b\n- c\n"
    assert store.personas_with_notes() == ["rigor"]


def test_append_starts_on_a_fresh_line_after_a_manual_edit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.shared_path.parent.mkdir(parents=True)
    store.shared_path.write_text("- edited by hand")  # no trailing newline
    store.append_shared(["- new"])
    assert store.read_shared() == "- edited by hand\n- new\n"


def test_resets_delete_files_and_backups(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.append_private("rigor", ["- a"])
    store.rewrite_private("rigor", "- b")
    store.append_shared(["- fact"])
    store.rewrite_shared("- fact2")
    assert store.reset_private("rigor") is True
    assert not store.private_path("rigor").parent.exists()
    assert store.reset_private("rigor") is False
    assert store.reset_shared() is True
    assert not store.shared_path.with_name("user.md.bak").exists()
    store.append_private("pragmatist", ["- x"])
    assert store.reset_all() is True
    assert not store.root.exists()
    assert store.reset_all() is False


@pytest.mark.parametrize("slug", ["../..", "a/b", "", "Upper", "."])
def test_bad_slugs_never_touch_the_filesystem(tmp_path: Path, slug: str) -> None:
    store = MemoryStore(tmp_path / "data")
    (tmp_path / "keep.txt").write_text("precious")
    with pytest.raises(InvalidSlugError):
        store.reset_private(slug)
    assert (tmp_path / "keep.txt").exists()


def test_personas_with_notes_skips_odd_directories(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.append_private("rigor", ["- a"])
    odd = store.root / "personas" / "Not A Slug"
    odd.mkdir()
    (odd / "notes.md").write_text("- x")
    assert store.personas_with_notes() == ["rigor"]
