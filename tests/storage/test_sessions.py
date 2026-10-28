from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from privy_council.models.session import (
    BudgetWarning,
    DebateState,
    ErrorEvent,
    Event,
    Message,
    MessageEvent,
    RoundStarted,
    SessionMeta,
    SessionState,
)
from privy_council.storage.sessions import (
    SessionLookupError,
    SessionStore,
    find_session,
    list_sessions,
    remove_session,
    topic_slug,
)

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def test_topic_slug() -> None:
    assert topic_slug("Should I max my PEA?") == "should-i-max-my-pea"
    assert topic_slug("!!!") == "session"
    long = topic_slug("word " * 30)
    assert len(long) <= 40 and not long.endswith("-")


def test_create_unique_dirs(tmp_path: Path) -> None:
    a = SessionStore.create(tmp_path, "finance", "Topic", NOW)
    b = SessionStore.create(tmp_path, "finance", "Topic", NOW)
    assert a.id == f"{NOW.astimezone():%Y-%m-%d}_finance_topic"
    assert b.id == f"{a.id}-2"
    assert a.directory.parent == tmp_path / "sessions"


def test_append_and_read_back(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "finance", "Topic", NOW, {"rigor": "Rigor"})
    events: list[Event] = [
        RoundStarted(round=1, label="Round 1 — blind takes"),
        MessageEvent(message=Message(speaker="rigor", text="careful", round=1)),
        MessageEvent(message=Message(speaker="user", text="noted")),
        MessageEvent(message=Message(speaker="system", text="notice")),
    ]
    for event in events:
        store.append(event)
    assert store.events() == events
    md = (store.directory / "transcript.md").read_text()
    assert "## Round 1 — blind takes" in md
    assert "<Rigor>** careful" in md and "<User>** noted" in md and "*notice*" in md


def test_meta_state_and_synthesis(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "finance", "Topic", NOW)
    meta = SessionMeta(
        id=store.id, council="finance", topic="Topic", moderator="chair", members=["rigor"]
    )
    meta.state = SessionState.PAUSED
    store.save_meta(meta)
    assert store.load_meta() == meta
    store.save_debate_state(DebateState(agreements=["x"]))
    assert "agreements" in (store.directory / "debate_state.json").read_text()
    store.save_synthesis("## Consensus\n")
    assert (store.directory / "synthesis.md").read_text() == "## Consensus\n"
    assert not list(store.directory.glob("*.tmp"))


def test_live_markdown_header_flags_and_alerts(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "finance", "Rent or buy?", NOW, {"rigor": "Rigor"})
    store.append(MessageEvent(message=Message(speaker="rigor", text="part", truncated=True)))
    store.append(MessageEvent(message=Message(speaker="rigor", text="local", fallback=True)))
    store.append(ErrorEvent(text="claude limit"))
    store.append(BudgetWarning(text="near cap"))
    md = store.transcript_md.read_text()
    assert md.startswith("# #finance — Rent or buy?\n\n*Started ")
    assert "<Rigor>** *[cut off]* part" in md
    assert "<Rigor>** *[fallback]* local" in md
    assert "> **Error:** claude limit" in md and "> **Budget:** near cap" in md


def test_latest_link_follows_newest_session(tmp_path: Path) -> None:
    first = SessionStore.create(tmp_path, "finance", "One", NOW)
    latest = tmp_path / "sessions" / "latest"
    assert latest.resolve() == first.directory.resolve()
    second = SessionStore.create(tmp_path, "finance", "Two", NOW)
    assert latest.resolve() == second.directory.resolve()
    assert (latest / "transcript.md").is_file()


def _saved(tmp_path: Path, topic: str, now: datetime) -> SessionStore:
    """Create a session with a session.json, as the engine would."""
    store = SessionStore.create(tmp_path, "finance", topic, now)
    store.save_meta(
        SessionMeta(
            id=store.id,
            council="finance",
            topic=topic,
            moderator="chair",
            members=[],
            created_at=now,
        )
    )
    return store


def test_list_sessions_newest_first_and_unreadable(tmp_path: Path) -> None:
    old = _saved(tmp_path, "Rent or buy", NOW)
    new = _saved(tmp_path, "PEA first", NOW)
    os.utime(old.directory / "session.json", (1, 1))
    os.utime(old.directory, (1, 1))
    broken = tmp_path / "sessions" / "broken"
    broken.mkdir()
    (broken / "session.json").write_text("{not json")
    os.utime(broken / "session.json", (0, 0))
    os.utime(broken, (0, 0))
    infos = list_sessions(tmp_path)
    assert [i.id for i in infos] == [new.id, old.id, "broken"]  # `latest` is skipped
    assert infos[0].meta is not None and infos[0].meta.topic == "PEA first"
    assert infos[2].meta is None


def test_find_session(tmp_path: Path) -> None:
    with pytest.raises(SessionLookupError, match="No saved sessions"):
        find_session(tmp_path, None)
    a = _saved(tmp_path, "Rent or buy", NOW)
    b = _saved(tmp_path, "Rent or sell", NOW)
    os.utime(a.directory / "session.json", (1, 1))
    os.utime(a.directory, (1, 1))
    assert find_session(tmp_path, None) == b.directory
    assert find_session(tmp_path, "latest") == b.directory
    assert find_session(tmp_path, a.id) == a.directory
    assert find_session(tmp_path, "buy") == a.directory
    with pytest.raises(SessionLookupError, match="matches 2 sessions"):
        find_session(tmp_path, "rent-or")
    with pytest.raises(SessionLookupError, match="No session matches 'nope'"):
        find_session(tmp_path, "nope")


def test_open_points_latest_and_remove(tmp_path: Path) -> None:
    a = _saved(tmp_path, "Rent or buy", NOW)
    b = _saved(tmp_path, "PEA first", NOW)
    latest = tmp_path / "sessions" / "latest"
    assert latest.resolve() == b.directory.resolve()
    opened = SessionStore.open(a.directory)
    assert latest.resolve() == a.directory.resolve()
    assert opened.created == NOW  # from session.json, so a retopic keeps the date
    remove_session(a.directory)
    assert not a.directory.exists() and not latest.is_symlink()
