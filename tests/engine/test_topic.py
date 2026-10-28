"""Sessions started without a question (`council new <council>`)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from privy_council.backends.fake import FakeBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine.protocol import DebateEngine
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona
from privy_council.storage.sessions import SessionStore
from tests.engine.test_protocol import MEMBERS, standard_responder
from tests.engine.test_protocol import personas as personas


def topicless(
    tmp_path: Path, personas: dict[str, Persona]
) -> tuple[DebateEngine, SessionStore, FakeBackend]:
    """An engine and store created with an empty topic, on one fake backend."""
    settings = Settings.model_validate({"root": tmp_path, "user": {"name": "User"}})
    fake = FakeBackend(responder=standard_responder([]))
    registry = BackendRegistry(
        settings, overrides={BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
    )
    store = SessionStore.create(settings.data_dir, "finance", "", datetime.now(UTC))
    council = CouncilPreset(name="finance", moderator="chair", members=MEMBERS)
    engine = DebateEngine(
        settings=settings,
        council=council,
        personas=personas,
        registry=registry,
        topic="",
        store=store,
    )
    return engine, store, fake


async def test_actions_wait_for_the_question(tmp_path: Path, personas: dict[str, Persona]) -> None:
    engine, store, _ = topicless(tmp_path, personas)
    assert engine.awaiting_topic
    assert store.directory.name.endswith("_finance_session")
    await engine.step()
    assert await engine.run(2) == 0
    await engine.ask("rigor", "hi")
    await engine.research()
    await engine.close()
    notices = [m.text for m in engine.messages if m.speaker == "system"]
    assert notices == ["No question yet: type it to start the session"] * 5
    assert engine.meta.runtime.round == 0


async def test_set_topic_renames_the_session_and_starts(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    engine, store, fake = topicless(tmp_path, personas)
    old = store.directory
    assert engine.set_topic("   ") is False
    assert engine.set_topic("Rent or buy?\nI live in Lyon.") is True
    assert engine.set_topic("Another?") is False  # only once
    assert not old.exists()
    assert store.directory.name.endswith("_finance_rent-or-buy-i-live-in-lyon")
    assert engine.meta.id == store.id
    assert store.load_meta().topic == "Rent or buy?\nI live in Lyon."
    assert (
        (store.directory / "transcript.md")
        .read_text()
        .startswith("# #finance — Rent or buy?\nI live in Lyon.")
    )
    latest = store.directory.parent / "latest"
    assert latest.resolve() == store.directory.resolve()
    await engine.step()
    assert engine.meta.runtime.round == 0 and engine.phase.value == "round"
    opening = fake.requests[0]
    assert opening.tag.purpose == "opening" and opening.tag.session == store.id
    assert "# Question\nRent or buy?\nI live in Lyon." in opening.messages[0].content
