from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from privy_council.models.persona import BackendKind, Persona, PersonaFrontmatter
from privy_council.models.session import (
    DebateState,
    ErrorEvent,
    Event,
    Message,
    MessageEvent,
    SessionMeta,
    StateUpdated,
)

EVENTS: TypeAdapter[Event] = TypeAdapter(Event)


def test_events_roundtrip_through_json() -> None:
    events: list[Event] = [
        MessageEvent(message=Message(speaker="rigor", text="hi", round=1)),
        StateUpdated(state=DebateState(agreements=["x"])),
        ErrorEvent(text="ollama down"),
    ]
    for event in events:
        restored = EVENTS.validate_json(EVENTS.dump_json(event))
        assert restored == event


def test_timestamps_are_timezone_aware() -> None:
    assert Message(speaker="user", text="hi").ts.tzinfo is not None


def _persona(model: str) -> Persona:
    """A minimal Ollama persona on `model`."""
    meta = PersonaFrontmatter(name="Rigor", slug="rigor", backend=BackendKind.OLLAMA, model=model)
    return Persona(meta=meta, body="You are Rigor.", path=Path("rigor.md"))


def test_record_model_keeps_each_seat_change_once() -> None:
    meta = SessionMeta(id="s", council="c", topic="t", moderator="chair", members=["rigor"])
    meta.record_model(_persona("qwen3.6:27b"))
    meta.record_model(_persona("qwen3.6:27b"))  # resumed, same model
    meta.record_model(_persona("nemotron-3.5-super:latest"))  # persona edited between runs
    assert meta.models == {"rigor": ["ollama:qwen3.6:27b", "ollama:nemotron-3.5-super:latest"]}


def test_sessions_from_before_m13_have_no_models() -> None:
    raw = '{"id": "s", "council": "c", "topic": "t", "moderator": "chair", "members": []}'
    assert SessionMeta.model_validate_json(raw).models == {}
