"""Resume: rebuild a session from its files and lose at most the in-flight turn."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import CompletionRequest, CompletionResult, TextSink
from privy_council.backends.fake import FakeBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.engine.notices import NOTICE_RESUMED_CLOSING
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.research import ResearchMode
from privy_council.engine.resume import (
    ResumeError,
    resume_notice,
    resume_state,
)
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    Event,
    MemoryProposed,
    MessageEvent,
    PendingTurn,
    Phase,
    SessionMeta,
    SessionState,
    StateUpdated,
)
from privy_council.storage.sessions import SessionStore
from tests.engine.test_protocol import (
    MEMBERS,
    Harness,
    Responder,
    decision,
    standard_responder,
)
from tests.engine.test_protocol import personas as personas


class HangingBackend(FakeBackend):
    """A fake whose calls matching `hang_on` never return (an in-flight turn at a crash)."""

    def __init__(self, hang_on: Callable[[CompletionRequest], bool], **kw: Any) -> None:
        """Set up the fake and the "a call is hanging" signal."""
        super().__init__(**kw)
        self.hang_on = hang_on
        self.hung = asyncio.Event()

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Hang forever on matching requests; answer the rest."""
        if self.hang_on(req):
            self.requests.append(req)
            self.hung.set()
            await asyncio.Event().wait()
        return await super().complete(req, on_text)


class Resumed:
    """A resumed engine on fresh fakes, collecting events."""

    def __init__(
        self, h: Harness, directory: Path, personas: dict[str, Persona], responder: Responder
    ) -> None:
        """Reopen `directory` with the harness settings."""
        self.claude = FakeBackend(responder=responder, name="claude")
        self.ollama = FakeBackend(responder=responder, name="ollama")
        registry = BackendRegistry(
            h.settings,
            h.ledger,
            overrides={BackendKind.CLAUDE_CLI: self.claude, BackendKind.OLLAMA: self.ollama},
        )
        self.store = SessionStore.open(directory, {"rigor": "Rigor"})
        self.engine = DebateEngine.resume(
            settings=h.settings,
            council=h.engine.council,
            personas=personas,
            registry=registry,
            store=self.store,
        )
        self.events: list[Event] = []
        self.engine.subscribe(self.events.append)

    def notices(self) -> list[str]:
        """Return the system lines emitted since the resume."""
        return [
            e.message.text
            for e in self.events
            if isinstance(e, MessageEvent) and e.message.speaker == "system"
        ]

    def turns(self) -> list[tuple[str | None, str]]:
        """Return (persona, purpose) of every call made after the resume."""
        return [(r.tag.persona, r.tag.purpose) for r in self.claude.requests + self.ollama.requests]


def state_of(engine: DebateEngine) -> SessionState:
    """Return the engine state (a call, so mypy doesn't narrow it across awaits)."""
    return engine.state


def nth_call(n: int, persona: str, purpose: str) -> Callable[[CompletionRequest], bool]:
    """Match the n-th call (1-based) by `persona` for `purpose`."""
    seen = 0

    def match(req: CompletionRequest) -> bool:
        """Count matching calls; True on the n-th."""
        nonlocal seen
        if req.tag.persona == persona and req.tag.purpose == purpose:
            seen += 1
            return seen == n
        return False

    return match


def kill_snapshot(h: Harness) -> Path:
    """Copy the session directory as it is now: what a SIGKILL would leave on disk."""
    target = h.store.directory.parent / f"{h.store.id}-killed"
    shutil.copytree(h.store.directory, target)
    return target


async def test_crash_mid_round_reruns_only_the_lost_turn(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("compounder", "answer on liquidity"), ("rigor", "respond to Compounder")]),
        decision(stop=True, reason="done"),
    ]
    responder = standard_responder(decisions)
    hanging = HangingBackend(
        nth_call(2, "rigor", "turn"),
        responder=responder,
        name="ollama",
    )
    h = Harness(tmp_path, personas, responder, ollama=hanging)
    await h.engine.step()  # opening
    await h.engine.step()  # blind round
    task = asyncio.create_task(h.engine.step())  # round 2: Compounder answers, Rigor hangs
    await hanging.hung.wait()
    killed = kill_snapshot(h)
    task.cancel()
    before = list(h.engine.messages)

    r = Resumed(h, killed, personas, standard_responder(decisions))
    engine = r.engine
    assert engine.messages == before  # everything said before the kill is back
    assert engine.state is SessionState.PAUSED  # was ROUND_RUNNING on disk
    assert engine.meta.runtime.pending == [
        PendingTurn(speaker="rigor", prompt="respond to Compounder")
    ]
    engine.announce_resume()
    assert r.notices() == ["Resumed in round 2: 1 turn(s) still to run. /step continues"]

    await engine.step()
    assert sorted(r.turns()) == [("chair", "moderate"), ("rigor", "turn")]  # no Compounder
    floor = [
        m for m in engine.messages if m.speaker == "chair" and "respond to Compounder" in m.text
    ]
    assert len(floor) == 1  # the re-run turn doesn't repeat its floor line
    assert [m.speaker for m in engine.messages if m.round == 2 and m.speaker in MEMBERS] == [
        "compounder",
        "rigor",
    ]
    assert engine.state is SessionState.PAUSED and engine.phase is Phase.SYNTHESIS

    await engine.close()
    assert state_of(engine) is SessionState.CLOSED
    reloaded = SessionStore(killed).load_meta()
    assert reloaded.state is SessionState.CLOSED and reloaded.runtime.phase is Phase.DONE


async def test_crash_during_the_opening_starts_it_again(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    responder = standard_responder([decision([("rigor", "go")])])
    claude = HangingBackend(nth_call(1, "chair", "opening"), responder=responder, name="claude")
    h = Harness(tmp_path, personas, responder, claude=claude)
    task = asyncio.create_task(h.engine.step())
    await claude.hung.wait()
    killed = kill_snapshot(h)
    task.cancel()

    r = Resumed(h, killed, personas, responder)
    assert r.engine.state is SessionState.CREATED
    r.engine.announce_resume()
    assert r.notices() == ["Resumed: the opening hasn't run yet. /step starts it"]
    await r.engine.step()
    assert r.turns() == [("chair", "opening")]
    assert r.engine.phase is Phase.ROUND


async def test_resume_keeps_research_mode_and_debate_state(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        standard_responder([decision([("rigor", "go")])]),
        research=ResearchMode.OFF,
    )
    await h.engine.step()
    await h.engine.step()
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    assert r.engine.research_mode is ResearchMode.OFF
    assert r.engine.meta.runtime.debate_state.agreements == ["keep 3 months of expenses"]
    assert any(isinstance(e, StateUpdated) for e in r.engine.history)
    r.engine.announce_resume()
    assert r.notices() == ["Resumed after round 1. /step runs the next one, /close wraps up"]


async def test_closed_session_is_refused_unless_facts_await_keep(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision(stop=True, reason="done")]))
    await h.engine.step()
    await h.engine.step()
    await h.engine.close()
    assert h.state() is SessionState.CLOSED
    with pytest.raises(ResumeError, match=r"is closed; its synthesis is in synthesis\.md"):
        Resumed(h, h.store.directory, personas, standard_responder([]))

    meta = h.store.load_meta()
    meta.runtime.memory_review.facts = ["- The user rents in Lyon"]
    h.store.save_meta(meta)
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    r.engine.announce_resume()
    assert r.notices() == ["Resumed a closed session: shared facts still await /keep"]
    assert [e.facts for e in r.events if isinstance(e, MemoryProposed)] == [
        ["- The user rents in Lyon"]
    ]


async def test_crash_while_writing_memory_closes_without_rewriting(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    meta = h.store.load_meta()
    meta.state = SessionState.CLOSING
    meta.runtime.phase = Phase.SYNTHESIS
    h.store.save_meta(meta)
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    r.engine.announce_resume()
    assert r.notices() == [NOTICE_RESUMED_CLOSING]
    assert r.engine.state is SessionState.CLOSED and r.engine.phase is Phase.DONE
    assert r.turns() == []


async def test_missing_persona_is_named(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    fewer = {slug: p for slug, p in personas.items() if slug != "rigor"}
    with pytest.raises(ResumeError, match="needs persona\\(s\\) rigor"):
        Resumed(h, h.store.directory, fewer, standard_responder([]))


async def test_resume_uses_the_saved_roster_not_the_preset(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    h.engine.council = CouncilPreset(name="finance", moderator="chair", members=["rigor"])
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    assert list(r.engine.members) == MEMBERS


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (SessionState.CREATED, SessionState.CREATED),
        (SessionState.OPENING, SessionState.CREATED),
        (SessionState.ROUND_RUNNING, SessionState.PAUSED),
        (SessionState.SYNTHESIZING, SessionState.PAUSED),
        (SessionState.PAUSED, SessionState.PAUSED),
        (SessionState.ERROR_RECOVERABLE, SessionState.ERROR_RECOVERABLE),
        (SessionState.CLOSING, SessionState.CLOSING),
    ],
)
def test_resume_state(stored: SessionState, expected: SessionState) -> None:
    assert resume_state(stored) is expected


def test_resume_notice_variants() -> None:
    meta = SessionMeta(id="s", council="c", topic="t", moderator="m", members=[])
    meta.runtime.phase = Phase.SYNTHESIS
    assert "writes the synthesis" in resume_notice(meta)
    meta.state = SessionState.ERROR_RECOVERABLE
    assert resume_notice(meta) == "Resumed after an error. /step retries"
