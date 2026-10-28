"""End-to-end engine tests with FakeBackend (no real model calls)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import (
    BackendRateLimitError,
    BackendTimeoutError,
    CompletionRequest,
    CompletionResult,
)
from privy_council.backends.fake import FakeBackend, Reply
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine.budget import Ledger, UsageEntry
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.research import ResearchMode
from privy_council.loaders.personas import load_personas
from privy_council.local_only import apply_local_only
from privy_council.models.council import CouncilPreset, MemoryMode, ProtocolSettings, RebuttalStyle
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    BudgetWarning,
    ErrorEvent,
    Event,
    MessageEvent,
    Phase,
    SessionState,
    StateUpdated,
    TextDelta,
)
from privy_council.search import SearchProvider
from privy_council.storage.sessions import SessionStore
from privy_council.util.language import Language
from tests.conftest import REPO_ROOT

MEMBERS = ["rigor", "compounder", "pragmatist", "contrarian"]  # compounder is Claude-backed
TOPIC = "Should I max my PEA before building a bigger emergency fund?"


def decision(
    turns: list[tuple[str, str]] | None = None, stop: bool = False, reason: str | None = None
) -> dict[str, Any]:
    """Build a moderator decision object."""
    return {
        "debate_state": {
            "positions": {"rigor": "fund first", "Compounder": "PEA first"},
            "agreements": ["keep 3 months of expenses"],
            "disagreements": [{"topic": "order", "sides": {"rigor": "fund", "@Compounder": "PEA"}}],
            "open_questions": ["How stable is the user's income?"],
        },
        "next_turns": [{"speaker": s, "prompt": p} for s, p in (turns or [])],
        "stop": stop,
        "stop_reason": reason,
    }


Responder = Callable[[CompletionRequest], Reply]


def standard_responder(decisions: list[dict[str, Any] | str]) -> Responder:
    """Answer by call purpose; moderator updates pop `decisions` in order."""

    def respond(req: CompletionRequest) -> Reply:
        """Return the scripted reply for this request."""
        purpose, persona = req.tag.purpose, req.tag.persona
        if purpose == "opening":
            return "Restating: PEA vs emergency fund. Sub-questions: liquidity, horizon."
        if purpose in ("moderate", "repair"):
            return decisions.pop(0)
        if purpose == "synthesis":
            return "## Consensus\nKeep 3 months first."
        return f"{persona} says something in round"

    return respond


@pytest.fixture
def personas() -> dict[str, Persona]:
    """The repo's example personas."""
    loaded, _ = load_personas(REPO_ROOT / "personas")
    return loaded


class Harness:
    """An engine wired to fake backends, collecting events."""

    def __init__(
        self,
        tmp_path: Path,
        personas: dict[str, Persona],
        responder: Responder,
        *,
        protocol: ProtocolSettings | None = None,
        fallback: bool = False,
        claude_script: list[Reply] | None = None,
        ollama: FakeBackend | None = None,
        claude: FakeBackend | None = None,
        research: ResearchMode = ResearchMode.ON_DEMAND,
        config: dict[str, Any] | None = None,
        memory: MemoryMode = MemoryMode.ON,
        topic: str = TOPIC,
        lang: Language | None = None,
        local_only: bool = False,
        search: SearchProvider | None = None,
        brief: str = "",
        research_focus: str = "",
    ) -> None:
        """Build settings, fakes, store and engine.

        `local_only` reseats the personas as `council new --local-only` does and guards
        the registry; the Claude fake stays wired so a leaked call would show up.
        """
        self.settings = Settings.model_validate(
            {
                "root": tmp_path,
                "claude": {"fallback_to_local": fallback},
                "user": {"name": "User"},  # not the login name: tests must not depend on it
                **(config or {}),
            }
        )
        self.claude = claude or FakeBackend(claude_script or [], responder=responder, name="claude")
        self.ollama = ollama or FakeBackend(responder=responder, name="ollama")
        self.ledger = Ledger.in_data_dir(self.settings.data_dir)
        if local_only:
            self.settings, personas = apply_local_only(self.settings, personas, {"chair"})
        registry = BackendRegistry(
            self.settings,
            self.ledger,
            overrides={BackendKind.CLAUDE_CLI: self.claude, BackendKind.OLLAMA: self.ollama},
            local_only=local_only,
        )
        council = CouncilPreset(
            name="finance",
            moderator="chair",
            members=MEMBERS,
            protocol=protocol or ProtocolSettings(),
            memory=memory,
            moderator_brief=brief,
            research_focus=research_focus,
        )
        self.store = SessionStore.create(
            self.settings.data_dir, "finance", topic, datetime.now(UTC), {"rigor": "Rigor"}
        )
        self.engine = DebateEngine(
            settings=self.settings,
            council=council,
            personas=personas,
            registry=registry,
            topic=topic,
            store=self.store,
            research=research,
            lang=lang,
            search=search,
        )
        self.events: list[Event] = []
        self.engine.subscribe(self.events.append)

    def state(self) -> SessionState:
        """Return the engine state (a call, so mypy doesn't narrow it across awaits)."""
        return self.engine.state

    def phase(self) -> Phase:
        """Return the engine phase (see `state`)."""
        return self.engine.phase

    def requests(self) -> list[CompletionRequest]:
        """Return all requests across both fakes."""
        return self.claude.requests + self.ollama.requests

    def texts(self, speaker: str) -> list[str]:
        """Return the texts posted by `speaker`."""
        return [m.text for m in self.engine.messages if m.speaker == speaker]


async def test_full_session_end_to_end(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("@Rigor", "respond to @Compounder on liquidity"), ("compounder", "answer")]),
        decision(stop=True, reason="consensus on 3 months first"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    engine = h.engine

    await engine.step()  # opening
    assert h.state() is SessionState.PAUSED
    assert h.texts("chair")[0].startswith("Restating")

    await engine.step()  # blind round + moderator update
    assert h.state() is SessionState.PAUSED
    assert sorted(m.speaker for m in engine.messages if m.round == 1 and m.speaker in MEMBERS) == (
        sorted(MEMBERS)
    )
    state = engine.meta.runtime.debate_state
    assert set(state.positions) == {"rigor", "compounder"}  # nicks normalized to slugs
    assert state.disagreements[0].sides == {"rigor": "fund", "compounder": "PEA"}
    assert [t.speaker for t in engine.meta.runtime.pending] == ["rigor", "compounder"]

    await engine.step()  # targeted rebuttal round
    assert "@Rigor: respond to @Compounder on liquidity" in h.texts("chair")
    assert h.phase() is Phase.SYNTHESIS
    assert engine.meta.runtime.stop_reason == "consensus on 3 months first"
    assert engine.meta.runtime.early_stop  # before max_rounds: /step would keep debating

    await engine.close()  # synthesis
    assert h.state() is SessionState.CLOSED
    assert h.phase() is Phase.DONE
    assert (h.store.directory / "synthesis.md").read_text().startswith("## Consensus")
    assert h.store.load_meta().state is SessionState.CLOSED
    assert any(isinstance(e, StateUpdated) for e in h.store.events())
    assert "**" in (h.store.directory / "transcript.md").read_text()

    purposes = [r.tag.purpose for r in h.claude.requests if r.tag.persona == "chair"]
    assert purposes == ["opening", "moderate", "moderate", "synthesis", "memory"]
    assert all(r.tag.session == h.store.id for r in h.requests())
    moderate = next(r for r in h.claude.requests if r.tag.purpose == "moderate")
    assert moderate.json_schema is not None
    assert "only attribute" in moderate.messages[0].content.lower()


async def test_blind_round_hides_other_answers(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    h.engine.say("I have a mortgage ending in 2028")
    await h.engine.step()
    await h.engine.step()
    turns = [r for r in h.requests() if r.tag.purpose == "turn"]
    assert len(turns) == 4
    blind = [r for r in turns if r.tag.persona != "contrarian"]
    for req in blind:
        content = req.messages[0].content
        assert "says something" not in content
        assert "mortgage ending in 2028" in content
        assert "Round 1, blind" in content
        assert "Playing devil's advocate:" not in content
        assert '"(estimate, from memory)"' in req.system
    # The devil's advocate goes last and sees everyone's blind take.
    [advocate] = [r for r in turns if r.tag.persona == "contrarian"]
    content = advocate.messages[0].content
    for member in ("rigor", "compounder", "pragmatist"):
        assert f"{member} says something" in content
    assert "argue against the leading position" in content
    assert "mortgage ending in 2028" in content
    assert h.ollama.requests[-1].tag.persona == "contrarian"


async def test_cross_round_sees_state_history_and_interjection(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "respond to Compounder")]),
        decision([("rigor", "again")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()
    h.engine.say("I forgot: my income is variable")
    await h.engine.step()
    rigor = [r for r in h.ollama.requests if r.tag.persona == "rigor"][-1].messages[0].content
    assert "Debate state" in rigor
    assert "compounder says something" in rigor  # visible after the blind round
    assert "income is variable" in rigor
    assert "respond to Compounder" in rigor
    second_update = [r for r in h.claude.requests if r.tag.purpose == "moderate"][1]
    text = second_update.messages[0].content
    assert "income is variable" in text
    assert "contrarian says something" not in text  # already folded into the state


async def test_user_facts_reach_members_and_the_next_update(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    first = decision([("rigor", "respond to Compounder")])
    first["debate_state"]["user_facts"] = ["fixed costs: 1,900 EUR a month"]
    h = Harness(tmp_path, personas, standard_responder([first, decision([("rigor", "x")])]))
    await h.engine.step()
    await h.engine.step()
    await h.engine.step()
    rigor = [r for r in h.ollama.requests if r.tag.persona == "rigor"][-1].messages[0].content
    assert "fixed costs: 1,900 EUR a month" in rigor
    second_update = [r for r in h.claude.requests if r.tag.purpose == "moderate"][1]
    assert "fixed costs: 1,900 EUR a month" in second_update.messages[0].content
    assert "Copy the earlier user_facts word for word" in second_update.messages[0].content
    # The second update left the field empty: the facts are kept, not lost.
    assert h.store.load_meta().runtime.debate_state.user_facts == ["fixed costs: 1,900 EUR a month"]


async def test_moderator_repair_then_success(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(
        tmp_path, personas, standard_responder(["not json at all", decision([("rigor", "x")])])
    )
    await h.engine.step()
    await h.engine.step()
    chair = [r for r in h.claude.requests if r.tag.persona == "chair"]
    assert [r.tag.purpose for r in chair] == ["opening", "moderate", "repair"]
    repair = chair[-1].messages
    assert [m.role for m in repair] == ["user", "assistant", "user"]
    assert "not valid JSON" in repair[-1].content
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]
    assert any("rejected, retrying" in t for t in h.texts("system"))


async def test_moderator_fails_twice_falls_back_to_round_robin(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    bad = decision([("nobody", "x")])
    h = Harness(tmp_path, personas, standard_responder(["{oops", bad]))
    await h.engine.step()
    await h.engine.step()
    assert h.state() is SessionState.PAUSED
    assert [t.speaker for t in h.engine.meta.runtime.pending] == MEMBERS
    assert any("round-robin" in t for t in h.texts("system"))
    assert any("names no council member" in t for t in h.texts("system"))


async def test_truncated_reply_is_kept_and_flagged(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = standard_responder([decision([("rigor", "x")])])

    def respond(req: CompletionRequest) -> Reply:
        """Cut off Rigor's answer."""
        if req.tag.persona == "rigor":
            return CompletionResult(
                text="partial ans", truncated=True, latency_s=0, backend="f", model="m"
            )
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await h.engine.step()
    await h.engine.step()
    rigor = next(m for m in h.engine.messages if m.speaker == "rigor")
    assert rigor.truncated and rigor.text == "partial ans"
    assert "Rigor was cut off (hit the output cap)" in h.texts("system")


async def test_rate_limit_pauses_and_retry_skips_done_turns(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    limit = BackendRateLimitError("limit", retry_at=datetime(2026, 10, 16, 18, tzinfo=UTC))
    calls = {"compounder": 0}
    base = standard_responder([decision([("rigor", "x")])])

    def respond(req: CompletionRequest) -> Reply:
        """Rate-limit Compounder's first call only."""
        if req.tag.persona == "compounder":
            calls["compounder"] += 1
            if calls["compounder"] == 1:
                return limit
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await h.engine.step()
    await h.engine.step()
    assert h.state() is SessionState.ERROR_RECOVERABLE
    # Contrarian (devil's advocate) waits until every blind take is in.
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["compounder", "contrarian"]
    assert any(isinstance(e, ErrorEvent) and "usage limit" in e.text for e in h.events)
    assert h.store.load_meta().runtime.round_in_progress

    await h.engine.step()  # retry: only Compounder runs, then the moderator
    assert h.state() is SessionState.PAUSED
    turns = [r.tag.persona or "" for r in h.requests() if r.tag.purpose == "turn"]
    assert sorted(turns) == sorted([*MEMBERS, "compounder"])
    assert h.engine.meta.runtime.round == 1


async def test_rate_limit_with_fallback_reroutes_locally(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = standard_responder([decision([("rigor", "x")])])

    def respond(req: CompletionRequest) -> Reply:
        """Claude is exhausted for members (the fake serves both backends)."""
        if req.tag.persona == "compounder" and req.model == "sonnet":
            return BackendRateLimitError("limit")
        return base(req)

    h = Harness(tmp_path, personas, respond, fallback=True)
    await h.engine.step()
    await h.engine.step()
    compounder = next(m for m in h.engine.messages if m.speaker == "compounder")
    assert compounder.fallback
    assert h.engine.claude_exhausted
    assert any("local fallback" in t for t in h.texts("system"))


async def test_member_error_skips_turn(tmp_path: Path, personas: dict[str, Persona]) -> None:
    base = standard_responder([decision([("rigor", "x")])])

    def respond(req: CompletionRequest) -> Reply:
        """Time out Pragmatist."""
        if req.tag.persona == "pragmatist":
            return BackendTimeoutError("took too long")
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await h.engine.step()
    await h.engine.step()
    assert h.state() is SessionState.PAUSED
    assert "Pragmatist couldn't answer: took too long" in h.texts("system")
    assert not h.engine.meta.runtime.round_in_progress


@pytest.mark.parametrize(
    ("max_rounds", "expected"),
    [(3, "the moderator ended the debate"), (2, "max rounds reached")],
)
async def test_stop_without_reason_gets_a_default(
    tmp_path: Path, personas: dict[str, Persona], max_rounds: int, expected: str
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "x")]), decision(stop=True)]
    protocol = ProtocolSettings(max_rounds=max_rounds)
    h = Harness(tmp_path, personas, standard_responder(decisions), protocol=protocol)
    for _ in range(3):  # opening, round 1, round 2
        await h.engine.step()
    assert h.phase() is Phase.SYNTHESIS
    assert h.engine.meta.runtime.stop_reason == expected


async def test_max_rounds_and_manual_stop(tmp_path: Path, personas: dict[str, Persona]) -> None:
    protocol = ProtocolSettings(max_rounds=2, auto_stop_on_consensus=False)
    decisions: list[dict[str, Any] | str] = [
        decision(stop=True, reason="ignored: auto stop is off"),
        decision([("rigor", "x")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions), protocol=protocol)
    await h.engine.step()
    await h.engine.step()
    assert h.phase() is Phase.ROUND
    assert [t.speaker for t in h.engine.meta.runtime.pending] == MEMBERS  # no turns → round-robin
    await h.engine.step()
    assert h.phase() is Phase.SYNTHESIS
    assert h.engine.meta.runtime.stop_reason == "max rounds reached"
    last = [r for r in h.claude.requests if r.tag.purpose == "moderate"][-1]
    assert "this was the last round" in last.messages[0].content


async def test_round_robin_style_ignores_targeting(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    protocol = ProtocolSettings(rebuttal_style=RebuttalStyle.ROUND_ROBIN)
    h = Harness(
        tmp_path, personas, standard_responder([decision([("rigor", "x")])]), protocol=protocol
    )
    await h.engine.step()
    await h.engine.step()
    assert [t.speaker for t in h.engine.meta.runtime.pending] == MEMBERS


async def test_close_mid_debate_and_after(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    await h.engine.step()
    await h.engine.close()
    assert h.state() is SessionState.CLOSED
    synthesis = next(r for r in h.claude.requests if r.tag.purpose == "synthesis")
    assert "closed by the user" in synthesis.messages[0].content
    await h.engine.close()
    assert h.texts("system")[-1] == "Session is already closed"


async def test_synthesis_failure_is_recoverable(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = standard_responder([])
    fail = {"left": 1}

    def respond(req: CompletionRequest) -> Reply:
        """Fail the first synthesis call."""
        if req.tag.purpose == "synthesis" and fail["left"]:
            fail["left"] -= 1
            return BackendTimeoutError("slow")
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await h.engine.step()
    await h.engine.close()
    assert h.state() is SessionState.ERROR_RECOVERABLE
    await h.engine.close()
    assert h.state() is SessionState.CLOSED


async def test_opening_failure_does_not_block(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(
        tmp_path, personas, standard_responder([]), claude_script=[BackendTimeoutError("x")]
    )
    await h.engine.step()
    assert h.state() is SessionState.PAUSED
    assert h.phase() is Phase.ROUND
    assert any("Opening failed" in t for t in h.texts("system"))


async def test_budget_warning_before_round(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "x")])]))
    for _ in range(40):
        h.ledger.record(UsageEntry(session=h.store.id, model="sonnet"))
    await h.engine.step()
    await h.engine.step()
    warnings = [e.text for e in h.events if isinstance(e, BudgetWarning)]
    assert warnings and "session usage to 42/40" in warnings[0]


async def test_events_persisted_before_listeners(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    seen: list[int] = []
    deltas: list[TextDelta] = []

    def listen(event: Event) -> None:
        """Record how many events were on disk when this one arrived (deltas aside)."""
        if isinstance(event, TextDelta):
            deltas.append(event)  # typing chunks are never saved
        else:
            seen.append(len(h.store.events()))

    h.engine.subscribe(listen)
    await h.engine.step()
    assert seen == list(range(1, len(seen) + 1))
    assert deltas and deltas[0].restart and not deltas[1].restart
    lines = (h.store.directory / "transcript.jsonl").read_text().splitlines()
    assert all(json.loads(line)["kind"] != "delta" for line in lines)
    assert any(isinstance(e, MessageEvent) for e in h.store.events())
