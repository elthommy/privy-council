"""Engine controls used by the TUI: run N, cooperative pause, /ask, estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import BackendRateLimitError, CompletionRequest
from privy_council.backends.fake import Reply
from privy_council.engine.research import ResearchMode
from privy_council.models.council import ProtocolSettings
from privy_council.models.persona import Persona
from privy_council.models.session import Phase, SessionState
from tests.engine.test_protocol import (
    MEMBERS,
    Harness,
    decision,
    standard_responder,
)
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed


async def test_run_counts_rounds_and_stops_at_synthesis(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "a")]),
        decision(stop=True, reason="consensus"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    done = await h.engine.run(5)  # opening (not counted), round 1, round 2 → stop
    assert done == 2
    assert h.phase() is Phase.SYNTHESIS
    assert h.state() is SessionState.PAUSED
    assert h.texts("system")[-1].startswith("Debate stopped early: consensus. /close")
    assert "The debate has ended: /close for the synthesis" not in h.texts("system")
    assert not h.engine.busy


async def test_run_n_stops_after_n(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "a")]) for _ in range(3)]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    assert await h.engine.run(2) == 2
    assert h.engine.meta.runtime.round == 2
    assert h.phase() is Phase.ROUND


async def test_pause_mid_blind_round_then_resume(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    base = standard_responder([decision([("rigor", "a")])])
    holder: dict[str, Harness] = {}

    def respond(req: CompletionRequest) -> Reply:
        """Press pause while Rigor (the first local turn) is answering."""
        if req.tag.persona == "rigor" and req.tag.purpose == "turn":
            holder["h"].engine.request_pause()
        return base(req)

    h = Harness(tmp_path, personas, respond)
    holder["h"] = h
    await h.engine.step()
    assert await h.engine.run(3) == 0
    rt = h.engine.meta.runtime
    assert h.state() is SessionState.PAUSED
    assert rt.round_in_progress
    assert [t.speaker for t in rt.pending] == ["pragmatist", "contrarian"]
    assert "Rigor" in [h.engine.nick(m.speaker) for m in h.engine.messages]
    assert "Pausing after the current turn…" in h.texts("system")
    assert "Paused. /step or /run N to continue" in h.texts("system")
    assert not any(r.tag.purpose == "moderate" for r in h.claude.requests)

    await h.engine.step()  # resumes the remaining turns, then moderates
    assert not h.engine.meta.runtime.round_in_progress
    turns = [r.tag.persona or "" for r in h.requests() if r.tag.purpose == "turn"]
    assert sorted(turns) == sorted(MEMBERS)  # nobody answered twice


async def test_pause_between_cross_turns(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    base = standard_responder(
        [decision([("rigor", "a"), ("contrarian", "b")]), decision([("rigor", "c")])]
    )
    holder: dict[str, Harness] = {}

    def respond(req: CompletionRequest) -> Reply:
        """Pause during Rigor's round-2 turn."""
        h = holder["h"]
        if req.tag.persona == "rigor" and h.engine.meta.runtime.round == 2:
            h.engine.request_pause()
        return base(req)

    h = Harness(tmp_path, personas, respond)
    holder["h"] = h
    await h.engine.step()
    await h.engine.step()
    await h.engine.run(1)
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["contrarian"]
    await h.engine.step()
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]  # round 3 planned


async def test_pause_when_idle_is_a_noop(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    h.engine.request_pause()
    await h.engine.step()  # the opening still runs
    assert h.phase() is Phase.ROUND


async def test_ask_single_turn(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "a")])]))
    await h.engine.step()
    await h.engine.step()
    await h.engine.ask("@Rigor", "what's your worst case?")
    ask = next(r for r in h.ollama.requests if r.tag.purpose == "ask")
    assert ask.tag.persona == "rigor"
    content = ask.messages[0].content
    assert "what's your worst case?" in content and "Debate state" in content
    assert h.texts("user")[-1] == "@Rigor what's your worst case?"
    assert h.texts("rigor")[-1] == "rigor says something in round"
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]  # plan untouched
    await h.engine.ask("nobody", "hi")
    assert "No member called 'nobody' in this channel" in h.texts("system")


async def test_busy_guard(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    h.engine.busy = True
    await h.engine.step()
    await h.engine.ask("rigor", "x")
    assert await h.engine.run(1) == 0
    assert h.texts("system").count("Busy: wait for the current turn, or /pause") == 3
    assert h.requests() == []


async def test_estimate_and_budget_warnings(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    # Compounder + Chair per round, plus the Chair's own searches (auto_max 3).
    assert h.engine.estimate_claude_calls(3) == 6 + 3
    assert h.engine.estimate_claude_calls(1) == 2 + 2  # at most 2 searches per update
    off = Harness(tmp_path, personas, standard_responder([]), research=ResearchMode.OFF)
    assert off.engine.estimate_claude_calls(3) == 6
    assert h.engine.budget_warnings(6) == []
    assert "session usage to 41/40" in h.engine.budget_warnings(41)[0]


async def test_synthesis_uses_synthesis_model(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    assert personas["chair"].meta.synthesis_model == "opus"  # the repo example
    h = Harness(tmp_path, personas, standard_responder([]))
    await h.engine.step()
    await h.engine.close()
    models = {r.tag.purpose: r.model for r in h.claude.requests}
    assert models == {"opening": "sonnet", "synthesis": "opus", "memory": "sonnet"}


async def test_synthesis_model_limit_falls_back_to_usual_model(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    base = standard_responder([])

    def respond(req: CompletionRequest) -> Reply:
        """Opus is rate-limited; sonnet still works."""
        if req.model == "opus":
            return BackendRateLimitError("opus weekly limit")
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await h.engine.step()
    await h.engine.close()
    assert h.state() is SessionState.CLOSED
    synthesis = [r.model for r in h.claude.requests if r.tag.purpose == "synthesis"]
    assert synthesis == ["opus", "sonnet"]
    assert "opus hit a usage limit; writing the synthesis with sonnet instead" in h.texts("system")


async def test_local_fallback_ignores_synthesis_model(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]), fallback=True)
    await h.engine.step()
    h.engine.claude_exhausted = True
    await h.engine.close()
    [synthesis] = [r for r in h.ollama.requests if r.tag.purpose == "synthesis"]
    assert synthesis.model == h.settings.claude.fallback_model


async def test_more_reopens_a_debate_that_reached_max_rounds(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "a")]) for _ in range(3)]
    h = Harness(
        tmp_path, personas, standard_responder(decisions), protocol=ProtocolSettings(max_rounds=2)
    )
    assert await h.engine.run(5) == 2
    assert h.phase() is Phase.SYNTHESIS
    notices = h.texts("system")
    assert (
        "Debate ended: max rounds reached. /step or /close for the synthesis, /more N adds rounds"
        in notices
    )
    assert notices[-1] == "The debate has ended: /close for the synthesis, or /more N to add rounds"
    h.engine.more(1)
    assert h.phase() is Phase.ROUND and h.engine.max_rounds == 3
    assert h.texts("system")[-1] == "1 more round(s): up to round 3. /step or /run N to continue"
    assert await h.engine.run(5) == 1  # the added round, then it ends again
    assert h.engine.meta.runtime.round == 3 and h.phase() is Phase.SYNTHESIS
    last = [r for r in h.claude.requests if r.tag.purpose == "moderate"][-1]
    assert "Round 3 of at most 3" in last.messages[0].content
    await h.engine.close()
    h.engine.more(1)
    assert h.texts("system")[-1] == "Session is already closed"


async def test_more_survives_a_resume(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "a")]) for _ in range(3)]
    responder = standard_responder(decisions)
    h = Harness(tmp_path, personas, responder, protocol=ProtocolSettings(max_rounds=1))
    await h.engine.run(1)
    h.engine.more(2)
    resumed = Resumed(h, h.store.directory, personas, responder)
    assert resumed.engine.max_rounds == 3
    assert resumed.engine.phase is Phase.ROUND
