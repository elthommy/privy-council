"""M7 dynamic moderator, end to end with FakeBackend: loop detection, early stop,
the silent-member guard, consensus and moderator-proposed /summon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import CompletionRequest
from privy_council.backends.fake import Reply
from privy_council.engine import prompts
from privy_council.engine.protocol import MODERATE_MIN_OUTPUT_TOKENS
from privy_council.loaders.councils import load_council
from privy_council.models.council import ProtocolSettings, RebuttalStyle
from privy_council.models.persona import Persona
from privy_council.models.session import Consensus, Phase, SessionState, StateUpdated
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import Harness, Responder, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed
from tests.engine.test_roster import notices


def stale(turns: list[tuple[str, str]] | None = None, **extra: Any) -> dict[str, Any]:
    """A decision saying the round brought no new arguments."""
    return {**decision(turns or [("rigor", "go")]), "new_arguments": [], **extra}


def moderate_prompts(h: Harness) -> list[str]:
    """Return the moderator's update requests, in order."""
    return [r.messages[0].content for r in h.claude.requests if r.tag.purpose == "moderate"]


async def test_moderator_sees_the_dynamics(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 2))
    for _ in range(3):
        await h.engine.step()  # opening, blind round, round 2
    first, second = moderate_prompts(h)
    assert "# Debate dynamics (counted by the engine)" in first
    assert "- @Rigor: 1 turn(s); spoke this round" in first
    assert "changes the decision): 0. The engine ends the debate after 2" in first
    assert "stop / stop_reason: always false: the members haven't challenged" in first
    # The fake repeats itself word for word, and only Rigor spoke in round 2.
    assert "- @Rigor: 2 turn(s); spoke this round; largely repeated" in second
    assert "- @Compounder: 1 turn(s); quiet for 1 round(s)" in second
    assert 'consensus: "full" if every member' in first
    assert "optimist (@Optimist" in first  # personas outside the channel, for `summon`


async def test_stalled_rounds_stop_the_debate_early(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    turns = [("rigor", "go"), ("compounder", "answer")]
    h = Harness(tmp_path, personas, standard_responder([stale(turns)] * 3))
    await h.engine.step()
    await h.engine.step()  # blind round: never stalled, nobody has been challenged yet
    assert h.engine.meta.runtime.stalled_rounds == 0
    await h.engine.step()  # round 2: stalled 1
    assert h.engine.meta.runtime.stalled_rounds == 1
    assert h.phase() is Phase.ROUND
    await h.engine.step()  # round 3: stalled 2 -> stop
    rt = h.engine.meta.runtime
    assert h.phase() is Phase.SYNTHESIS
    assert rt.early_stop and rt.stop_reason == "no new arguments in the last 2 rounds"
    # Kept for /step, with the members quiet for 2 rounds added by the guard.
    assert [t.speaker for t in rt.pending] == ["rigor", "compounder", "pragmatist", "contrarian"]
    assert notices(h)[-1].startswith("Debate stopped early: no new arguments in the last 2")
    updates = [e for e in h.events if isinstance(e, StateUpdated)]
    assert [u.stalled_rounds for u in updates] == [0, 1, 2]
    assert updates[-1].new_arguments == [] and updates[-1].decision_open


async def test_a_position_change_resets_the_stall(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = standard_responder([stale(), stale(), stale()])

    def respond(req: CompletionRequest) -> Reply:
        """Rigor changes position in round 2."""
        if req.tag.persona == "rigor" and h.engine.meta.runtime.round == 2:
            return "I update: three months is enough"
        return base(req)

    h = Harness(tmp_path, personas, respond)
    for _ in range(3):
        await h.engine.step()
    assert h.engine.meta.runtime.stalled_rounds == 0
    assert h.phase() is Phase.ROUND


async def test_moderator_vetoes_a_false_update_marker(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = standard_responder([stale(position_changes=[])] * 3)

    def respond(req: CompletionRequest) -> Reply:
        """Rigor marks an update in round 2, but only to restate its view."""
        if req.tag.persona == "rigor" and h.engine.meta.runtime.round == 2:
            return "I update: my view is that the fund comes first, as before"
        return base(req)

    h = Harness(tmp_path, personas, respond)
    for _ in range(3):
        await h.engine.step()
    assert h.engine.meta.runtime.stalled_rounds == 1  # the moderator saw no change
    assert "position_changes: slugs of the members" in moderate_prompts(h)[0]


async def test_stall_detection_can_be_disabled(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    for protocol in (
        ProtocolSettings(stall_rounds=0),
        ProtocolSettings(auto_stop_on_consensus=False),
    ):
        h = Harness(
            tmp_path / str(id(protocol)),
            personas,
            standard_responder([stale()] * 4),
            protocol=protocol,
        )
        for _ in range(5):
            await h.engine.step()
        assert h.phase() is Phase.ROUND
        assert h.engine.meta.runtime.stalled_rounds == 3
        assert "The engine ends the debate" not in moderate_prompts(h)[0]


async def test_salon_runs_to_max_rounds(tmp_path: Path, personas: dict[str, Persona]) -> None:
    salon = load_council(REPO_ROOT / "councils" / "salon.yaml")
    settled = stale(stop=True, stop_reason="agreed", consensus="full", decision_open=False)
    h = Harness(
        tmp_path,
        personas,
        standard_responder([settled] * 5),
        protocol=salon.protocol,
        memory=salon.memory,
    )
    for _ in range(5):
        await h.engine.step()  # opening, rounds 1-4: consensus and stalls never stop it
        assert h.phase() is Phase.ROUND and not h.engine.meta.runtime.early_stop
    await h.engine.step()  # round 5 = max_rounds
    assert h.engine.meta.runtime.round == 5 and h.phase() is Phase.SYNTHESIS


async def test_step_after_an_early_stop_keeps_debating(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        decision([("pragmatist", "weigh in")], stop=True, reason="agreed"),
        decision([("rigor", "go")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    for _ in range(3):
        await h.engine.step()
    assert h.engine.meta.runtime.early_stop
    await h.engine.step()  # continues with the moderator's plan
    rt = h.engine.meta.runtime
    assert "Debate continues. Next round: Pragmatist" in notices(h)
    assert rt.round == 3 and not rt.early_stop and rt.stop_reason is None
    assert h.texts("pragmatist")[-1] == "pragmatist says something in round"
    assert h.phase() is Phase.ROUND


async def test_run_continues_after_an_early_stop_then_stops_again(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        decision([("rigor", "go")], stop=True, reason="agreed."),
        decision([("rigor", "go")], stop=True, reason="still agreed"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    assert await h.engine.run(5) == 2  # blind round, round 2, then the early stop
    assert "Debate stopped early: agreed. /close" in notices(h)[-1]  # no doubled period
    assert await h.engine.run(5) == 1  # round 3, stopped early again
    assert h.engine.meta.runtime.stop_reason == "still agreed"
    assert "The debate has ended: /close for the synthesis" not in notices(h)
    await h.engine.close()
    assert h.state() is SessionState.CLOSED


async def test_max_rounds_is_final(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(
        tmp_path,
        personas,
        standard_responder([stale()]),
        protocol=ProtocolSettings(max_rounds=1),
    )
    await h.engine.step()
    await h.engine.step()
    rt = h.engine.meta.runtime
    assert h.phase() is Phase.SYNTHESIS and not rt.early_stop and rt.pending == []
    assert "summon: always null." in moderate_prompts(h)[0]


async def test_quiet_members_get_the_floor(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 3))
    for _ in range(3):
        await h.engine.step()  # opening, blind round, round 2
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]  # quiet 1 round
    await h.engine.step()  # round 3: the others have now sat out 2 rounds
    pending = h.engine.meta.runtime.pending
    assert [t.speaker for t in pending] == ["rigor", "compounder", "pragmatist", "contrarian"]
    assert pending[1].prompt == prompts.SILENT_TURN_PROMPT["en"]
    assert "Compounder gets the floor too: quiet for 2 rounds" in notices(h)


async def test_silent_guard_off(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(
        tmp_path,
        personas,
        standard_responder([decision([("rigor", "go")])] * 3),
        protocol=ProtocolSettings(max_silent_rounds=0),
    )
    for _ in range(4):
        await h.engine.step()
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]
    h2 = Harness(
        tmp_path / "rr",
        personas,
        standard_responder([decision([("rigor", "go")])] * 3),
        protocol=ProtocolSettings(rebuttal_style=RebuttalStyle.ROUND_ROBIN),
    )
    for _ in range(4):
        await h2.engine.step()
    assert not any("gets the floor too" in n for n in notices(h2))


async def test_consensus_reaches_the_ui(tmp_path: Path, personas: dict[str, Persona]) -> None:
    raw = {**decision([("rigor", "go")]), "consensus": "partial"}
    h = Harness(tmp_path, personas, standard_responder([raw]))
    await h.engine.step()
    await h.engine.step()
    assert h.engine.meta.runtime.consensus is Consensus.PARTIAL
    update = next(e for e in h.events if isinstance(e, StateUpdated))
    assert update.consensus is Consensus.PARTIAL


def summon_responder(*summons: dict[str, str] | None) -> Responder:
    """Moderator updates that each propose the given summon (None: no proposal)."""
    return standard_responder([{**decision([("rigor", "go")]), "summon": s} for s in summons])


async def test_moderator_proposes_a_summon(tmp_path: Path, personas: dict[str, Persona]) -> None:
    proposal = {"persona": "@Optimist", "reason": "nobody argues the upside."}
    h = Harness(tmp_path, personas, summon_responder(proposal, proposal))
    await h.engine.step()
    assert await h.engine.run(3) == 1  # the proposal pauses the run, like a question
    ask = next(m for m in h.engine.messages if m.to_user)
    assert ask.speaker == "chair"
    assert ask.text == (
        "@User I'd like to bring in @Optimist: nobody argues the upside. "
        "Type /summon optimist to allow it."
    )
    assert h.engine.meta.runtime.proposed_summons == ["optimist"]
    await h.engine.step()
    assert "optimist (@Optimist" not in moderate_prompts(h)[-1]  # never offered twice...
    assert sum("bring in @Optimist" in t for t in h.texts("chair")) == 1  # ...nor relayed
    assert "optimist" not in h.engine.members  # the user decides


async def test_bad_summon_proposals_are_dropped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        summon_responder(
            {"persona": "nobody", "reason": "x"},
            {"persona": "chair", "reason": "x"},
            {"persona": "rigor", "reason": "already here"},
        ),
    )
    for _ in range(4):
        await h.engine.step()
    ignored = [n for n in notices(h) if n.startswith("Moderator proposed summoning")]
    assert ignored == [
        "Moderator proposed summoning 'nobody', who can't join; ignored",
        "Moderator proposed summoning 'chair', who can't join; ignored",
    ]
    assert not any("bring in" in t for t in h.texts("chair"))
    assert h.engine.meta.runtime.proposed_summons == []


async def test_summon_after_an_early_stop_joins_the_next_round(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        decision([("rigor", "go")], stop=True, reason="agreed"),
        decision([("rigor", "go")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    for _ in range(3):
        await h.engine.step()
    assert h.engine.meta.runtime.early_stop
    h.engine.summon("optimist")
    assert notices(h)[-1] == "Optimist joined the channel and speaks next round"
    await h.engine.step()
    assert h.texts("optimist") == ["optimist says something in round"]


async def test_resume_keeps_the_dynamics(tmp_path: Path, personas: dict[str, Persona]) -> None:
    responder = standard_responder(
        [stale(summon={"persona": "optimist", "reason": "x"}), stale(), stale()]
    )
    h = Harness(tmp_path, personas, responder)
    for _ in range(4):
        await h.engine.step()  # rounds 2 and 3 stall: early stop
    r = Resumed(h, h.store.directory, personas, standard_responder([decision([("rigor", "a")])]))
    r.engine.announce_resume()
    rt = r.engine.meta.runtime
    assert (rt.stalled_rounds, rt.early_stop, rt.proposed_summons) == (2, True, ["optimist"])
    assert r.notices()[-1].startswith("Resumed: the debate stopped early")
    await r.engine.step()
    assert r.engine.phase is Phase.ROUND and r.engine.meta.runtime.stalled_rounds == 0


async def test_moderator_updates_get_a_larger_output_cap(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()
    await h.engine.step()
    caps = {r.tag.purpose: r.max_output_tokens for r in h.claude.requests}
    assert caps["moderate"] == MODERATE_MIN_OUTPUT_TOKENS
    assert caps["opening"] == personas["chair"].meta.max_output_tokens  # Chair's own cap


async def test_moderator_is_told_what_it_already_asked(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    asking = {**decision([("rigor", "go")]), "questions_for_user": ["How much is left monthly?"]}
    h = Harness(tmp_path, personas, standard_responder([asking, decision([("rigor", "go")])]))
    for _ in range(3):
        await h.engine.step()
    first, second = moderate_prompts(h)
    assert "Never ask something the user already answered." in first
    assert 'You already asked the user: "How much is left monthly?". Never ask' in second
    assert "decide with what you know, ask nothing more" in second
    assert h.engine.meta.runtime.asked_questions == ["How much is left monthly?"]


async def test_settled_decision_counts_as_stalled(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    settled = {
        **decision([("rigor", "go")]),
        "new_arguments": ["@Rigor: €2,500 rather than €2,000"],
        "decision_open": False,
    }
    h = Harness(tmp_path, personas, standard_responder([settled] * 3))
    for _ in range(4):
        await h.engine.step()
    rt = h.engine.meta.runtime
    assert rt.early_stop
    assert rt.stop_reason == "the remaining disagreements wouldn't change the decision"


async def test_no_early_stop_after_round_1(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")], stop=True, reason="all agree blind"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()
    assert h.phase() is Phase.ROUND
    assert [t.speaker for t in h.engine.meta.runtime.pending] == ["rigor"]
