"""`/summon` and `/kick`: changing the channel's members between turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.engine import prompts
from privy_council.models.council import ProtocolSettings
from privy_council.models.persona import Persona
from privy_council.models.session import MessageEvent, PendingTurn, Phase, RosterChanged
from tests.engine.test_protocol import (
    MEMBERS,
    Harness,
    decision,
    standard_responder,
)
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed


def notices(h: Harness) -> list[str]:
    """Return every system line so far."""
    return [
        e.message.text
        for e in h.events
        if isinstance(e, MessageEvent) and e.message.speaker == "system"
    ]


def rosters(h: Harness) -> list[list[str]]:
    """Return the member lists announced by RosterChanged events."""
    return [e.members for e in h.events if isinstance(e, RosterChanged)]


async def test_summon_before_round_one_joins_the_blind_round(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()  # opening
    h.engine.summon("Skeptic")
    assert notices(h)[-1] == "Skeptic joined the channel and speaks next round"
    assert rosters(h) == [[*MEMBERS, "skeptic"]]
    assert h.engine.meta.runtime.pending[-1] == PendingTurn(speaker="skeptic")  # blind

    await h.engine.step()  # blind round
    [turn] = [r for r in h.requests() if r.tag.persona == "skeptic"]
    assert (
        prompts.task(prompts.BLIND_TASK, "en") in turn.messages[0].content
    )  # a blind take, like everyone's
    assert h.texts("skeptic") == ["skeptic says something in round"]
    saved = h.store.load_meta()
    assert saved.members == [*MEMBERS, "skeptic"]
    assert set(saved.models) == {"chair", *MEMBERS, "skeptic"}
    assert saved.models["skeptic"] == [f"ollama:{personas['skeptic'].meta.model}"]


async def test_summon_later_gets_a_catch_up_turn_and_joins_the_roster(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "respond")]),
        decision([("rigor", "again")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()  # round 1 done; round 2 planned for Rigor
    h.engine.summon("@counsel")
    assert h.engine.meta.runtime.pending == [
        PendingTurn(speaker="rigor", prompt="respond"),
        PendingTurn(speaker="counsel", prompt=prompts.JOIN_PROMPT["en"]),
    ]
    await h.engine.step()  # round 2
    assert f"@Counsel: {prompts.JOIN_PROMPT['en']}" in h.texts("chair")
    [turn] = [r for r in h.requests() if r.tag.persona == "counsel"]
    assert "rigor says something" in turn.messages[0].content  # sees the debate so far
    moderate = [r for r in h.requests() if r.tag.purpose == "moderate"][-1]
    assert "counsel" in moderate.system  # the moderator's roster now lists Counsel


async def test_summon_refusals(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    h.engine.summon("")
    h.engine.summon("nobody")
    h.engine.summon("rigor")
    h.engine.summon("chair")
    h.engine.busy = True
    h.engine.summon("skeptic")
    h.engine.busy = False
    assert notices(h) == [
        "usage: /summon <persona> (council personas list)",
        "No persona called 'nobody'; available: "
        "analyst, coach, counsel, counterfactual, economist, empiricist, epicurus, fan, "
        "fixer, forecaster, historian, insider, kindness, logistician, macro, meta, "
        "nietzsche, optimist, plato, polsci, realist, saver, skeptic, sociologist, socrates, "
        "specialist, steelman, stoic, strategist, value, vitality",
        "Rigor is already in the channel",
        "Chair is a moderator persona and can't join as a member",
        "Busy: wait for the current turn, or /pause",
    ]
    assert rosters(h) == []


async def test_summon_after_the_debate_ended(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(
        tmp_path,
        personas,
        standard_responder([decision(stop=True, reason="done")]),
        protocol=ProtocolSettings(max_rounds=1),
    )
    await h.engine.step()
    await h.engine.step()
    assert h.phase() is Phase.SYNTHESIS
    assert not h.engine.meta.runtime.early_stop
    h.engine.summon("optimist")
    assert h.engine.meta.runtime.pending == []
    assert notices(h)[-1].startswith("Optimist joined the channel; the debate has ended")
    await h.engine.ask("optimist", "Thoughts?")
    assert h.texts("optimist") == ["optimist says something in round"]


async def test_kick(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "go"), ("pragmatist", "go")])]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()
    h.engine.kick("@Rigor")
    assert notices(h)[-1] == "Rigor left the channel"
    assert list(h.engine.members) == ["compounder", "pragmatist", "contrarian"]
    assert h.engine.meta.runtime.pending == [PendingTurn(speaker="pragmatist", prompt="go")]
    assert h.store.load_meta().members == ["compounder", "pragmatist", "contrarian"]
    assert h.engine.nick("rigor") == "Rigor"  # old messages keep the nick

    h.engine.kick("chair")
    h.engine.kick("rigor")
    h.engine.kick("")
    for slug in ["compounder", "pragmatist"]:
        h.engine.kick(slug)
    h.engine.kick("contrarian")
    assert notices(h)[-6:] == [
        "Chair moderates this session and can't be kicked",
        "No member called 'rigor' in this channel",
        "usage: /kick <nick>",
        "Compounder left the channel",
        "Pragmatist left the channel",
        "The channel needs at least one member",
    ]


async def test_roster_survives_a_resume(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()
    h.engine.summon("skeptic")
    h.engine.kick("compounder")
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    assert list(r.engine.members) == ["rigor", "pragmatist", "contrarian", "skeptic"]
    assert PendingTurn(speaker="skeptic") in r.engine.meta.runtime.pending
    assert all(t.speaker != "compounder" for t in r.engine.meta.runtime.pending)


async def test_closed_session_refuses_roster_changes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision(stop=True, reason="done")]))
    await h.engine.step()
    await h.engine.step()
    await h.engine.close()
    h.engine.summon("skeptic")
    h.engine.kick("rigor")
    assert notices(h)[-2:] == ["The session is closed", "The session is closed"]
