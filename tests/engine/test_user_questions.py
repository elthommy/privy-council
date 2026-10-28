"""The moderator forwards questions to the user; `/run` pauses for them, `/step` doesn't."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import CompletionRequest
from privy_council.backends.fake import Reply
from privy_council.engine.budget import UsageEntry
from privy_council.engine.research import ResearchMode
from privy_council.models.council import ProtocolSettings
from privy_council.models.persona import Persona
from tests.engine.test_protocol import Harness, Responder, decision, standard_responder
from tests.engine.test_protocol import personas as personas

QUESTION = "How many months of expenses do you already have saved?"


def asking(turns: list[tuple[str, str]], *questions: str, **kw: Any) -> dict[str, Any]:
    """A moderator decision that also forwards questions to the user."""
    return {**decision(turns, **kw), "questions_for_user": list(questions)}


def questions_posted(h: Harness) -> list[str]:
    """Return the texts of messages addressed to the user."""
    return [m.text for m in h.engine.messages if m.to_user]


async def test_run_pauses_after_the_round_that_asked(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        asking([("rigor", "go")], QUESTION, "Is your job stable?", "A third one?"),
        decision([("rigor", "go")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    done = await h.engine.run(5)
    assert done == 2  # the blind round, then the round whose update asked
    assert h.engine.meta.runtime.round == 2
    assert questions_posted(h) == [f"@User {QUESTION}\n@User Is your job stable?"]
    chair_question = next(m for m in h.engine.messages if m.to_user)
    assert chair_question.speaker == "chair"
    notices = [m.text for m in h.engine.messages if m.speaker == "system"]
    assert notices[-1] == "/run paused for Chair's question"

    h.engine.say("About 2 months.")
    assert await h.engine.run(1) == 1  # carries on once the user wants
    moderate = [r for r in h.claude.requests if r.tag.purpose == "moderate"][-1]
    assert QUESTION in moderate.messages[0].content  # it sees what it asked...
    assert "About 2 months." in moderate.messages[0].content  # ...and the answer
    md = (h.store.directory / "transcript.md").read_text()
    assert "*[question for you]*" in md


async def test_a_question_asked_before_is_not_asked_again(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        asking([("rigor", "go")], QUESTION),
        asking([("rigor", "go")], f"@User: {QUESTION}"),  # re-asked, a colon added
        decision([("rigor", "go")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    assert await h.engine.run(5) == 2
    h.engine.say("About 2 months.")
    assert await h.engine.run(2) == 2  # no second pause for the same question
    assert questions_posted(h) == [f"@User {QUESTION}"]


async def test_step_shows_questions_without_extra_pause(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([asking([("rigor", "go")], QUESTION)]))
    await h.engine.step()
    await h.engine.step()
    assert questions_posted(h) == [f"@User {QUESTION}"]
    notices = [m.text for m in h.engine.messages if m.speaker == "system"]
    assert not any("question for you" in n for n in notices)


async def test_questions_are_forwarded_when_the_debate_stops(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        asking([], QUESTION, stop=True, reason="agreed"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.run(3)  # no early stop after round 1: the stop comes after round 2
    assert questions_posted(h) == [f"@User {QUESTION}"]
    assert h.engine.phase.value == "synthesis"


async def test_personas_address_the_user_by_name(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [asking([("rigor", "go")], QUESTION)]
    h = Harness(tmp_path, personas, standard_responder(decisions), config={"user": {"name": "Ada"}})
    h.engine.say("I rent.")
    await h.engine.step()
    await h.engine.step()
    assert questions_posted(h) == [f"@Ada {QUESTION}"]
    turn = next(r for r in h.ollama.requests if r.tag.persona == "rigor")
    assert "The user's name is Ada; address them by name." in turn.system
    assert "<Ada> I rent." in turn.messages[0].content
    moderate = next(r for r in h.claude.requests if r.tag.purpose == "moderate")
    assert "The user's name is Ada" in moderate.system
    assert "<Ada>** I rent." in (h.store.directory / "transcript.md").read_text()


FACT = "- S&P 500 closed at 7,700 on 16 Oct 2026 (source: example.org)."


def researching(*focuses: str) -> Responder:
    """Moderator updates request `focuses`; research calls return FACT."""
    base = standard_responder([{**decision([("rigor", "go")]), "research": list(focuses)}] * 3)

    def respond(req: CompletionRequest) -> Reply:
        """Answer research calls with FACT, else the standard replies."""
        return FACT if req.tag.purpose == "research" else base(req)

    return respond


async def test_chair_researches_on_its_own_before_the_next_round(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, researching("S&P 500 close on 16 Oct 2026"))
    await h.engine.step()
    await h.engine.step()  # blind round + update, which asks for one search
    research = [r for r in h.claude.requests if r.tag.purpose == "research"]
    assert len(research) == 1 and research[0].tools
    [note] = h.engine.meta.runtime.research
    assert note.auto and note.focus == "S&P 500 close on 16 Oct 2026"
    notices = [m.text for m in h.engine.messages if m.speaker == "system"]
    assert "Chair is looking up: S&P 500 close on 16 Oct 2026" in notices
    assert not questions_posted(h)
    moderate = next(r for r in h.claude.requests if r.tag.purpose == "moderate")
    assert "at most 2 web searches" in moderate.messages[0].content
    await h.engine.step()  # the next round argues with the facts
    turn = [r for r in h.ollama.requests if r.tag.persona == "rigor"][-1]
    assert "7,700 on 16 Oct" in turn.messages[0].content


async def test_past_the_auto_limit_the_chair_asks_first(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config: dict[str, Any] = {"research": {"auto_max_per_session": 1}}
    h = Harness(tmp_path, personas, researching("VIX close", "Nasdaq close"), config=config)
    assert await h.engine.run(3) == 1  # paused: one lookup is relayed as a question
    assert len(h.engine.meta.runtime.research) == 1
    assert questions_posted(h) == [
        "@User I'd like to look up: Nasdaq close (web searches so far this session: 1). "
        "Type /research Nasdaq close to allow it."
    ]
    await h.engine.research("Nasdaq close")  # the user allows it: it runs
    assert [n.focus for n in h.engine.meta.runtime.research] == ["VIX close", "Nasdaq close"]


async def test_the_chair_keeps_asking_after_many_user_searches(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config: dict[str, Any] = {"research": {"auto_max_per_session": 0}}
    h = Harness(tmp_path, personas, researching("VIX close"), config=config)
    await h.engine.step()
    for focus in ["a", "b", "c", "d", "e", "f"]:
        await h.engine.research(focus)
    await h.engine.step()
    assert questions_posted(h) == [
        "@User I'd like to look up: VIX close (web searches so far this session: 6). "
        "Type /research VIX close to allow it."
    ]


async def test_lookups_go_to_the_user_when_research_is_off(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    mode, reason = ResearchMode.OFF, "web research is off"
    h = Harness(tmp_path, personas, researching("VIX close"), research=mode)
    await h.engine.step()
    await h.engine.step()
    assert not [r for r in h.claude.requests if r.tag.purpose == "research"]
    assert questions_posted(h) == [
        f"@User I can't look up VIX close ({reason}); answer if you know."
    ]
    moderate = next(r for r in h.claude.requests if r.tag.purpose == "moderate")
    rule_off = "web research is off: always empty" in moderate.messages[0].content
    assert rule_off is (mode is ResearchMode.OFF)


async def test_budget_worries_make_the_chair_ask(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, researching("VIX close"))
    await h.engine.step()
    for _ in range(40):  # at the session soft cap
        h.ledger.record(UsageEntry(session=h.store.id, model="sonnet"))
    await h.engine.step()
    assert not h.engine.meta.runtime.research
    assert questions_posted(h) == [
        "@User I'd like to look up: VIX close (the Claude budget is getting tight). "
        "Type /research VIX close to allow it."
    ]


async def test_no_chair_search_on_the_last_round(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    protocol = ProtocolSettings(max_rounds=2)
    h = Harness(tmp_path, personas, researching("VIX close"), protocol=protocol)
    await h.engine.run(2)
    # Round 1's update searches; the last one, which ends the debate, doesn't.
    assert [n.focus for n in h.engine.meta.runtime.research] == ["VIX close"]
    moderate = [r for r in h.claude.requests if r.tag.purpose == "moderate"]
    assert "at most 2 web searches" in moderate[0].messages[0].content
    assert "this is the last update" in moderate[-1].messages[0].content
    assert h.engine.phase.value == "synthesis"


async def test_no_chair_search_when_it_stops_early(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        {**decision([("rigor", "go")], stop=True, reason="agreed"), "research": ["VIX close"]},
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.run(3)
    assert h.engine.phase.value == "synthesis"
    assert not [r for r in h.claude.requests if r.tag.purpose == "research"]
    assert not questions_posted(h)
    await h.engine.close()  # the synthesis still runs
    assert [r for r in h.claude.requests if r.tag.purpose == "synthesis"]
