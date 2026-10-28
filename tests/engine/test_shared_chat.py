"""Several people in one chat (M16 part A): authors, prompts, questions, memory."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from privy_council.engine import prompts
from privy_council.memory.distill import SessionDigest
from privy_council.models.council import MemoryMode
from privy_council.models.persona import Persona
from privy_council.models.session import Message
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed

PEOPLE = "Several people share this chat: User, Ana, Ben."


def asking(*questions: str) -> dict[str, Any]:
    """A moderator decision that forwards questions to the people."""
    return {**decision([("rigor", "go")]), "questions_for_user": list(questions)}


def systems(h: Harness, purpose: str) -> list[str]:
    """Return the system prompts of the requests with this purpose, in order."""
    return [r.system for r in h.requests() if r.tag.purpose == purpose]


def users(h: Harness, purpose: str) -> list[str]:
    """Return the first user message of the requests with this purpose, in order."""
    return [r.messages[0].content for r in h.requests() if r.tag.purpose == purpose]


async def test_one_person_keeps_the_prompts_word_for_word(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 2))
    await h.engine.step()
    await h.engine.step()
    h.engine.say("My income is variable")  # the owner: no author
    await h.engine.step()
    assert h.engine.guests == []
    member = prompts.member_protocol(personas["rigor"], personas["chair"], "User")
    turn = systems(h, "turn")[-1]
    assert f"{member}\n\nWrite your reply in English." in turn  # nothing added between
    assert all("Several people" not in s for s in systems(h, "moderate"))
    assert all("The people wrote" not in u for u in users(h, "moderate"))
    assert all("at most 2 questions where" in u for u in users(h, "moderate"))


async def test_guests_reach_every_prompt_by_name(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 2))
    await h.engine.step()
    await h.engine.step()
    h.engine.say("I'd rather keep cash", author="Ana")
    h.engine.say("Same here", author="Ben")
    h.engine.say("Me too", author="Ana")  # listed once
    h.engine.say("Noted", author="User")  # the owner's own name is not a guest
    await h.engine.step()
    assert h.engine.guests == ["Ana", "Ben"]
    turn = systems(h, "turn")[-1]
    assert PEOPLE in turn and "User asked the question" in turn
    moderate = systems(h, "moderate")[-1]
    assert PEOPLE in moderate and '("User: ...")' in moderate
    assert '("@Ana how much can you put in?"), never to User on their behalf' in moderate
    assert "when unsure" not in moderate
    assert "drop it from open_questions" in moderate
    assert 'say "the user", they mean each of these people' in moderate
    assert "at most 2 questions per person, each to one person by @name" in users(h, "moderate")[-1]
    assert "changes their mind corrects" in moderate
    assert '("Ana: wants ...")' in moderate and "member slugs only" in moderate
    rigor = [r for r in h.ollama.requests if r.tag.persona == "rigor"][-1]
    assert "<Ana> I'd rather keep cash" in rigor.messages[0].content
    assert "<Ben> Same here" in rigor.messages[0].content


async def test_questions_keep_the_person_the_moderator_named(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        asking("@Ana How much cash do you hold?", "Is your job stable?"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    h.engine.say("Hi all", author="Ana")
    await h.engine.step()
    await h.engine.step()
    asked = [m.text for m in h.engine.messages if m.to_user]
    assert asked == ["@Ana How much cash do you hold?\n@User Is your job stable?"]


async def test_one_person_questions_always_go_to_the_owner(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [asking("@Ana what's your view?")]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()
    assert [m.text for m in h.engine.messages if m.to_user] == ["@User @Ana what's your view?"]


async def test_ask_and_transcript_show_the_author(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    await h.engine.step()
    await h.engine.ask("rigor", "why cash?", author="Ana")
    asked = next(m for m in h.engine.messages if m.speaker == "user")
    assert asked.author == "Ana"
    md = (h.store.directory / "transcript.md").read_text()
    assert "<Ana>**" in md


async def test_resume_keeps_the_guests(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()
    h.engine.say("Cash for me", author="Ana")
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    assert r.engine.guests == ["Ana"]
    assert next(m for m in r.engine.messages if m.speaker == "user").author == "Ana"


def test_shared_memory_hears_only_the_owner() -> None:
    messages = [
        Message(speaker="user", text="I rent in Lyon"),
        Message(speaker="user", text="I own in Paris", author="Ana"),
    ]
    digest = SessionDigest(
        council="finance",
        topic="Rent or buy?",
        day=date(2026, 10, 24),
        synthesis="",
        messages=messages,
        state_text=None,
        nick_of=str.title,
    )
    assert digest.user_lines() == ["Rent or buy?", "I rent in Lyon"]


async def test_guest_lines_alone_propose_no_shared_facts(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]), memory=MemoryMode.ON)
    h.engine.say("I own in Paris", author="Ana")
    assert not h.engine._proposes_facts()
    h.engine.say("I rent in Lyon")
    assert h.engine._proposes_facts()


async def test_the_update_lists_what_the_people_said_since_the_last_one(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 3))
    await h.engine.step()
    h.engine.say("I can put in 9,000", author="Ana")
    await h.engine.step()
    h.engine.say("I changed my mind: split by kilometres")
    h.engine.say("Correction: 5,000, not 15,000", author="Ben")
    await h.engine.step()
    first, second = users(h, "moderate")[-2:]
    answered = "every question they answered.\n"
    assert f"{answered}<Ana> I can put in 9,000\n\n" in first
    lines = second.split(answered)[1].split("\n\n")[0]
    assert (
        lines
        == "<User> I changed my mind: split by kilometres\n<Ben> Correction: 5,000, not 15,000"
    )


def questioning(*questions: str) -> dict[str, Any]:
    """A moderator decision whose debate state holds these open questions."""
    out = decision([("rigor", "go")])
    out["debate_state"]["open_questions"] = list(questions)
    return out


async def test_an_answered_question_copied_back_is_dropped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    asked = ["@User can you sign?", "@Ana how often do you drive?"]
    decisions: list[dict[str, Any] | str] = [questioning(*asked)] * 3
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()  # round 1: both questions open
    assert h.engine.meta.runtime.debate_state.open_questions == asked
    h.engine.say("Daily, three days a week", author="Ana")
    await h.engine.step()
    assert h.engine.meta.runtime.debate_state.open_questions == ["@User can you sign?"]


async def test_each_person_gets_up_to_two_questions(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    asked = ["@User A?", "@Ana B?", "@Ben C?", "@Ana D?", "@Ana E?"]
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "go")]), asking(*asked)]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    h.engine.say("Hi", author="Ana")
    h.engine.say("Hello", author="Ben")
    await h.engine.step()
    await h.engine.step()
    forwarded = [m.text for m in h.engine.messages if m.to_user][-1]
    assert forwarded.splitlines() == ["@User A?", "@Ana B?", "@Ben C?", "@Ana D?"]
    schema = [r.json_schema for r in h.requests() if r.tag.purpose == "moderate"][-1]
    assert schema is not None and schema["properties"]["questions_for_user"]["maxItems"] == 6


async def test_a_guest_question_written_down_is_forwarded(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    owner_only = asking("@User can you sign?")
    owner_only["debate_state"]["open_questions"] = [
        "@User can you sign?",
        "@Ana what's your budget?",
    ]
    later = asking("@User when do you start?")
    later["debate_state"]["open_questions"] = owner_only["debate_state"]["open_questions"]
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "go")]), owner_only, later]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    h.engine.say("Hi", author="Ana")
    await h.engine.step()
    await h.engine.step()
    asked = [m.text for m in h.engine.messages if m.to_user]
    assert asked[-1].splitlines() == ["@User can you sign?", "@Ana what's your budget?"]
    await h.engine.step()  # asked before: never forwarded twice
    assert [m.text for m in h.engine.messages if m.to_user][-1] == "@User when do you start?"


async def test_the_people_lines_in_the_task_are_capped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])] * 2))
    await h.engine.step()
    for i in range(200):
        h.engine.say(f"line {i} " + "x" * 100, author="Ana")
    await h.engine.step()
    task = users(h, "moderate")[-1].split("every question they answered.\n")[1]
    assert len(task) < 4100 and "line 199" in task and "line 0 " not in task
