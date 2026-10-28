"""Debate language (M10): detection, `/lang`, the language line and French labels."""

from __future__ import annotations

import json
from pathlib import Path

from privy_council.backends.base import BackendRateLimitError, ChatMessage, CompletionRequest
from privy_council.engine import dynamics, prompts
from privy_council.engine.moderation import round_robin
from privy_council.models.persona import Persona
from privy_council.models.session import Message, SessionMeta
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed
from tests.engine.test_topic import topicless

FRENCH_TOPIC = "Dois-je remplir mon PEA avant de constituer une épargne de précaution ?"
FRENCH = prompts.LANGUAGE_LINES["fr"]
ENGLISH = prompts.LANGUAGE_LINES["en"]


def french(tmp_path: Path, personas: dict[str, Persona], rounds: int = 3) -> Harness:
    """A harness on a French question, with enough moderator decisions."""
    decisions = [decision([("rigor", "réponds à @Compounder")])] * rounds
    return Harness(tmp_path, personas, standard_responder(list(decisions)), topic=FRENCH_TOPIC)


def french_line(req: CompletionRequest) -> str:
    """Return the French line a request should end with (JSON calls keep their keys)."""
    return prompts.language_line("fr", structured=req.json_schema is not None)


def last_user(req: CompletionRequest) -> str:
    """Return the last user message of a request."""
    return [m.content for m in req.messages if m.role == "user"][-1]


def notices(h: Harness) -> list[str]:
    """Return the system lines posted so far."""
    return [m.text for m in h.engine.messages if m.speaker == "system"]


def test_language_is_detected_and_saved(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = french(tmp_path, personas)
    assert h.engine.meta.lang == "fr"
    assert h.store.load_meta().lang == "fr"
    assert Harness(tmp_path / "en", personas, standard_responder([])).engine.meta.lang == "en"


def test_explicit_language_beats_detection(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]), topic=FRENCH_TOPIC, lang="en")
    assert h.engine.meta.lang == "en"


def test_typed_question_is_detected(tmp_path: Path, personas: dict[str, Persona]) -> None:
    engine, _, _ = topicless(tmp_path, personas)
    assert engine.meta.lang is None and engine.meta.language == "en"
    engine.set_topic(FRENCH_TOPIC)
    assert engine.meta.lang == "fr"

    chosen, _, _ = topicless(tmp_path / "chosen", personas)
    chosen.set_language("en")  # before the question: kept, not re-detected
    chosen.set_topic(FRENCH_TOPIC)
    assert chosen.meta.lang == "en"


async def test_every_call_ends_with_the_language_line(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = french(tmp_path, personas)
    await h.engine.step()  # opening
    await h.engine.step()  # blind round
    await h.engine.step()  # cross round
    await h.engine.ask("rigor", "et le livret A ?")
    await h.engine.vote()
    await h.engine.summary()
    await h.engine.close()  # synthesis and memory
    purposes = {r.tag.purpose for r in h.requests()}
    assert {"opening", "turn", "moderate", "ask", "vote", "summary", "synthesis"} <= purposes
    assert {"memory", "shared_memory"} <= purposes
    for req in h.requests():
        assert req.system.endswith(french_line(req)), req.tag
        assert last_user(req).endswith(french_line(req)), req.tag
        assert ENGLISH not in req.system
        # Member turns never mention JSON: gpt-oss then answered in JSON.
        assert ("JSON" in french_line(req)) == (req.json_schema is not None)


async def test_french_labels_and_headings(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = french(tmp_path, personas)
    await h.engine.step()
    await h.engine.step()
    await h.engine.step()
    h.engine.say("Je suis en CDI depuis quatre ans.")  # something for shared memory
    await h.engine.close()
    turns = [r for r in h.requests() if r.tag.purpose == "turn"]
    blind, cross = turns[0], turns[-1]
    assert "**Position :**" in last_user(blind) and "**Hypothèse clé :**" in last_user(blind)
    assert "**Position:**" not in last_user(blind)
    assert '"Je change d\'avis :"' in blind.system and "I update:" not in blind.system
    # The marker sits in the member rules only: at the end of the task, small models
    # opened every cross-round turn with it, even when they kept their position.
    assert "Je change d'avis" in cross.system and "Je change d'avis" not in last_user(cross)
    shared = next(r for r in h.requests() if r.tag.purpose == "shared_memory")
    assert '("L\'utilisateur ...")' in last_user(shared)
    notes = next(r for r in h.requests() if r.tag.purpose == "memory")
    assert 'Refer to the user as "l\'utilisateur"' in last_user(notes)
    [synthesis] = [r for r in h.requests() if r.tag.purpose == "synthesis"]
    assert "## Désaccords restants" in last_user(synthesis)
    assert "## Prochaines étapes suggérées" in last_user(synthesis)


async def test_english_sessions_keep_english_labels(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()
    await h.engine.step()
    blind = next(r for r in h.requests() if r.tag.purpose == "turn")
    assert prompts.task(prompts.BLIND_TASK, "en") in last_user(blind)
    assert '"I update:"' in blind.system
    assert last_user(blind).endswith(ENGLISH) and blind.system.endswith(ENGLISH)


async def test_language_line_doesnt_double_on_fallback(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        standard_responder([]),
        topic=FRENCH_TOPIC,
        fallback=True,
        claude_script=[BackendRateLimitError("limit")],
    )
    await h.engine.step()  # Chair is Claude-backed: the opening falls back
    opening = [r for r in h.ollama.requests if r.tag.purpose == "opening"][-1]
    assert opening.system.count(FRENCH) == 1
    assert last_user(opening).count(FRENCH) == 1


def test_consolidation_keeps_the_notes_language(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = french(tmp_path, personas)
    chat = [ChatMessage(role="user", content="notes")]
    assert h.engine._localized("system", chat, "consolidate") == ("system", chat)
    system, localized = h.engine._localized("system", chat, "memory")
    assert system == f"system\n\n{FRENCH}"
    assert localized[0].content == f"notes\n\n{FRENCH}"


def test_repair_line_goes_on_the_last_user_message(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = french(tmp_path, personas)
    chat = [
        ChatMessage(role="user", content="task"),
        ChatMessage(role="assistant", content="bad json"),
        ChatMessage(role="user", content="repair"),
    ]
    _, localized = h.engine._localized("system", chat, "repair")
    assert [m.content for m in localized] == ["task", "bad json", f"repair\n\n{FRENCH}"]


async def test_lang_command(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = french(tmp_path, personas)
    await h.engine.step()  # opening: announces the language
    assert "Debate language: French. /lang en switches it before round 1" in notices(h)
    h.engine.set_language("")
    assert notices(h)[-1] == "Debate language: French"
    h.engine.set_language("de")
    assert notices(h)[-1] == "usage: /lang en | fr"
    h.engine.set_language("en")
    assert notices(h)[-1] == "Debate language set to English"
    assert h.store.load_meta().lang == "en"
    await h.engine.step()  # blind round: members speak
    assert all(r.system.endswith(ENGLISH) for r in h.requests() if r.tag.purpose == "turn")
    h.engine.set_language("fr")
    assert notices(h)[-1] == "The debate language (English) can't change once members have spoken"
    assert h.engine.meta.lang == "en"


async def test_resume_keeps_the_language(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = french(tmp_path, personas)
    await h.engine.step()
    resumed = Resumed(
        h, h.store.directory, personas, standard_responder([decision([("rigor", "x")])] * 3)
    )
    assert resumed.engine.meta.lang == "fr"
    await resumed.engine.step()
    requests = resumed.claude.requests + resumed.ollama.requests
    assert all(r.system.endswith(french_line(r)) for r in requests)


def test_sessions_before_m10_load_as_english() -> None:
    raw = {"id": "x", "council": "finance", "topic": "Que faire ?", "moderator": "chair"}
    meta = SessionMeta.model_validate_json(json.dumps({**raw, "members": ["rigor"]}))
    assert meta.lang is None and meta.language == "en"


def test_french_update_marker_counts() -> None:
    def said(text: str) -> bool:
        """Return whether a round-2 message with this text counts as an update."""
        messages = [Message(speaker="rigor", text=text, round=2)]
        return dynamics.member_stats(messages, ["rigor"], 2)[0].updated

    assert said("Je change d'avis : trois mois suffisent")
    assert said("**Je change d\u2019avis :** trois mois")
    assert said("je change d'avis: trois mois")
    assert said("I update: three months")
    assert not said("Je ne change pas d'avis : six mois")
    assert not said("Si je change d'avis, je le dirai")


def test_engine_written_floor_prompts_are_french() -> None:
    [turn] = round_robin(["rigor"], "fr")
    assert turn.prompt == prompts.FALLBACK_TURN_PROMPT["fr"]
    for table in (prompts.FALLBACK_TURN_PROMPT, prompts.SILENT_TURN_PROMPT, prompts.JOIN_PROMPT):
        assert set(table) == {"en", "fr"}


def test_french_line_pins_the_form_of_address() -> None:
    assert "« tu », never « vous »" in FRENCH


def test_label_tables_match() -> None:
    assert prompts.LABELS["en"].keys() == prompts.LABELS["fr"].keys()
    assert set(prompts.LANGUAGE_LINES) == {"en", "fr"}
