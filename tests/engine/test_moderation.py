from __future__ import annotations

from typing import Any

import pytest

from privy_council.backends.base import (
    BackendRateLimitError,
    BackendTimeoutError,
    ChatMessage,
    CompletionResult,
)
from privy_council.engine.moderation import (
    MAX_USER_FACTS,
    DecisionError,
    asked_before,
    cap_questions,
    decide,
    drop_answered,
    drop_asked,
    drop_quoted_facts,
    guest_question,
    keep_user_facts,
    moderator_schema,
    parse_decision,
    resolve_speaker,
    round_robin,
)
from privy_council.loaders.personas import load_personas
from privy_council.models.persona import Persona
from privy_council.models.session import DebateState
from tests.conftest import REPO_ROOT

VALID: dict[str, Any] = {
    "debate_state": {"positions": {"Rigor": "wait"}},
    "next_turns": [
        {"speaker": "@rigor", "prompt": "p1"},
        {"speaker": "Rigor", "prompt": "dup"},
        {"speaker": "ghost", "prompt": "p2"},
        {"speaker": "Contrarian", "prompt": "p3"},
    ],
}


@pytest.fixture
def members() -> dict[str, Persona]:
    """Finance members from the repo examples."""
    loaded, _ = load_personas(REPO_ROOT / "personas")
    return {s: loaded[s] for s in ["rigor", "compounder", "pragmatist", "contrarian"]}


def result(text: str = "", structured: dict[str, Any] | None = None, **kw: Any) -> CompletionResult:
    """Build a CompletionResult."""
    return CompletionResult(
        text=text, structured=structured, latency_s=0, backend="f", model="m", **kw
    )


def test_resolve_speaker(members: dict[str, Persona]) -> None:
    assert resolve_speaker("@Rigor", members) == "rigor"
    assert resolve_speaker(" COMPOUNDER ", members) == "compounder"
    assert resolve_speaker("chair", members) is None


def test_parse_normalizes_and_dedupes(members: dict[str, Persona]) -> None:
    decision = parse_decision(result(structured=VALID), members)
    assert [(t.speaker, t.prompt) for t in decision.next_turns] == [
        ("rigor", "p1"),
        ("contrarian", "p3"),
    ]
    assert decision.debate_state.positions == {"rigor": "wait"}


def test_parse_text_with_code_fence(members: dict[str, Persona]) -> None:
    text = '```json\n{"debate_state": {}, "stop": true, "stop_reason": "done"}\n```'
    assert parse_decision(result(text), members).stop


@pytest.mark.parametrize(
    ("res", "match"),
    [
        (result("nope"), "not valid JSON"),
        (result("[1]"), "expected a JSON object"),
        (result("{", truncated=True), "cut off"),
        (
            result(
                structured={"debate_state": {}, "next_turns": [{"speaker": "x", "prompt": "y"}]}
            ),
            "names no council member",
        ),
    ],
)
def test_parse_errors(res: CompletionResult, match: str, members: dict[str, Persona]) -> None:
    with pytest.raises(DecisionError, match=match):
        parse_decision(res, members)


async def test_decide_backend_error_then_success(members: dict[str, Persona]) -> None:
    replies: list[CompletionResult | Exception] = [
        BackendTimeoutError("slow"),
        result(structured=VALID),
    ]
    purposes: list[str] = []

    async def call(messages: list[ChatMessage], purpose: str) -> CompletionResult:
        """Pop the next scripted reply."""
        purposes.append(purpose)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    decision, problems = await decide(call, "req", members)
    assert decision is not None
    assert purposes == ["moderate", "repair"]
    assert problems == ["BackendTimeoutError: slow"]


async def test_decide_propagates_rate_limit(members: dict[str, Persona]) -> None:
    async def call(messages: list[ChatMessage], purpose: str) -> CompletionResult:
        """Always rate-limited."""
        raise BackendRateLimitError("limit")

    with pytest.raises(BackendRateLimitError):
        await decide(call, "req", members)


def test_round_robin() -> None:
    turns = round_robin(["a", "b"])
    assert [t.speaker for t in turns] == ["a", "b"]
    assert all(t.prompt for t in turns)


def test_moderator_schema_always_requires_a_turn() -> None:
    schema = moderator_schema()
    assert {"debate_state", "next_turns", "stop"} <= set(schema["required"])
    assert schema["properties"]["next_turns"]["minItems"] == 1


def test_questions_for_user_are_optional_and_cleaned(members: dict[str, Persona]) -> None:
    assert parse_decision(result(structured=VALID), members).questions_for_user == []
    asked = {
        **VALID,
        "questions_for_user": ["  How stable  is your income? ", "", "A?", "A?", "B?"],
    }
    decision = parse_decision(result(structured=asked), members)
    assert decision.questions_for_user == ["How stable is your income?", "A?", "B?"]
    schema = moderator_schema()
    assert schema["properties"]["questions_for_user"]["maxItems"] == 2
    assert "questions_for_user" not in schema["required"]
    assert moderator_schema(3)["properties"]["questions_for_user"]["maxItems"] == 6


def test_questions_are_capped_per_person() -> None:
    one = cap_questions(["A?", "@Thomas B?", "C?"], "Thomas", [])
    assert one == ["A?", "@Thomas B?"]  # alone: 2 in all, as before
    asked = ["@Thomas A?", "@Ana B?", "@thomas C?", "D?", "@Ana E?", "@Ana F?", "@Ben G?"]
    assert cap_questions(asked, "Thomas", ["Ana", "Ben"]) == [
        "@Thomas A?",
        "@Ana B?",
        "@thomas C?",
        "@Ana E?",
        "@Ben G?",
    ]


def test_research_requests_are_optional_cleaned_and_capped(members: dict[str, Persona]) -> None:
    assert parse_decision(result(structured=VALID), members).research == []
    asked = {**VALID, "research": ["VIX  close 16 Oct", "VIX close 16 Oct", "a", "b"]}
    assert parse_decision(result(structured=asked), members).research == ["VIX close 16 Oct", "a"]
    assert moderator_schema()["properties"]["research"]["maxItems"] == 2


def test_user_facts_are_optional_cleaned_and_capped(members: dict[str, Persona]) -> None:
    assert parse_decision(result(structured=VALID), members).debate_state.user_facts == []
    facts = [" shaves  every 4 days", "shaves every 4 days", "", *(f"fact {i}" for i in range(20))]
    stated = {**VALID, "debate_state": {**VALID["debate_state"], "user_facts": facts}}
    kept = parse_decision(result(structured=stated), members).debate_state.user_facts
    assert kept[0] == "shaves every 4 days" and len(kept) == MAX_USER_FACTS
    schema = moderator_schema()["$defs"]["DebateState"]["properties"]["user_facts"]
    assert schema["maxItems"] == MAX_USER_FACTS


def test_keep_user_facts() -> None:
    before = DebateState(user_facts=["shaves every 4 days"])
    assert keep_user_facts(before, DebateState()).user_facts == ["shaves every 4 days"]
    corrected = DebateState(user_facts=["shaves every 3 days"])
    assert keep_user_facts(before, corrected).user_facts == ["shaves every 3 days"]
    assert keep_user_facts(DebateState(), DebateState()).user_facts == []


def test_research_copied_into_user_facts_is_dropped() -> None:
    research = (
        "- Locking in a specific featured Resonator costs at most 160 pulls, or 25,600 "
        "Astrite: https://shattered.io/pity\n- Lucilla\u2019s banner rerun runs to November 11"
    )
    before = DebateState(user_facts=["Thomas has 40 pity count"])
    copied = "Locking in a specific featured Resonator costs at most 160 pulls, or 25,600 Astrite"
    new = DebateState(
        user_facts=["Thomas has 40 pity count", "Thomas cannot afford 120 pulls", copied]
    )
    kept = drop_quoted_facts(before, new, research).user_facts
    assert kept == ["Thomas has 40 pity count", "Thomas cannot afford 120 pulls"]


def test_a_short_or_earlier_fact_found_in_the_research_stays() -> None:
    research = "- Lucilla's banner rerun runs to November 11, 2026, with 160 pulls to lock her"
    earlier = "Lucilla's banner rerun runs to November 11, 2026"
    before = DebateState(user_facts=[earlier])
    new = DebateState(user_facts=[earlier, "160 pulls to lock her"])
    assert drop_quoted_facts(before, new, research) == new


def test_questions_copied_back_to_someone_who_spoke_are_dropped() -> None:
    previous = DebateState(
        open_questions=["@Thomas how much can you put in?", "@Ana do you need it daily?"]
    )
    new = DebateState(
        open_questions=[
            "@Thomas how much can you put in?",  # copied, and Thomas has spoken: answered
            "@Ana do you need it daily?",  # Ana hasn't spoken
            "@Thomas would you sign an agreement?",  # new: still open
        ]
    )
    kept = drop_answered(previous, new, {"thomas"}).open_questions
    assert kept == ["@Ana do you need it daily?", "@Thomas would you sign an agreement?"]


def test_a_guest_question_is_forwarded_when_no_guest_is_asked() -> None:
    open_questions = ["@Thomas A?", "@Ana B?", "@Ben C?"]
    guests = ["Ana", "Ben"]
    assert guest_question(["@Thomas A?"], open_questions, guests, []) == ["@Ana B?"]
    assert guest_question(["@Ben C?"], open_questions, guests, []) == []  # a guest is asked
    assert guest_question([], open_questions, guests, ["@Ana B?"]) == ["@Ben C?"]
    assert guest_question([], ["@Thomas A?"], guests, []) == []  # no guest question


def test_a_question_asked_before_is_dropped_even_with_a_colon_added() -> None:
    asked = ["@Thomas do you want Suoming for your teams or just for collecting?"]
    again = "@Thomas: do you want Suoming for your teams or just for collecting?"
    assert drop_asked([again, "@Thomas how many pulls do you have?"], asked, "Thomas") == [
        "@Thomas how many pulls do you have?"
    ]


def test_the_same_question_to_another_person_is_kept() -> None:
    asked = ["@Thomas do you want her for your teams or for collecting?"]
    assert drop_asked(["@Ana do you want her for your teams or for collecting?"], asked, "Thomas")


def test_a_question_naming_nobody_goes_to_the_owner() -> None:
    asked = ["@Thomas how much do you earn per patch?"]
    assert asked_before("How much do you earn per patch?", asked, "Thomas")
    assert not asked_before("How much do you earn per patch?", asked, "Ana")


def test_a_question_that_only_shares_a_few_words_is_new() -> None:
    asked = ["@Thomas can you afford the extra pulls for hard pity now?"]
    assert not asked_before("@Thomas can you afford a second copy now?", asked, "Thomas")
