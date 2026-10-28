from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from privy_council.models.moderator import ModeratorDecision

VALID = {
    "debate_state": {
        "positions": {"rigor": "build the fund first"},
        "agreements": ["keep costs low"],
        "disagreements": [{"topic": "fund size", "sides": {"rigor": "6 months"}}],
        "open_questions": ["How stable is your income?"],
    },
    "next_turns": [{"speaker": "rigor", "prompt": "Respond to compounder on fund size."}],
    "stop": False,
    "stop_reason": None,
}


def test_valid_decision_parses_from_json() -> None:
    decision = ModeratorDecision.model_validate_json(json.dumps(VALID))
    assert decision.next_turns[0].speaker == "rigor"
    assert decision.debate_state.disagreements[0].sides == {"rigor": "6 months"}


def test_stop_without_reason_is_accepted() -> None:
    decision = ModeratorDecision.model_validate({**VALID, "stop": True, "next_turns": []})
    assert decision.stop and decision.stop_reason is None


def test_stop_with_reason_needs_no_turns() -> None:
    decision = ModeratorDecision.model_validate(
        {**VALID, "stop": True, "stop_reason": "consensus", "next_turns": []}
    )
    assert decision.stop


def test_continuing_requires_turns() -> None:
    with pytest.raises(ValidationError, match="next_turns must not be empty"):
        ModeratorDecision.model_validate({**VALID, "next_turns": []})


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        ModeratorDecision.model_validate({**VALID, "verdict": "yes"})


def test_json_schema_is_exportable() -> None:
    schema = ModeratorDecision.model_json_schema()
    assert "debate_state" in schema["required"]
