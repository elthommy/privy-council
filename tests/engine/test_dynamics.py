"""Unit tests for the engine-counted debate dynamics (pure functions) and their prompt text."""

from __future__ import annotations

import pytest

from privy_council.backends.base import CompletionResult
from privy_council.engine import prompts
from privy_council.engine.dynamics import (
    MemberStats,
    marks_update,
    member_stats,
    overlap,
    silent_members,
    stalled,
)
from privy_council.engine.moderation import DecisionError, moderator_schema, parse_decision
from privy_council.models.persona import Persona
from privy_council.models.session import Consensus, Message
from tests.engine.test_protocol import decision
from tests.engine.test_protocol import personas as personas


def msg(speaker: str, text: str, round_: int) -> Message:
    """Build a chat message."""
    return Message(speaker=speaker, text=text, round=round_)


def stats(slug: str, **kw: object) -> MemberStats:
    """Build member stats with neutral defaults."""
    values: dict[str, object] = {"turns": 1, "silent_rounds": 0, "updated": False, "repeat": 0.0}
    values.update(kw)
    return MemberStats(slug=slug, **values)  # type: ignore[arg-type]


def test_overlap() -> None:
    text = "keep three months of expenses in cash before investing anything"
    assert overlap(text, text) == 1.0
    assert overlap(text, "a completely different sentence about other things") == 0.0
    assert overlap("too short", "too short") == 0.0  # fewer words than one n-gram
    assert 0 < overlap(text, text + " in the PEA wrapper this year") < 1


def test_member_stats() -> None:
    messages = [
        msg("chair", "Opening", 0),
        msg("rigor", "Fund first because liquidity matters most for the user", 1),
        msg("compounder", "PEA first: compounding beats cash over the long run", 1),
        msg("rigor", "Fund first because liquidity matters most for the user, again", 2),
        msg("pragmatist", "I update: three months, then the PEA", 2),
        msg("system", "Next round", 2),
    ]
    by_slug = {s.slug: s for s in member_stats(messages, ["rigor", "compounder", "pragmatist"], 2)}
    rigor, compounder, pragmatist = by_slug["rigor"], by_slug["compounder"], by_slug["pragmatist"]
    assert (rigor.turns, rigor.silent_rounds, rigor.updated) == (2, 0, False)
    assert rigor.repeat > 0.7
    assert (compounder.turns, compounder.silent_rounds, compounder.repeat) == (1, 1, 0.0)
    assert pragmatist.updated and pragmatist.silent_rounds == 0
    never = member_stats(messages, ["contrarian"], 2)[0]
    assert (never.turns, never.silent_rounds) == (0, 2)


def test_stalled() -> None:
    quiet = [stats("rigor"), stats("compounder")]
    assert stalled([], True, quiet)  # nothing new
    assert stalled(["@Rigor: a detail"], False, quiet)  # new, but the decision is settled
    assert not stalled(["@Rigor: a layoff cuts the credit line"], True, quiet)
    assert not stalled(None, True, quiet)  # not judged
    assert not stalled([], True, [stats("rigor"), stats("compounder", updated=True)])
    assert stalled([], False, [stats("rigor"), stats("compounder", updated=True)])  # converging


@pytest.mark.parametrize(
    "text",
    [
        "I update: **25-30%** (from my prior 25-35%).",
        "Some context first.\n\nI update: about 8% (range 4-14%).",
        "I update: I lower my confidence to 70 for the violence specifically.",
        "I update: 25-35% stands, but the conditional is now 50-55%, not 42%.",
        "Je change d'avis : trois mois suffisent.",
    ],
)
def test_real_updates_count(text: str) -> None:
    assert marks_update(text)


# The false markers seen in live checks and replays (2026-10-18), plus French forms.
@pytest.mark.parametrize(
    "text",
    [
        "The framing gap is where the battle lives.\n\nI update: **no change**. My position stays.",
        "I update: no change to my 25-35% range.",
        "I update: my position is unchanged. The evidence points to cost of living.",
        "I update: if Russia drops the Donbas precondition, I would move up.",
        "I update: none.",
        "I update: I still think the Livret A wins.",
        "Je change d'avis : non, le Livret A reste le meilleur choix.",
        "Je change d'avis : aucun changement.",
        "Je change d'avis : je maintiens ma position.",
        "No marker here at all.",
    ],
)
def test_denied_or_missing_updates_dont_count(text: str) -> None:
    assert not marks_update(text)


def test_one_real_marker_is_enough() -> None:
    assert marks_update("I update: no change on the fund. I update: the PEA now comes first.")


def test_moderator_confirms_or_vetoes_markers() -> None:
    marked = [stats("rigor"), stats("compounder", updated=True)]
    assert not stalled([], True, marked, None)  # not judged: the marker counts
    assert not stalled([], True, marked, ["compounder"])  # confirmed
    assert stalled([], True, marked, [])  # vetoed: a false marker
    assert stalled([], True, [stats("rigor"), stats("compounder")], ["rigor"])  # unmarked


def test_silent_members() -> None:
    record = [stats("rigor", silent_rounds=2), stats("compounder", silent_rounds=1)]
    assert silent_members(record, planned=[], limit=2) == ["rigor"]
    assert silent_members(record, planned=["rigor"], limit=2) == []
    assert silent_members(record, planned=[], limit=1) == ["rigor", "compounder"]
    assert silent_members(record, planned=[], limit=0) == []


def test_render_dynamics() -> None:
    record = [
        stats("rigor", turns=3, updated=True),
        stats("compounder", turns=2, silent_rounds=2, repeat=0.8),
        stats("pragmatist", repeat=0.2),
    ]
    text = prompts.render_dynamics(record, str.capitalize, stalled=1, stall_limit=2)
    assert text.startswith("# Debate dynamics")
    assert "- @Rigor: 3 turn(s); spoke this round; said they changed position" in text
    assert "- @Compounder: 2 turn(s); quiet for 2 round(s); largely repeated" in text
    assert "(80% overlap)" in text
    assert "@Pragmatist: 1 turn(s); spoke this round\n" in text  # 20%: not flagged
    assert "changes the decision): 1. The engine ends the debate after 2" in text
    off = prompts.render_dynamics(record, str.capitalize, stalled=1, stall_limit=0)
    assert "made no progress" not in off


def test_summon_rule(personas: dict[str, Persona]) -> None:
    assert prompts.summon_rule([]) == prompts.SUMMON_RULE_OFF
    rule = prompts.summon_rule([personas["optimist"]])
    assert "optimist (@Optimist: advisor" in rule
    assert "The user decides" in rule


def test_schema_requires_the_new_fields() -> None:
    schema = moderator_schema()
    required = set(schema["required"])
    assert {"consensus", "new_arguments", "decision_open", "position_changes"} <= required
    assert schema["properties"]["position_changes"]["type"] == "array"
    assert schema["properties"]["new_arguments"]["type"] == "array"
    assert "summon" not in schema["required"]


def structured(raw: dict[str, object]) -> CompletionResult:
    """Wrap a decision as a schema-constrained reply."""
    return CompletionResult(text="", structured=raw, latency_s=0, backend="fake", model="fake")


def test_parse_new_fields(personas: dict[str, Persona]) -> None:
    members = {s: personas[s] for s in ("rigor", "compounder")}
    raw = {
        **decision([("rigor", "go")]),
        "consensus": "partial",
        "new_arguments": [
            "@Rigor: the card line can be cut",
            " ",
            "@Rigor: the card line can be cut",
        ],
        "decision_open": False,
        "position_changes": ["@Rigor", "Compounder", "nobody", "rigor"],
        "summon": {"persona": "optimist", "reason": "upside cases"},
    }
    parsed = parse_decision(structured(raw), members)
    assert parsed.consensus is Consensus.PARTIAL
    assert parsed.new_arguments == ["@Rigor: the card line can be cut"]  # cleaned
    assert parsed.decision_open is False
    assert parsed.position_changes == ["rigor", "compounder"]  # slugs, unknown and repeat dropped
    without = {k: v for k, v in raw.items() if k != "position_changes"}
    assert parse_decision(structured(without), members).position_changes is None
    assert parsed.summon is not None and parsed.summon.persona == "optimist"
    with pytest.raises(DecisionError, match="consensus"):
        parse_decision(structured({**raw, "consensus": "maybe"}), members)
