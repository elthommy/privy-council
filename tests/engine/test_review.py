"""`/vote` and `/summary`: taking stock without moving the debate on."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import (
    BackendOutputError,
    CompletionRequest,
    CompletionResult,
    Usage,
)
from privy_council.backends.fake import Reply
from privy_council.engine import prompts
from privy_council.engine.review_flow import parse_vote
from privy_council.models.persona import Persona
from privy_council.models.session import (
    ErrorEvent,
    MessageEvent,
    Phase,
    TextDelta,
    TurnEnded,
    VoteTaken,
)
from privy_council.models.vote import Vote
from tests.engine.test_protocol import MEMBERS, Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed
from tests.engine.test_roster import notices

SUMMARY = "## Consensus so far\nKeep a buffer."


def reviewing(decisions: list[dict[str, Any] | str], votes: dict[str, Reply] | None = None) -> Any:
    """Answer votes per member (default: a valid vote) and summaries; else as usual."""
    standard = standard_responder(decisions)

    def respond(req: CompletionRequest) -> Reply:
        """Return the scripted reply for this request."""
        persona = req.tag.persona or ""
        if req.tag.purpose == "vote":
            # Double space: the position is normalized to single spaces.
            default = {"position": f"{persona}: fund  first", "confidence": 70}
            return (votes or {}).get(persona, default)
        if req.tag.purpose == "summary":
            return SUMMARY
        return standard(req)

    return respond


async def after_round_one(h: Harness) -> None:
    """Run the opening and the blind round."""
    await h.engine.step()
    await h.engine.step()


def vote_events(h: Harness) -> list[VoteTaken]:
    """Return the VoteTaken events so far."""
    return [e for e in h.events if isinstance(e, VoteTaken)]


async def test_vote_asks_every_member_and_leaves_the_debate_where_it_was(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])]))
    await after_round_one(h)
    rt = h.engine.meta.runtime
    pending, messages = list(rt.pending), len(h.engine.messages)

    await h.engine.vote()
    [event] = vote_events(h)
    assert event.round == 1
    assert [v.speaker for v in event.votes] == MEMBERS  # roster order
    assert event.votes[0] == Vote(speaker="rigor", position="rigor: fund first", confidence=70)
    assert len(h.engine.messages) == messages  # a table, not chat messages
    assert rt.pending == pending and h.phase() is Phase.ROUND
    assert h.store.load_meta().runtime.last_vote == event.votes

    calls = [r for r in h.requests() if r.tag.purpose == "vote"]
    assert {r.tag.persona for r in calls} == set(MEMBERS)
    assert all(r.json_schema is not None for r in calls)
    assert all(prompts.VOTE_TASK in r.messages[0].content for r in calls)
    assert "rigor says something" in calls[0].messages[0].content  # sees the debate so far
    assert [r.tag.persona for r in h.claude.requests if r.tag.purpose == "vote"] == ["compounder"]

    md = h.store.transcript_md.read_text(encoding="utf-8")
    assert "### Vote after round 1" in md
    assert "| Rigor | 70% | rigor: fund first |" in md


async def test_vote_failures_become_rows(tmp_path: Path, personas: dict[str, Persona]) -> None:
    votes: dict[str, Reply] = {
        "rigor": BackendOutputError("boom"),
        "compounder": "I think we should wait.",
        "pragmatist": {"position": "wait", "confidence": 150},
    }
    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])], votes))
    await after_round_one(h)
    await h.engine.vote()
    [event] = vote_events(h)
    rigor, compounder, pragmatist, contrarian = event.votes
    assert rigor.error == "boom"
    assert compounder.error is not None and "not valid JSON" in compounder.error
    assert pragmatist.error is not None and "less than or equal to 100" in pragmatist.error
    assert contrarian.error is None and contrarian.confidence == 70
    assert "| Rigor | — | *no vote: boom* |" in h.store.transcript_md.read_text(encoding="utf-8")
    # A vote posts no message: each turn ends explicitly, the failed one too.
    assert sorted(e.speaker for e in h.events if isinstance(e, TurnEnded)) == sorted(MEMBERS)


def reply(text: str, truncated: bool = False) -> CompletionResult:
    """Return a plain-text result, as a backend without structured output gives."""
    return CompletionResult(
        text=text, usage=Usage(), latency_s=0.1, truncated=truncated, backend="fake", model="m"
    )


def test_parse_vote_reads_text_json_and_flags_fallback() -> None:
    result = reply('```json\n{"position": "wait", "confidence": 40}\n```')
    assert parse_vote("rigor", result, True) == Vote(
        speaker="rigor", position="wait", confidence=40, fallback=True
    )
    cut = reply('{"position": "wa', truncated=True)
    assert "cut off" in (parse_vote("rigor", cut, False).error or "")


async def test_vote_and_summary_refusals(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, reviewing([decision(stop=True, reason="done")]))
    await h.engine.step()  # opening only
    await h.engine.vote()
    await h.engine.summary()
    assert notices(h)[-2:] == [
        "Nothing to vote yet: /step runs round 1 first",
        "Nothing to summarize yet: /step runs round 1 first",
    ]
    await h.engine.step()
    await h.engine.close()
    await h.engine.vote()
    assert notices(h)[-1] == "The session is closed"
    assert not [r for r in h.requests() if r.tag.purpose in ("vote", "summary")]


async def test_synthesis_sees_the_latest_vote(tmp_path: Path, personas: dict[str, Persona]) -> None:
    votes: dict[str, Reply] = {"rigor": BackendOutputError("boom")}
    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])], votes))
    await after_round_one(h)
    await h.engine.vote()
    await h.engine.close()
    [synthesis] = [r for r in h.requests() if r.tag.purpose == "synthesis"]
    prompt = synthesis.messages[0].content
    assert "The user called a vote after round 1" in prompt
    assert "- Compounder (70%): compounder: fund first" in prompt
    assert "Rigor (" not in prompt  # no vote, no line
    assert "You already asked the user" not in prompt  # nothing asked


async def test_synthesis_lists_the_questions_asked(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])]))
    await after_round_one(h)
    h.engine.meta.runtime.asked_questions.append("Teams or collecting?")
    await h.engine.close()
    [synthesis] = [r for r in h.requests() if r.tag.purpose == "synthesis"]
    assert 'You already asked the user during the debate: "Teams or collecting?"' in (
        synthesis.messages[0].content
    )


async def test_summary_posts_a_streamed_chair_message_and_the_debate_goes_on(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        decision([("rigor", "on")]),
    ]
    h = Harness(tmp_path, personas, reviewing(decisions))
    await after_round_one(h)
    pending = list(h.engine.meta.runtime.pending)
    await h.engine.summary()
    assert h.texts("chair")[-1] == SUMMARY
    assert notices(h)[-1] == "Interim summary; the debate continues: /step or /run N"
    assert any(isinstance(e, TextDelta) and e.speaker == "chair" for e in h.events)
    [call] = [r for r in h.requests() if r.tag.purpose == "summary"]
    assert "interim summary after round 1" in call.messages[0].content
    assert call.model == personas["chair"].meta.model  # not synthesis_model
    assert h.engine.meta.runtime.pending == pending and h.phase() is Phase.ROUND
    await h.engine.step()  # round 2 runs as planned
    assert h.texts("rigor")[-1] == "rigor says something in round"


async def test_summary_failure_is_reported(tmp_path: Path, personas: dict[str, Persona]) -> None:
    standard = reviewing([decision([("rigor", "go")])])

    def respond(req: CompletionRequest) -> Reply:
        """Fail the summary only."""
        return BackendOutputError("garbage") if req.tag.purpose == "summary" else standard(req)

    h = Harness(tmp_path, personas, respond)
    await after_round_one(h)
    await h.engine.summary()
    assert isinstance(h.events[-1], ErrorEvent)
    assert h.events[-1].text == "Summary failed (garbage); /summary to retry"
    assert not h.engine.busy and h.phase() is Phase.ROUND


async def test_vote_survives_a_resume(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])]))
    await after_round_one(h)
    await h.engine.vote()
    r = Resumed(h, h.store.directory, personas, reviewing([]))
    assert [e for e in r.engine.history if isinstance(e, VoteTaken)] == vote_events(h)
    assert r.engine.meta.runtime.last_vote == vote_events(h)[0].votes
    assert not any(isinstance(e, MessageEvent) and "vote" in e.message.text for e in r.events)


def test_vote_schema_carries_the_length_rule() -> None:
    from privy_council.engine.review_flow import VOTE_SCHEMA

    assert VOTE_SCHEMA["properties"]["position"]["description"] == prompts.VOTE_POSITION_RULE
    assert "at most 25 words" in prompts.VOTE_TASK


def test_long_positions_are_shortened_for_display_only(tmp_path: Path) -> None:
    from privy_council.storage.sessions import SessionStore
    from privy_council.tui.commands import VOTE_DISPLAY_CHARS, shorten, vote_rows

    assert shorten("short", 10) == "short"
    assert shorten("one two three four", 12) == "one two…"
    long = "word " * 100
    [(_, _, shown)] = vote_rows([Vote(speaker="rigor", position=long, confidence=50)], str)
    assert len(shown) <= VOTE_DISPLAY_CHARS and shown.endswith("…")

    from datetime import UTC, datetime

    store = SessionStore.create(tmp_path, "finance", "t", datetime.now(UTC))
    event = VoteTaken(round=1, votes=[Vote(speaker="rigor", position=long.strip(), confidence=50)])
    store.append(event)
    assert long.strip() in store.transcript_md.read_text(encoding="utf-8")  # kept whole on disk
