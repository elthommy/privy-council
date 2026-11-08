from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import CompletionResult, Usage
from privy_council.config import MemoryConfig
from privy_council.engine import context
from privy_council.loaders.personas import load_personas
from privy_council.memory.distill import (
    MemoryOutputError,
    MemoryWriter,
    NoteChange,
    ProposedFact,
    SessionDigest,
    _apply_changes,
    fresh_notes,
    parse_check,
    parse_consolidated,
    parse_notes,
    parse_selection,
    parse_shared,
    quoted_facts,
)
from privy_council.memory.store import MemoryStore
from privy_council.models.persona import Persona
from privy_council.models.session import Message
from tests.conftest import REPO_ROOT


def result(text: str = "", structured: Any = None, truncated: bool = False) -> CompletionResult:
    """Build a CompletionResult for parser tests."""
    return CompletionResult(
        text=text,
        structured=structured,
        usage=Usage(),
        latency_s=0.0,
        backend="fake",
        model="m",
        truncated=truncated,
    )


def test_parse_notes() -> None:
    assert parse_notes("Here you go:\n- one\n- two\n- three\n- four") == ["one", "two", "three"]
    assert parse_notes("NONE") == []
    assert parse_notes("none.") == []
    assert parse_notes("") == []
    assert parse_notes("\nJust a sentence.\nAnd more.") == ["Just a sentence."]
    long = parse_notes("- " + "word " * 100)[0]
    assert len(long) <= 300 and long.endswith("word…")


def test_parse_shared_structured_text_and_errors() -> None:
    facts = [{"fact": "The user rents.", "quote": "I rent"}]
    assert parse_shared(result(structured={"facts": facts}))[0].fact == "The user rents."
    fenced = '```json\n{"facts": [{"fact": "F", "quote": "Q"}]}\n```'
    assert parse_shared(result(text=fenced))[0].quote == "Q"
    with pytest.raises(MemoryOutputError, match="not valid JSON"):
        parse_shared(result(text="sure! here are facts"))
    with pytest.raises(MemoryOutputError, match="wrong shape"):
        parse_shared(result(structured={"facts": [{"fact": "no quote"}]}))


def test_quote_guard_drops_facts_the_user_never_wrote() -> None:
    sources = ["Should I buy?", "I\u2019m 34 and I  rent in Lyon."]  # typographic apostrophe
    facts = [
        ProposedFact(fact="The user is 34.", quote="I'm 34"),  # typographic quote matches
        ProposedFact(fact="The user rents in Lyon.", quote="i rent in lyon"),
        ProposedFact(fact="The user is risk-averse.", quote="I hate risk"),  # never said
        ProposedFact(fact="The user is in Lyon.", quote="in"),  # too short to prove anything
        ProposedFact(fact="The user is 34.", quote="I'm 34"),  # duplicate
    ]
    kept, unquoted = quoted_facts(facts, sources, known="")
    assert kept == ["The user is 34.", "The user rents in Lyon."]
    assert unquoted == 2


def test_quote_guard_skips_facts_already_on_file() -> None:
    known = "- 2026-01-02 (finance): The user is 34.\n"
    facts = [ProposedFact(fact="The user is 34.", quote="I'm 34")]
    assert quoted_facts(facts, ["I'm 34"], known) == ([], 0)


def test_parse_consolidated() -> None:
    merged = "- 2026-10-17 (f): a\n- b"
    assert parse_consolidated("Sure:\n- 2026-10-17 (f): a\n- b", 100) == (merged, "")
    assert parse_consolidated("no bullets here", 100) == (None, "it had no bullet lines")
    over = parse_consolidated("- " + "x" * 200, 100)
    assert over == (None, "it was 202 bytes, over the 100-byte limit")


def test_parse_consolidated_keeps_the_council_tag() -> None:
    ok = "- 2026-10-17 (finance): a\n- 2026-10-18 (finance): b"
    assert parse_consolidated(ok, 100, "finance") == (ok, "")
    for bad in ("- 2026-10-17 (finance): a\n- b", "- 2026-10-17 (career): a"):
        merged, why = parse_consolidated(bad, 100, "finance")
        assert merged is None and "(finance)" in why


def test_parse_selection() -> None:
    assert parse_selection("1 3", 3) == [0, 2]
    assert parse_selection("3,1, 1", 3) == [0, 2]
    assert parse_selection("ALL", 2) == [0, 1]
    assert parse_selection("none", 2) == []
    assert parse_selection("4", 3) is None
    assert parse_selection("0", 3) is None
    assert parse_selection("", 3) is None
    assert parse_selection("some", 3) is None


class Recorder:
    """A `Complete` returning scripted texts and recording prompts."""

    def __init__(self, *replies: CompletionResult) -> None:
        """Queue the replies."""
        self.replies = list(replies)
        self.calls: list[tuple[str, str, context.Prompt, dict[str, Any] | None]] = []

    async def __call__(
        self,
        persona: Persona,
        prompt: context.Prompt,
        purpose: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_output: int | None = None,
    ) -> CompletionResult:
        """Record the call and return the next reply."""
        self.calls.append((persona.slug, purpose, prompt, json_schema))
        return self.replies.pop(0)


@pytest.fixture
def personas() -> dict[str, Persona]:
    """The repo's example personas."""
    return load_personas(REPO_ROOT / "personas")[0]


def digest(messages: list[Message]) -> SessionDigest:
    """A session digest with a fixed date."""
    return SessionDigest(
        council="finance",
        topic="Rent or buy?",
        day=date(2026, 10, 17),
        synthesis="## Consensus\nRent for now.",
        messages=messages,
        state_text=None,
        nick_of=str.capitalize,
    )


def writer(tmp_path: Path, rec: Recorder, **cfg: Any) -> MemoryWriter:
    """A writer over tmp_path with the recorder as the model call."""
    return MemoryWriter(MemoryStore(tmp_path), MemoryConfig(**cfg), rec, lambda _p: 4000)


async def test_write_notes_sees_own_and_user_lines_only(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rec = Recorder(result("- The user chose to rent.\n- I argued for buying."))
    w = writer(tmp_path, rec)
    w.store.append_private("rigor", ["- 2026-01-01 (finance): older note"])
    messages = [
        Message(speaker="rigor", text="Buy, the numbers say so."),
        Message(speaker="compounder", text="Index funds instead."),
        Message(speaker="user", text="I'll rent."),
    ]
    assert await w.write_notes(personas["rigor"], digest(messages)) == 2
    _, purpose, prompt, _ = rec.calls[0]
    assert purpose == "memory"
    assert "Buy, the numbers" in prompt.user and "I'll rent." in prompt.user
    assert "Index funds" not in prompt.user
    assert "older note" in prompt.user  # its notes, so it doesn't repeat them
    assert "Rent for now." in prompt.user
    assert w.store.read_private("rigor").splitlines()[1:] == [
        "- 2026-10-17 (finance): The user chose to rent.",
        "- 2026-10-17 (finance): I argued for buying.",
    ]


async def test_propose_facts_applies_the_quote_guard(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    facts = [
        {"fact": "The user rents.", "quote": "I rent"},
        {"fact": "The user is cautious.", "quote": "I am cautious"},
    ]
    rec = Recorder(result(structured={"facts": facts}))
    w = writer(tmp_path, rec)
    messages = [
        Message(speaker="user", text="I rent a flat."),
        Message(speaker="rigor", text="I am cautious about that."),  # a member, not the user
    ]
    kept, unquoted = await w.propose_facts(personas["chair"], digest(messages))
    assert kept == ["- 2026-10-17 (finance): The user rents."]
    assert unquoted == 1
    _, purpose, prompt, schema = rec.calls[0]
    assert purpose == "shared_memory" and schema is not None
    assert "I rent a flat." in prompt.user and "I am cautious" not in prompt.user


def finance_notes(count: int) -> list[str]:
    """Return `count` finance notes of about 75 bytes each."""
    return [f"- 2026-10-0{i} (finance): note {i} " + "x" * 50 for i in range(1, count + 1)]


async def test_consolidate_rewrites_or_keeps(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rigor = personas["rigor"]
    rec = Recorder(
        result("- 2026-10-17 (finance): merged"),
        result("- " + "y" * 500),  # still over the cap
        result("- fine", truncated=True),  # the retry, cut off
    )
    w = writer(tmp_path, rec, private_cap_bytes=200)
    w.store.append_private("rigor", finance_notes(5))
    assert w.private_over_cap(rigor, "finance")
    assert await w.consolidate(rigor, "finance") is True
    assert w.store.read_private("rigor") == "- 2026-10-17 (finance): merged\n"
    w.store.append_private("rigor", finance_notes(5))
    assert await w.consolidate(rigor, "finance") is False
    assert "merged" in w.store.read_private("rigor")  # left as it was
    assert [c[1] for c in rec.calls] == ["consolidate"] * 3
    assert "under 140 characters" in rec.calls[0][2].user
    assert "from the finance council" in rec.calls[0][2].user


async def test_consolidate_retries_once_lower_with_the_reason(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rec = Recorder(result("- " + "y" * 500), result("- 2026-10-17 (finance): short"))
    w = writer(tmp_path, rec, private_cap_bytes=200)
    w.store.append_private("rigor", finance_notes(5))
    assert await w.consolidate(personas["rigor"], "finance") is True
    retry = rec.calls[1][2].user
    assert "unusable: it was 502 bytes, over the 200-byte limit" in retry
    assert retry.endswith("in under 100 characters in total.")
    assert w.store.read_private("rigor") == "- 2026-10-17 (finance): short\n"


async def test_cap_and_merge_are_per_council(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rigor = personas["rigor"]
    rec = Recorder(result("- 2026-10-17 (finance): merged"))
    w = writer(tmp_path, rec, private_cap_bytes=200)
    career = ["- 2026-10-01 (career): career note " + "z" * 50 for _ in range(2)]
    w.store.append_private("rigor", [career[0], "- my own line", *finance_notes(2), career[1]])
    # 2 career notes + 2 finance notes: each council is under the cap, the file isn't.
    assert not w.private_over_cap(rigor, "finance")
    assert not w.private_over_cap(rigor, "career")
    w.store.append_private("rigor", finance_notes(3))
    assert w.private_over_cap(rigor, "finance")
    assert await w.consolidate(rigor, "finance") is True
    assert "career note" not in rec.calls[0][2].user and "my own line" not in rec.calls[0][2].user
    assert w.store.read_private("rigor").splitlines() == [
        career[0],
        "- my own line",
        career[1],
        "- 2026-10-17 (finance): merged",
    ]


OLD_LESSON = (
    "- 2026-10-01 (career): Next time, confirm whether the user's notes record the budget "
    "as agreed or only discussed."
)


def test_a_lesson_copied_with_a_tail_repeats_the_old_one() -> None:
    new = (
        "Next session, confirm whether the user's notes record the budget as agreed or only "
        "discussed, and ask their monthly rent."
    )
    assert fresh_notes([new], OLD_LESSON) == []


def test_a_note_that_changes_a_figure_or_the_subject_is_new() -> None:
    old = "- 2026-10-01 (finance): The user has a 5k emergency fund."
    assert fresh_notes(["The user has a 6k emergency fund."], old) == [
        "The user has a 6k emergency fund."
    ]
    assert fresh_notes(["I argued for index funds; Skeptic disagreed."], OLD_LESSON) == [
        "I argued for index funds; Skeptic disagreed."
    ]


def test_two_new_notes_that_say_the_same_keep_the_first() -> None:
    notes = ["The user prefers renting over buying.", "The user prefers renting to buying."]
    assert fresh_notes(notes, "") == notes[:1]


def test_note_changes_rewrite_and_never_drop() -> None:
    drafts = {"rigor": ["a", "b"], "optimist": ["c"]}
    numbered = [("rigor", 0), ("rigor", 1), ("optimist", 0)]
    changes = [
        NoteChange(n=1, note="- a fixed"),
        NoteChange(n=3, note=""),  # a local Chair emptied correct notes: never a drop
        NoteChange(n=2, note="b"),  # unchanged: not counted
        NoteChange(n=9, note="out of range"),
    ]
    revised, fixed = _apply_changes(drafts, numbered, changes)
    assert revised == {"rigor": ["a fixed", "b"], "optimist": ["c"]}
    assert fixed == 1
    assert drafts["rigor"] == ["a", "b"]  # the input is left alone


def test_parse_check_reads_text_or_structured_and_rejects_garbage() -> None:
    text = result('```json\n{"changes": [{"n": 1, "note": "x"}]}\n```')
    assert parse_check(text) == [NoteChange(n=1, note="x")]
    assert parse_check(result(structured={"changes": []})) == []
    with pytest.raises(MemoryOutputError):
        parse_check(result('{"changes": [{"n": "one"}]}'))
