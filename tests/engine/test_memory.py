"""Memory through the engine: writing at close, reading in prompts, /keep and /forget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import (
    BackendRateLimitError,
    BackendTimeoutError,
    CompletionRequest,
)
from privy_council.backends.fake import Reply
from privy_council.memory.store import MemoryStore
from privy_council.models.council import MemoryMode
from privy_council.models.persona import Persona
from privy_council.models.session import MemoryProposed, MessageEvent, SessionState
from tests.engine.test_protocol import Harness, Responder, decision, standard_responder
from tests.engine.test_protocol import personas as personas

SPEAKERS = ["chair", "rigor", "compounder", "pragmatist", "contrarian"]
USER_LINE = "I rent a flat in Lyon and I earn 3k a month."
FACTS = [
    {"fact": "The user rents a flat in Lyon.", "quote": "I rent a flat in Lyon"},
    {"fact": "The user earns 3k a month.", "quote": "I earn 3k a month"},
    {"fact": "The user is risk-averse.", "quote": "I avoid risk"},  # never said
]


def memory_responder(**overrides: Reply) -> Responder:
    """The standard debate replies plus memory replies; `overrides` maps purpose -> reply."""
    base = standard_responder([decision([("rigor", "go")]) for _ in range(3)])

    def respond(req: CompletionRequest) -> Reply:
        """Answer memory purposes, else the standard debate reply."""
        purpose = req.tag.purpose
        if purpose in overrides:
            return overrides[purpose]
        if purpose == "memory":
            return f"- {req.tag.persona} argued about the PEA."
        if purpose == "shared_memory":
            return {"facts": FACTS}
        if purpose == "consolidate":
            return "- 2026-10-17 (finance): merged"
        return base(req)

    return respond


def notices(h: Harness) -> list[str]:
    """Return the system lines posted so far."""
    return [m.text for m in h.engine.messages if m.speaker == "system"]


def proposals(h: Harness) -> list[MemoryProposed]:
    """Return the memory proposals emitted so far."""
    return [e for e in h.events if isinstance(e, MemoryProposed)]


async def closed_session(h: Harness, *, say: str | None = None) -> None:
    """Run the opening and the blind round (everyone speaks), a user line, then close."""
    await h.engine.step()
    await h.engine.step()
    if say:
        h.engine.say(say)
    await h.engine.close()


async def test_close_writes_private_notes_for_every_persona(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    await closed_session(h)
    assert h.state() is SessionState.CLOSED
    store = MemoryStore(h.settings.data_dir)
    assert store.personas_with_notes() == sorted(SPEAKERS)
    note = store.read_private("rigor").strip()
    assert note.endswith("(finance): rigor argued about the PEA.") and note.startswith("- 20")
    assert "Rigor saved 1 memory note(s)" in notices(h)
    assert notices(h)[-1] == "Session closed; synthesis saved"
    # The user said nothing: no shared proposal call, nothing to approve.
    assert all(r.tag.purpose != "shared_memory" for r in h.requests())
    assert not proposals(h) and store.read_shared() == ""
    # Local turns are grouped by model: both qwen3.6 personas, then gemma4.
    local = [r.tag.persona for r in h.ollama.requests if r.tag.purpose == "memory"]
    assert local == ["rigor", "pragmatist", "contrarian"]
    claude = {r.tag.persona: r.model for r in h.claude.requests if r.tag.purpose == "memory"}
    assert claude == {"chair": "sonnet", "compounder": "sonnet"}


async def test_second_session_uses_notes_from_the_first_and_reset_works(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    first = Harness(tmp_path, personas, memory_responder())
    await closed_session(first)

    second = Harness(tmp_path, personas, memory_responder())
    await second.engine.step()  # opening
    await second.engine.step()  # blind round
    by_persona = {r.tag.persona: r for r in second.requests() if r.tag.purpose == "turn"}
    rigor_prompt = by_persona["rigor"].messages[0].content
    assert "# Memory from past sessions" in rigor_prompt
    assert "rigor argued about the PEA." in rigor_prompt
    assert "pragmatist argued" not in rigor_prompt  # private notes stay private
    opening = next(r for r in second.claude.requests if r.tag.purpose == "opening")
    assert "chair argued about the PEA." in opening.messages[0].content

    second.engine.forget("@Rigor")
    assert "Type /forget @Rigor again to confirm" in notices(second)[-1]
    assert MemoryStore(second.settings.data_dir).read_private("rigor")  # not yet
    second.engine.forget("@Rigor")
    assert notices(second)[-1] == "Forgot Rigor's private notes"
    await second.engine.ask("rigor", "worst case?")
    ask = next(r for r in second.ollama.requests if r.tag.purpose == "ask")
    assert "rigor argued" not in ask.messages[0].content


async def test_shared_facts_wait_for_keep_and_reach_everyone(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    await closed_session(h, say=USER_LINE)
    store = MemoryStore(h.settings.data_dir)
    assert "Dropped 1 proposed fact(s) not backed by your own words" in notices(h)
    [proposal] = proposals(h)
    assert [f.split("): ")[1] for f in proposal.facts] == [
        "The user rents a flat in Lyon.",
        "The user earns 3k a month.",
    ]
    assert store.read_shared() == ""  # nothing written without approval
    assert h.store.load_meta().runtime.memory_review.facts == proposal.facts
    assert h.engine.memory_review_pending
    shared_req = next(r for r in h.claude.requests if r.tag.purpose == "shared_memory")
    assert shared_req.model == "sonnet" and shared_req.json_schema is not None

    await h.engine.keep("2")
    assert store.read_shared().strip().endswith("The user earns 3k a month.")
    assert "Saved 1 fact(s) to shared memory" in notices(h)
    assert not h.engine.memory_review_pending
    assert not h.store.load_meta().runtime.memory_review.pending

    later = Harness(tmp_path, personas, memory_responder())
    await later.engine.step()
    await later.engine.step()
    contrarian = next(r for r in later.ollama.requests if r.tag.persona == "contrarian")
    assert "Facts the user stated in past sessions:" in contrarian.messages[0].content
    assert "The user earns 3k a month." in contrarian.messages[0].content


async def test_keep_usage_and_none(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    await h.engine.keep("all")
    assert notices(h)[-1] == "Nothing awaits approval"
    await closed_session(h, say=USER_LINE)
    await h.engine.keep("7")
    assert notices(h)[-1] == "usage: /keep 1 3 | all | none"
    await h.engine.keep("none")
    assert notices(h)[-1] == "No facts saved"
    assert MemoryStore(h.settings.data_dir).read_shared() == ""


async def test_approval_off_saves_facts_directly(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config: dict[str, Any] = {"memory": {"require_shared_approval": False}}
    h = Harness(tmp_path, personas, memory_responder(), config=config)
    await closed_session(h, say=USER_LINE)
    assert not proposals(h)
    assert "Saved 2 fact(s) to shared memory" in notices(h)
    assert "rents a flat in Lyon" in MemoryStore(h.settings.data_dir).read_shared()


async def test_shared_merge_is_proposed_when_user_md_outgrows_its_cap(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config: dict[str, Any] = {"memory": {"shared_cap_bytes": 120}}
    h = Harness(tmp_path, personas, memory_responder(), config=config)
    store = MemoryStore(h.settings.data_dir)
    store.append_shared(["- 2026-01-01 (finance): The user has a PEA opened in 2019."])
    await closed_session(h, say=USER_LINE)
    await h.engine.keep("all")
    rewrite = proposals(h)[-1]
    assert rewrite.rewrite == "- 2026-10-17 (finance): merged" and not rewrite.facts
    assert "The user has a PEA" in store.read_shared()  # unchanged until approved
    merge = [r for r in h.claude.requests if r.tag.purpose == "consolidate"]
    assert [r.tag.persona for r in merge] == ["chair"]

    await h.engine.keep("3")
    assert notices(h)[-1].startswith("usage: /keep all")
    await h.engine.keep("all")
    assert store.read_shared() == "- 2026-10-17 (finance): merged\n"
    assert "The user has a PEA" in store.shared_path.with_name("user.md.bak").read_text()
    assert not h.engine.memory_review_pending


async def test_private_notes_are_merged_past_the_cap(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config: dict[str, Any] = {"memory": {"private_cap_bytes": 150}}
    h = Harness(tmp_path, personas, memory_responder(), config=config)
    store = MemoryStore(h.settings.data_dir)
    store.append_private("rigor", ["- 2026-01-01 (finance): " + "old " * 30])
    await closed_session(h)
    assert store.read_private("rigor") == "- 2026-10-17 (finance): merged\n"
    assert any(n.startswith("Rigor's notes passed 150 bytes and were merged") for n in notices(h))
    merge = next(r for r in h.ollama.requests if r.tag.purpose == "consolidate")
    assert merge.tag.persona == "rigor"  # on its own backend


async def test_memory_failures_never_block_the_close(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = memory_responder(shared_memory="not json at all")

    def respond(req: CompletionRequest) -> Reply:
        """Fail Claude memory calls with a usage limit and Rigor's with a timeout."""
        if req.tag.purpose == "memory" and req.tag.persona in ("chair", "compounder"):
            return BackendRateLimitError("limit")
        if req.tag.purpose == "memory" and req.tag.persona == "rigor":
            return BackendTimeoutError("slow")
        return base(req)

    h = Harness(tmp_path, personas, respond)
    await closed_session(h, say=USER_LINE)
    assert h.state() is SessionState.CLOSED
    lines = notices(h)
    assert any(n.startswith("Chair couldn't write memory (limit)") for n in lines)
    assert any(n.startswith("Rigor couldn't write memory (slow)") for n in lines)
    assert any(n.startswith("Shared memory proposal failed (not valid JSON") for n in lines)
    assert MemoryStore(h.settings.data_dir).personas_with_notes() == ["contrarian", "pragmatist"]
    assert not proposals(h)


async def test_memory_false_persona_neither_reads_nor_writes_notes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rigor = personas["rigor"]
    forgetful = rigor.model_copy(update={"meta": rigor.meta.model_copy(update={"memory": False})})
    h = Harness(tmp_path, {**personas, "rigor": forgetful}, memory_responder())
    store = MemoryStore(h.settings.data_dir)
    store.append_private("rigor", ["- old private note"])
    store.append_shared(["- shared fact"])
    await h.engine.step()
    await h.engine.step()
    turn = next(r for r in h.ollama.requests if r.tag.persona == "rigor")
    assert "old private note" not in turn.messages[0].content
    assert "shared fact" in turn.messages[0].content
    await h.engine.close()
    assert all(not (r.tag.persona == "rigor" and r.tag.purpose == "memory") for r in h.requests())


async def test_forget_targets_and_confirmation(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    store = MemoryStore(h.settings.data_dir)
    store.append_private("rigor", ["- a"])
    store.append_private("outsider", ["- not in this council"])
    store.append_shared(["- fact"])

    h.engine.forget("")
    assert notices(h)[-1] == "usage: /forget @nick | shared | all"
    h.engine.forget("@Nobody")
    assert notices(h)[-1] == "usage: /forget @nick | shared | all"

    h.engine.forget("shared")
    h.engine.forget("rigor")  # a different target restarts the confirmation
    assert store.read_shared() and store.read_private("rigor")
    h.engine.forget("shared")
    h.engine.forget("shared")
    assert notices(h)[-1] == "Forgot the shared facts about you"
    h.engine.forget("shared")
    h.engine.forget("shared")
    assert notices(h)[-1] == "Nothing to forget: the shared facts about you was already empty"

    h.engine.forget("outsider")
    h.engine.forget("outsider")
    assert notices(h)[-1] == "Forgot outsider's private notes"
    h.engine.forget("all")
    h.engine.forget("all")
    assert notices(h)[-1].startswith("Forgot all memory")
    assert not store.root.exists()


async def test_close_estimate_and_events_on_disk(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    await h.engine.step()
    assert h.engine.estimate_close_calls() == 2  # synthesis + Chair's notes (it opened)
    h.engine.say(USER_LINE)
    assert h.engine.estimate_close_calls() == 3  # + the shared proposal
    await h.engine.step()
    assert h.engine.estimate_close_calls() == 4  # + Compounder's notes (it spoke)
    claude_before = len(h.claude.requests)
    await h.engine.close()
    assert len(h.claude.requests) - claude_before == 4
    assert any(isinstance(e, MemoryProposed) for e in h.store.events())
    md = (h.store.directory / "transcript.md").read_text()
    assert "**Shared memory, proposed facts:**" in md
    assert "> - " in md and "rents a flat in Lyon" in md
    assert any(isinstance(e, MessageEvent) for e in h.store.events())


async def test_members_who_never_spoke_write_no_notes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    await h.engine.step()  # only the moderator's opening
    await h.engine.ask("contrarian", "first reaction?")
    await h.engine.close()
    writers = {r.tag.persona for r in h.requests() if r.tag.purpose == "memory"}
    assert writers == {"chair", "contrarian"}
    assert "Rigor didn't speak this session; no notes written" in notices(h)
    assert MemoryStore(h.settings.data_dir).personas_with_notes() == ["chair", "contrarian"]


async def test_notes_from_another_council_are_left_out_unless_scope_is_global(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    store = MemoryStore(tmp_path / "data")
    store.append_private(
        "rigor", ["- 2026-10-01 (career): asked about a raise", "- 2026-10-02 (finance): PEA"]
    )
    store.append_shared(["- 2026-10-01 (career): The user is a nurse."])

    def rigor_prompt(h: Harness) -> str:
        """Return Rigor's blind-round prompt."""
        turn = next(r for r in h.requests() if r.tag.persona == "rigor" and r.tag.purpose == "turn")
        return str(turn.messages[0].content)

    scoped_run = Harness(tmp_path, personas, memory_responder())
    assert scoped_run.settings.data_dir == tmp_path / "data"
    await scoped_run.engine.step()
    await scoped_run.engine.step()
    prompt = rigor_prompt(scoped_run)
    assert "(finance): PEA" in prompt and "asked about a raise" not in prompt
    assert "The user is a nurse." in prompt  # shared facts stay global

    global_run = Harness(
        tmp_path, personas, memory_responder(), config={"memory": {"scope": "global"}}
    )
    await global_run.engine.step()
    await global_run.engine.step()
    assert "asked about a raise" in rigor_prompt(global_run)


async def test_note_writer_sees_only_this_councils_notes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    MemoryStore(tmp_path / "data").append_private(
        "rigor", ["- 2026-10-01 (career): asked about a raise"]
    )
    h = Harness(tmp_path, personas, memory_responder())
    await closed_session(h)
    writer = next(r for r in h.requests() if r.tag.persona == "rigor" and r.tag.purpose == "memory")
    assert "asked about a raise" not in str(writer.messages[0].content)


MEMORY_PURPOSES = {"memory", "shared_memory", "consolidate"}


async def test_read_only_council_reads_memory_but_writes_nothing(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    store = MemoryStore(tmp_path / "data")
    store.append_private("chair", ["- 2026-10-01 (finance): kept rounds short"])
    store.append_shared(["- 2026-10-01 (finance): The user is a nurse."])
    before = (store.read_private("chair"), store.read_shared())
    h = Harness(tmp_path, personas, memory_responder(), memory=MemoryMode.READ_ONLY)
    await h.engine.step()
    h.engine.say(USER_LINE)
    await h.engine.step()
    opening = next(r for r in h.claude.requests if r.tag.purpose == "opening")
    assert "kept rounds short" in str(opening.messages[0].content)
    turn = next(r for r in h.requests() if r.tag.persona == "rigor" and r.tag.purpose == "turn")
    assert "The user is a nurse." in str(turn.messages[0].content)
    assert h.engine.estimate_close_calls() == 1  # the synthesis only
    await h.engine.close()
    assert not {r.tag.purpose for r in h.requests()} & MEMORY_PURPOSES
    assert (store.read_private("chair"), store.read_shared()) == before
    assert store.personas_with_notes() == ["chair"]
    assert "Memory is read-only for this council: nothing saved" in notices(h)
    assert not proposals(h)


async def test_memory_off_council_neither_reads_nor_writes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    store = MemoryStore(tmp_path / "data")
    store.append_private("rigor", ["- 2026-10-01 (finance): liked index funds"])
    store.append_shared(["- 2026-10-01 (finance): The user is a nurse."])
    h = Harness(tmp_path, personas, memory_responder(), memory=MemoryMode.OFF)
    await h.engine.step()
    h.engine.say(USER_LINE)
    await h.engine.step()
    prompts = " ".join(str(m.content) for r in h.requests() for m in r.messages)
    assert "liked index funds" not in prompts and "The user is a nurse." not in prompts
    assert "# Memory from past sessions" not in prompts
    assert h.engine.estimate_close_calls() == 1
    await h.engine.close()
    assert not {r.tag.purpose for r in h.requests()} & MEMORY_PURPOSES
    assert store.personas_with_notes() == ["rigor"]
    assert "Memory is off for this council: nothing saved" in notices(h)
