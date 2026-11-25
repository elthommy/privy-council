"""Web research by the moderator (fake backends; the CLI tool flags are tested in backends/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import (
    BackendRateLimitError,
    BackendTimeoutError,
    CompletionRequest,
    CompletionResult,
)
from privy_council.backends.fake import Reply
from privy_council.config import Settings
from privy_council.engine import prompts
from privy_council.engine.research import (
    ResearchMode,
    ResearchRequiredError,
    fetches,
    resolve_mode,
    searches,
)
from privy_council.models.council import CouncilPreset, ProtocolSettings
from privy_council.models.persona import Persona
from privy_council.models.session import (
    ErrorEvent,
    Phase,
    ResearchAdded,
    ResearchNote,
    SessionState,
)
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas

FACTS = "- Gold is about $4,170/oz (2026-10-16, https://example.org/gold)"


def research_result(
    text: str = FACTS, queries: tuple[str, ...] = ("gold price",), urls: tuple[str, ...] = ()
) -> Reply:
    """A research reply carrying the WebSearch and WebFetch inputs like the real backend does."""
    calls = [{"name": "WebSearch", "input": {"query": q}} for q in queries]
    calls += [{"name": "WebFetch", "input": {"url": u, "prompt": "p"}} for u in urls]
    return CompletionResult(
        text=text, latency_s=0, backend="fake", model="sonnet", raw={"tool_calls": calls}
    )


def responder(
    decisions: list[dict[str, Any] | str], research: Reply | list[Reply] | None = None
) -> Any:
    """Standard responder plus a scripted research reply (a list: one per call)."""
    base = standard_responder(decisions)
    script = research if isinstance(research, list) else None
    single = research if not isinstance(research, list) else None

    def respond(req: CompletionRequest) -> Reply:
        """Answer research calls with `research`, everything else as usual."""
        if req.tag.purpose == "research":
            if script is not None:
                return script.pop(0)
            return single if single is not None else research_result()
        return base(req)

    return respond


def research_requests(h: Harness) -> list[CompletionRequest]:
    """Return the research calls the moderator made."""
    return [r for r in h.claude.requests if r.tag.purpose == "research"]


@pytest.mark.parametrize(
    ("enabled", "auto", "flag", "expected"),
    [
        (True, False, None, ResearchMode.ON_DEMAND),
        (True, True, None, ResearchMode.AUTO),
        (False, True, None, ResearchMode.OFF),
        (False, False, True, ResearchMode.AUTO),
        (True, True, False, ResearchMode.OFF),
        (True, False, None, ResearchMode.ON_DEMAND),
    ],
)
def test_resolve_mode(
    tmp_path: Path, enabled: bool, auto: bool, flag: bool | None, expected: ResearchMode
) -> None:
    settings = Settings.model_validate({"root": tmp_path, "research": {"enabled": enabled}})
    council = CouncilPreset(
        name="c",
        moderator="chair",
        members=["rigor"],
        protocol=ProtocolSettings(auto_research=auto),
    )
    assert resolve_mode(settings, council, flag) is expected


@pytest.mark.parametrize(("enabled", "flag"), [(True, None), (False, None), (True, True)])
def test_required_research_ignores_config(tmp_path: Path, enabled: bool, flag: bool | None) -> None:
    settings = Settings.model_validate({"root": tmp_path, "research": {"enabled": enabled}})
    council = CouncilPreset(
        name="geo",
        moderator="chair",
        members=["rigor"],
        protocol=ProtocolSettings(require_research=True),
    )
    assert resolve_mode(settings, council, flag) is ResearchMode.REQUIRED
    with pytest.raises(ResearchRequiredError, match="'geo' requires web research"):
        resolve_mode(settings, council, False)


def test_searches_extracts_only_web_search_queries() -> None:
    result = CompletionResult(
        text="",
        latency_s=0,
        backend="f",
        model="m",
        raw={
            "tool_calls": [
                {"name": "WebSearch", "input": {"query": "a"}},
                {"name": "WebFetch", "input": {"url": "https://x"}},
                {"name": "WebSearch", "input": {}},
                {"name": "WebFetch", "input": {"url": "https://x"}},  # read twice, shown once
            ]
        },
    )
    assert searches(result) == ["a"]
    assert fetches(result) == ["https://x"]


@pytest.mark.parametrize(
    ("urls", "notice"),
    [
        (("https://example.org/gold",), "Chair read web pages: https://example.org/gold"),
        ((), "Chair answered without searching the web"),
    ],
)
async def test_research_notice_without_a_search(
    tmp_path: Path, personas: dict[str, Persona], urls: tuple[str, ...], notice: str
) -> None:
    research = research_result(queries=(), urls=urls)
    h = Harness(
        tmp_path,
        personas,
        responder([decision([("rigor", "x")])], research),
        research=ResearchMode.AUTO,
    )
    await h.engine.step()
    system = h.texts("system")
    assert notice in system
    assert not any("searched the web" in t for t in system)
    [note] = h.store.load_meta().runtime.research
    assert note.searches == [] and note.fetched == list(urls)
    research_md = (h.store.directory / "dossier" / "research.md").read_text()
    assert ("· read: https://example.org/gold" in research_md) == bool(urls)


async def test_auto_research_before_opening_reaches_blind_round(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path, personas, responder([decision([("rigor", "x")])]), research=ResearchMode.AUTO
    )
    await h.engine.step()  # research, then the opening framed with its facts
    [req] = research_requests(h)
    assert [r.tag.purpose for r in h.claude.requests] == ["research", "opening"]
    assert "$4,170/oz" in h.claude.requests[1].messages[0].content
    assert req.tools == ["WebSearch", "WebFetch"]
    assert req.timeout_s == 300 and req.max_output_tokens == 3000
    assert "Search the web" in req.messages[0].content
    assert "search for that event itself" in req.messages[0].content
    assert "My search didn't confirm that [event]" in h.claude.requests[1].messages[0].content
    assert "Chair searched the web: gold price" in h.texts("system")
    assert any(isinstance(e, ResearchAdded) for e in h.events)
    assert h.phase() is Phase.ROUND

    await h.engine.step()  # blind round: every member sees the same facts
    for r in h.requests():
        if r.tag.purpose in ("turn", "moderate"):
            assert "Fresh facts (web search by the moderator" in r.messages[0].content
            assert "$4,170/oz" in r.messages[0].content
    research_md = (h.store.directory / "dossier" / "research.md").read_text()
    assert "searched: gold price" in research_md and FACTS in research_md
    assert "### Research:" in h.store.transcript_md.read_text()
    assert h.store.load_meta().runtime.research[0].searches == ["gold price"]


async def test_the_councils_brief_and_focus_reach_the_chair(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([decision([("rigor", "x")])]),
        research=ResearchMode.AUTO,
        brief="# This council\nAsk whether the pull is guaranteed.",
        research_focus="the pity rules and the price of a pull",
    )
    await h.engine.step()
    [req] = research_requests(h)
    assert "Focus: the pity rules and the price of a pull" in req.messages[0].content
    opening = h.claude.requests[1]
    assert "Ask whether the pull is guaranteed." in opening.system
    # The focus is an instruction: the note gets a short heading (2026-10-27 Discord debate).
    assert h.store.load_meta().runtime.research[0].focus == "background for the question"


async def test_on_demand_research_with_focus(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, responder([decision([("rigor", "x")])]))
    await h.engine.step()
    assert research_requests(h) == []  # on-demand: nothing after the opening
    await h.engine.research("current real interest rates")
    [req] = research_requests(h)
    assert "Focus: current real interest rates" in req.messages[0].content
    await h.engine.research(None)  # a second search sees the first one's facts
    assert "$4,170/oz" in research_requests(h)[1].messages[0].content
    assert not h.engine.busy


async def test_later_research_is_told_which_pages_were_read(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    filing = "https://example.org/10-q"
    first = research_result(urls=(filing, filing))
    h = Harness(
        tmp_path, personas, responder([decision([("rigor", "x")])], [first, research_result()])
    )
    await h.engine.step()
    await h.engine.research("the 10-Q")
    await h.engine.research("debt maturities")
    assert "Chair read web pages: https://example.org/10-q" in h.texts("system")  # once
    first_req, second_req = research_requests(h)
    assert "already read" not in first_req.messages[0].content
    prompt = second_req.messages[0].content
    assert f"Pages earlier searches already read (their facts are above): {filing}." in prompt
    assert "Don't fetch them again." in prompt


async def test_research_off_and_user_research_uncapped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    off = Harness(tmp_path / "off", personas, responder([]), research=ResearchMode.OFF)
    await off.engine.research("x")
    assert research_requests(off) == []
    assert any("Research is off" in t for t in off.texts("system"))

    # The moderator's own allowance doesn't limit the user's /research.
    h = Harness(
        tmp_path / "many",
        personas,
        responder([]),
        config={"research": {"auto_max_per_session": 0}},
    )
    for focus in "abcdefg":
        await h.engine.research(focus)
    assert len(research_requests(h)) == 7
    assert len(h.engine.meta.runtime.research) == 7


async def test_research_failure_never_blocks_the_debate(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([], research=BackendTimeoutError("slow search")),
        research=ResearchMode.AUTO,
    )
    await h.engine.step()
    assert h.phase() is Phase.ROUND
    assert any(
        isinstance(e, ErrorEvent) and "Research failed (slow search)" in e.text for e in h.events
    )


async def test_required_research_failure_pauses_until_a_retry_succeeds(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([], research=[BackendTimeoutError("slow search"), research_result()]),
        research=ResearchMode.REQUIRED,
    )
    await h.engine.step()
    assert h.state() is SessionState.ERROR_RECOVERABLE
    assert h.phase() is Phase.OPENING
    assert [r.tag.purpose for r in h.claude.requests] == ["research"]  # no opening yet
    assert any(isinstance(e, ErrorEvent) and "slow search" in e.text for e in h.events)

    await h.engine.step()  # /step retries the search, then opens
    assert [r.tag.purpose for r in h.claude.requests] == ["research", "research", "opening"]
    assert h.phase() is Phase.ROUND


async def test_required_research_needs_a_claude_moderator(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, responder([]), fallback=True, research=ResearchMode.REQUIRED)
    h.engine.claude_exhausted = True  # Chair now routes to the local fallback model
    await h.engine.step()
    assert h.state() is SessionState.ERROR_RECOVERABLE
    assert h.requests() == []
    assert any(
        isinstance(e, ErrorEvent) and "needs a Claude-backed moderator" in e.text for e in h.events
    )


async def test_rate_limited_research_is_not_rerouted_locally(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([], research=BackendRateLimitError("limit")),
        fallback=True,
    )
    await h.engine.research("x")
    assert not any(r.tag.purpose == "research" for r in h.ollama.requests)
    assert any(isinstance(e, ErrorEvent) and "Research failed" in e.text for e in h.events)


async def test_research_needs_a_claude_moderator(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, responder([]), fallback=True)
    h.engine.claude_exhausted = True  # Chair now routes to the local fallback model
    await h.engine.research("x")
    assert h.requests() == []
    assert any("needs a Claude-backed moderator" in t for t in h.texts("system"))


async def test_facts_are_capped_to_their_budget_share(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    huge = "- fact " + "x" * 60_000
    h = Harness(tmp_path, personas, responder([decision([("rigor", "x")])], research_result(huge)))
    await h.engine.step()
    await h.engine.research(None)
    await h.engine.step()
    rigor = next(r for r in h.ollama.requests if r.tag.persona == "rigor")
    content = rigor.messages[0].content
    # qwen3:8b: (8192 - 1500) tokens * 4 chars * 0.25 share ≈ 6.7k chars of facts at most.
    assert len(content) < 9_000
    assert "[…trimmed]" in content and "Round 1, blind" in content


async def test_seeded_research_replaces_the_first_search(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path, personas, responder([decision([("rigor", "x")])]), research=ResearchMode.REQUIRED
    )
    note = ResearchNote(focus="shared", text=FACTS, searches=["gold price"])
    h.engine.seed_research([note])
    await h.engine.step()  # the opening, framed with the seeded facts, no search of its own
    assert research_requests(h) == []
    assert "$4,170/oz" in h.claude.requests[0].messages[0].content
    assert [e.note for e in h.events if isinstance(e, ResearchAdded)] == [note]
    assert h.store.load_meta().runtime.research == [note]

    h.engine.seed_research([note.model_copy(update={"focus": "late"})])  # after the opening
    assert h.engine.meta.runtime.research == [note]


async def test_members_see_fact_tags_and_a_wrong_citation_is_logged(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    base = responder([decision([("rigor", "x")])])

    def respond(req: CompletionRequest) -> Reply:
        """Rigor cites the fact with a price it doesn't give."""
        if req.tag.purpose == "turn" and req.tag.persona == "rigor":
            return "Gold is near $3,900/oz [F1]."
        return base(req)

    h = Harness(tmp_path, personas, respond, research=ResearchMode.AUTO)
    await h.engine.step()  # research, then the opening
    await h.engine.step()  # blind round
    turn = next(r for r in h.requests() if r.tag.purpose == "turn")
    assert "- [F1] Gold is about $4,170/oz" in turn.messages[0].content
    assert "put its tag right after it" in turn.system
    assert prompts.FACT_TAG_RULE in turn.messages[-1].content.split("# Your task")[-1]
    [flaw] = [f for f in h.engine.meta.runtime.flaws if f.member == "rigor"]
    assert (flaw.kind.value, flaw.note) == ("miscited", "$3,900 is not in [F1].")
    update = [r for r in h.requests() if r.tag.purpose == "moderate"][-1]
    assert '$3,900 is not in [F1]. In "' in update.messages[-1].content  # the Chair sees it


async def forecast_with_late_facts(
    tmp_path: Path, personas: dict[str, Persona], config: dict[str, Any] | None = None
) -> Harness:
    """Run two rounds of a forecast, then the user's lookup, then the close."""
    floor = [("rigor", "go"), ("contrarian", "go")]
    base = responder([decision(floor), decision(floor)])

    def respond(req: CompletionRequest) -> Reply:
        """Give every member turn an estimate line."""
        return "Same view.\n**Estimate:** 12% (7-20%)" if req.tag.purpose == "turn" else base(req)

    protocol = ProtocolSettings(estimate="probability")
    h = Harness(tmp_path, personas, respond, protocol=protocol, config=config)
    for _ in range(3):  # opening, blind round, round 2
        await h.engine.step()
    await h.engine.research("the closing price")
    await h.engine.close()
    return h


async def test_members_get_a_last_turn_on_facts_that_came_after_their_estimate(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = await forecast_with_late_facts(tmp_path, personas)
    late = [r for r in h.requests() if "fresh facts arrived" in r.messages[0].content]
    late = [r for r in late if r.tag.purpose == "turn"]
    speakers = sorted(str(r.tag.persona) for r in late)
    assert speakers == ["compounder", "pragmatist", "rigor"]  # no devil
    assert "the closing price" in late[0].messages[0].content
    assert "Gold is about $4,170" in late[0].messages[0].content
    assert any(t.startswith("Facts arrived after the last turn of") for t in h.texts("system"))
    assert h.state() is SessionState.CLOSED
    assert len(h.engine.meta.runtime.estimates["pragmatist"]) == 2


async def test_late_facts_turns_can_be_turned_off(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    config = {"engine": {"late_facts_turns": False}}
    h = await forecast_with_late_facts(tmp_path, personas, config)
    assert not any("fresh facts arrived" in r.messages[0].content for r in h.requests())


async def test_a_third_lookup_of_the_same_thing_is_skipped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    empty = research_result(
        "- Totals only (https://example.org/q)\nMissing: maturities by quarter",
        queries=("acme debt maturities",),
    )
    h = Harness(
        tmp_path,
        personas,
        responder([], [empty, empty, research_result(), research_result()]),
        config={"research": {"auto_max_per_session": 9}},
    )
    focus = "Acme 10-Q debt maturities by quarter 2027"
    for _ in range(3):
        await h.engine._auto_research([focus])
    assert len(research_requests(h)) == 2
    notes = h.engine.meta.runtime.research
    assert notes[0].missing == ["maturities by quarter"]
    assert any("Skipped: searched 2 times already" in t for t in h.texts("system"))
    second = research_requests(h)[1].messages[0].content
    assert "Queries earlier searches already ran: acme debt maturities." in second
    assert "Still not found after them: maturities by quarter." in second
    # A new date is a new lookup, and the user's /research always runs.
    await h.engine._auto_research(["Acme 10-Q debt maturities by quarter 2028"])
    await h.engine.research(focus)
    assert len(research_requests(h)) == 4


async def test_the_users_answer_to_the_opening_gets_a_lookup_before_round_1(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    replies = [research_result(), research_result("- Index at 16,240 (2026-11-19, https://e.org)")]
    h = Harness(
        tmp_path,
        personas,
        responder([decision([("rigor", "x")])], replies),
        research=ResearchMode.AUTO,
        protocol=ProtocolSettings(max_rounds=2, estimate="probability"),
    )
    await h.engine.step()  # first search, then the opening
    h.engine.say("a 10% fall from the record counts")
    await h.engine.step()  # the terms lookup, then round 1
    first, terms = research_requests(h)
    assert "Focus: the terms as the user's answer" in terms.messages[0].content
    assert "a 10% fall from the record counts" in terms.messages[0].content
    assert "a 10% fall" not in first.messages[0].content
    notes = h.engine.meta.runtime.research
    assert [n.focus for n in notes] == [
        "background for the question",
        "where the question's terms stand today",
    ]
    assert "Chair looks up where your answer's terms stand today" in h.texts("system")
    turns = [r for r in h.requests() if r.tag.purpose == "turn"]
    assert turns and all("16,240" in r.messages[0].content for r in turns)


async def test_no_answer_to_the_opening_no_terms_lookup(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([decision([("rigor", "x")])]),
        research=ResearchMode.AUTO,
        protocol=ProtocolSettings(max_rounds=2, estimate="probability"),
    )
    await h.engine.step()
    await h.engine.step()
    assert len(research_requests(h)) == 1


async def answer_session(
    tmp_path: Path, personas: dict[str, Persona], focus: str, answer: str | None
) -> Harness:
    """Run a forecast whose Chair asks after round 1; the user answers (or not)."""
    asks = {**decision([("rigor", "x")]), "questions_for_user": ["Does a delayed launch count?"]}
    replies = [research_result(), research_result("- 7 delays in 2025 (https://e.org/w)")]
    base = responder([asks, *[decision([("rigor", "x")])] * 2], replies)

    def respond(req: CompletionRequest) -> Reply:
        """Answer the lookup call with `focus`."""
        if req.tag.purpose == "answer_lookup":
            return focus
        return base(req)

    h = Harness(
        tmp_path,
        personas,
        respond,
        research=ResearchMode.AUTO,
        protocol=ProtocolSettings(max_rounds=3, estimate="probability"),
    )
    await h.engine.step()
    await h.engine.step()
    if answer:
        h.engine.say(answer)
    await h.engine.step()
    return h


def lookup_calls(h: Harness) -> list[CompletionRequest]:
    """Return the calls asking what an answer changes."""
    return [r for r in h.requests() if r.tag.purpose == "answer_lookup"]


async def test_an_answer_between_rounds_gets_a_lookup_before_the_next_round(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    reply = '{"answer": "decides", "focus": "delayed rocket launches, count in 2025"}'
    h = await answer_session(tmp_path, personas, reply, "yes, delays count")
    (call,) = lookup_calls(h)
    assert not call.tools
    assert "Does a delayed launch count?" in call.messages[0].content
    assert "yes, delays count" in call.messages[0].content
    note = h.engine.meta.runtime.research[-1]
    assert note.focus == "delayed rocket launches, count in 2025" and not note.auto
    last_turn = [r for r in h.requests() if r.tag.purpose == "turn"][-1]
    assert "7 delays in 2025" in last_turn.messages[0].content
    # The next round has no new answer, so nothing more is asked.
    await h.engine.step()
    assert len(lookup_calls(h)) == 1


@pytest.mark.parametrize(
    "reply",
    [
        '{"answer": "decides", "focus": ""}',
        '{"answer": "declines", "focus": "delays"}',
        '{"focus": "delays"}',
        "no idea",
    ],
)
async def test_an_answer_that_changes_nothing_gets_no_search(
    tmp_path: Path, personas: dict[str, Persona], reply: str
) -> None:
    h = await answer_session(tmp_path, personas, reply, "I leave it to you")
    assert len(lookup_calls(h)) == 1
    assert len(research_requests(h)) == 1


async def test_no_answer_between_rounds_no_lookup(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = await answer_session(tmp_path, personas, '{"answer": "decides", "focus": "x"}', None)
    assert not lookup_calls(h)
