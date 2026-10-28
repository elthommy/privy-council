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
