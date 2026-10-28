"""Local web search (SearXNG) in local-only sessions: pure helpers and the engine flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import (
    BackendHealth,
    BackendUnavailableError,
    CompletionRequest,
    CompletionResult,
)
from privy_council.backends.fake import Reply
from privy_council.config import Settings
from privy_council.engine import context, prompts
from privy_council.engine.local_research import (
    QUERIES_SCHEMA,
    check_summary,
    fallback_query,
    merge_results,
    missing_facts,
    parse_queries,
    render_results,
    shorter_query,
    without_missing,
)
from privy_council.engine.moderation import DecisionError
from privy_council.engine.research import ResearchMode, ResearchRequiredError, resolve_mode
from privy_council.models.council import CouncilPreset, ProtocolSettings
from privy_council.models.persona import Persona
from privy_council.models.research import SearchResult
from privy_council.models.session import ErrorEvent, ResearchAdded, ResearchNote, SessionState
from privy_council.storage.sessions import RESEARCH_RAW_JSONL
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas

LOCAL = {"research": {"local": {"enabled": True}}}
ECB = SearchResult(
    title="Key ECB interest rates",
    url="https://www.ecb.europa.eu/rates",
    snippet="Deposit facility: 2.50% from 16 September 2026.",
    published="2026-09-16",
)
INSEE = SearchResult(
    title="Inflation in October",
    url="https://www.insee.fr/cpi",
    snippet="Consumer prices rose 1.1% year on year.",
)


def result(text: str) -> CompletionResult:
    """A plain reply."""
    return CompletionResult(text=text, latency_s=0, backend="fake", model="m")


class FakeSearch:
    """Scripted hits per query; an exception value is raised instead."""

    name = "fake-search"

    def __init__(self, hits: dict[str, list[SearchResult] | Exception] | None = None) -> None:
        """Default: every query finds ECB and INSEE."""
        self.hits = hits or {}
        self.queries: list[tuple[str, str]] = []

    async def search(self, query: str, lang: str) -> list[SearchResult]:
        """Return (or raise) the scripted hits."""
        self.queries.append((query, lang))
        found = self.hits.get(query, [ECB, INSEE])
        if isinstance(found, Exception):
            raise found
        return found

    async def health(self) -> BackendHealth:
        """Always healthy."""
        return BackendHealth(backend="local search", ok=True)


SUMMARY = (
    "- ECB deposit rate 2.50% since 16 September 2026 (https://www.ecb.europa.eu/rates)\n"
    "- Inflation 1.1% year on year (https://www.insee.fr/cpi)\n"
    "- Gold is at $4,170/oz (https://www.insee.fr/cpi)\n"  # figure not in the hits
    "- Unemployment 7.3% (https://made-up.example/jobs)"  # URL not in the hits
)


def local_responder(
    decisions: list[dict[str, Any] | str],
    queries: list[Reply] | None = None,
    summaries: list[str] | None = None,
) -> Any:
    """Standard responder plus scripted local research calls (summaries: SUMMARY by default)."""
    base = standard_responder(decisions)
    script = list(queries) if queries is not None else [{"queries": ["ecb rate", "insee cpi"]}]
    notes = list(summaries or [])

    def respond(req: CompletionRequest) -> Reply:
        """Queries (and their repair) from `script`; the summary is SUMMARY."""
        if req.json_schema == QUERIES_SCHEMA:  # the queries call and its repair
            return script.pop(0)
        if req.tag.purpose == "research":
            return notes.pop(0) if notes else SUMMARY
        return base(req)

    return respond


def harness(
    tmp_path: Path,
    personas: dict[str, Persona],
    *,
    search: FakeSearch | None = None,
    queries: list[Reply] | None = None,
    research: ResearchMode = ResearchMode.REQUIRED,
    config: dict[str, Any] | None = None,
    summaries: list[str] | None = None,
) -> Harness:
    """A local-only engine with local search on."""
    return Harness(
        tmp_path,
        personas,
        local_responder([decision([("rigor", "x")])], queries, summaries),
        research=research,
        local_only=True,
        config=config if config is not None else LOCAL,
        search=search if search is not None else FakeSearch(),
    )


# --- pure helpers ---------------------------------------------------------------


def test_parse_queries_cleans_dedupes_and_caps() -> None:
    reply = {"queries": ["  ecb   rate ", "ECB rate", "", "insee cpi", "gold", "x" * 300]}
    got = parse_queries(
        CompletionResult(text="", structured=reply, latency_s=0, backend="f", model="m"), 3
    )
    assert got == ["ecb rate", "insee cpi", "gold"]


@pytest.mark.parametrize("text", ["not json", '{"queries": []}', '{"queries": ["  "]}'])
def test_parse_queries_rejects_unusable_replies(text: str) -> None:
    with pytest.raises(DecisionError):
        parse_queries(result(text), 4)


def test_fallback_query_uses_the_focus_else_the_topic() -> None:
    assert fallback_query("ECB rate", "Should I buy?") == "ECB rate"
    assert fallback_query(prompts.RESEARCH_DEFAULT_FOCUS, "Should  I\nbuy?") == "Should I buy?"
    standing = "the game, the character's kit, pity rules"
    assert fallback_query(standing, "Pull Lucilla?", standing) == "Pull Lucilla?"


def test_merge_interleaves_dedupes_and_caps() -> None:
    a = [SearchResult(title="a1", url="https://a/1/"), SearchResult(title="a2", url="https://a/2")]
    b = [SearchResult(title="dup", url="https://a/1"), SearchResult(title="b2", url="https://b/2")]
    assert [r.title for r in merge_results([a, b], 10)] == ["a1", "a2", "b2"]
    assert [r.title for r in merge_results([a, b], 2)] == ["a1", "a2"]
    assert merge_results([], 5) == []


def test_render_results_numbers_hits_with_dates() -> None:
    text = render_results([ECB, INSEE])
    assert text.startswith("[1] Key ECB interest rates (2026-09-16)\nhttps://www.ecb.europa.eu")
    assert "[2] Inflation in October\nhttps://www.insee.fr/cpi\nConsumer" in text


def test_check_summary_drops_unbacked_figures_and_unknown_urls() -> None:
    kept, dropped = check_summary(SUMMARY, [ECB, INSEE])
    assert kept.splitlines() == SUMMARY.splitlines()[:2]
    assert [d.split(" (")[0] for d in dropped] == ["- Gold is at $4,170/oz", "- Unemployment 7.3%"]


def test_check_summary_keeps_context_figures_continuations_and_url_digits() -> None:
    text = (
        "- Your budget of €3,500 is in the question\n"
        "  and this line continues it.\n"
        "- No result gives the 2027 forecast.\n"
        "- Rates: see https://www.ecb.europa.eu/rates/ (2026-09-16)"
    )
    kept, dropped = check_summary(text, [ECB], context="I can spend €3,500.")
    assert dropped == []
    assert kept == text


def test_check_summary_restores_a_lost_missing_prefix() -> None:
    text = (
        "- Rates: see https://www.ecb.europa.eu/rates/ (2026-09-16)\n"
        "- Wuthering Waves pity and 50/50 rules and the price of 1 pull."
    )
    kept, dropped = check_summary(text, [ECB])
    assert dropped == []
    assert (
        kept.splitlines()[-1]
        == "- Missing: Wuthering Waves pity and 50/50 rules and the price of 1 pull"
    )
    assert missing_facts(kept).startswith("Wuthering Waves pity")
    # a note with its own Missing line, or ending with a cited fact, is left alone
    assert check_summary(MISSING, [ECB])[0] == MISSING
    assert check_summary(text.splitlines()[0], [ECB])[0] == text.splitlines()[0]


# --- engine flow ----------------------------------------------------------------


async def test_local_only_research_searches_summarizes_and_checks(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    search = FakeSearch()
    h = harness(tmp_path, personas, search=search)
    await h.engine.step()  # local search, then the opening
    assert h.claude.requests == []  # nothing reached Claude
    purposes = [r.tag.purpose for r in h.ollama.requests]
    assert purposes[:3] == ["research_queries", "research", "opening"]
    assert search.queries == [("ecb rate", "en"), ("insee cpi", "en")]
    summary_prompt = h.ollama.requests[1].messages[0].content
    assert "[1] Key ECB interest rates (2026-09-16)" in summary_prompt
    note = next(e.note for e in h.events if isinstance(e, ResearchAdded))
    assert note.source == "local" and note.searches == ["ecb rate", "insee cpi"]
    assert "2.50%" in note.text and "$4,170" not in note.text and "made-up" not in note.text
    system = h.texts("system")
    assert any("queries leave this PC" in t for t in system)
    assert any(
        "searched the web (local search): ecb rate; insee cpi (2 result(s))" in t for t in system
    )
    assert any("2 research bullet(s) dropped" in t for t in system)
    raw = json.loads((h.store.directory / RESEARCH_RAW_JSONL).read_text().splitlines()[0])
    assert raw["queries"] == ["ecb rate", "insee cpi"] and len(raw["results"]) == 2
    assert raw["summary"] == SUMMARY and len(raw["dropped"]) == 2
    research_md = (h.store.directory / "dossier" / "research.md").read_text()
    assert "local search · searched: ecb rate; insee cpi" in research_md


MISSING = (
    "- ECB deposit rate 2.50% since 16 September 2026 (https://www.ecb.europa.eu/rates)\n"
    "- Missing: Wuthering Waves pity and 50/50 rules"
)


@pytest.mark.parametrize(
    ("note", "facts"),
    [
        (MISSING, "Wuthering Waves pity and 50/50 rules"),
        ("- a fact\n**Missing:** the price of one pull", "the price of one pull"),
        ("- Results do not state the pity rules.", ""),
        ("- Missing pieces are common in gacha", ""),
    ],
)
def test_missing_facts(note: str, facts: str) -> None:
    assert missing_facts(note) == facts


async def test_a_follow_up_search_before_round_1_for_what_is_missing(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    search = FakeSearch()
    queries: list[Reply] = [{"queries": ["ecb rate"]}, {"queries": ["wuthering waves pity"]}]
    h = harness(tmp_path, personas, search=search, queries=queries, summaries=[MISSING, SUMMARY])
    await h.engine.step()  # search, follow-up search, then the opening
    purposes = [r.tag.purpose for r in h.ollama.requests]
    assert purposes[:5] == ["research_queries", "research"] * 2 + ["opening"]
    assert "Chair searches again for what is missing: Wuthering Waves pity and 50/50 rules" in (
        h.texts("system")
    )
    follow_up = h.ollama.requests[2].messages[0].content
    assert "Focus: Wuthering Waves pity and 50/50 rules" in follow_up
    assert "ECB deposit rate 2.50%" in follow_up  # written with the first note in hand
    notes = h.engine.meta.runtime.research
    assert [n.focus for n in notes][1] == "Wuthering Waves pity and 50/50 rules"
    assert not any(n.auto for n in notes)  # not the Chair's own search quota
    assert [q for q, _ in search.queries] == ["ecb rate", "wuthering waves pity"]


@pytest.mark.parametrize("follow_up", [True, False])
async def test_no_follow_up_without_a_missing_line_or_when_off(
    tmp_path: Path, personas: dict[str, Persona], follow_up: bool
) -> None:
    config = {"research": {"local": {"enabled": True, "follow_up": follow_up}}}
    summaries = [SUMMARY if follow_up else MISSING]
    h = harness(tmp_path, personas, config=config, summaries=summaries)
    await h.engine.step()
    assert [r.tag.purpose for r in h.ollama.requests][:3] == [
        "research_queries",
        "research",
        "opening",
    ]


async def test_queries_are_repaired_once_then_fall_back_to_the_focus(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    search = FakeSearch()
    h = harness(
        tmp_path,
        personas,
        search=search,
        queries=["nope", "still nope"],
        research=ResearchMode.ON_DEMAND,
    )
    await h.engine.step()
    await h.engine.research("ECB deposit rate")
    purposes = [r.tag.purpose for r in h.ollama.requests if r.tag.persona == "chair"]
    assert purposes[-3:] == ["research_queries", "repair", "research"]
    assert search.queries == [("ECB deposit rate", "en")]
    assert any("gave no usable search queries" in t for t in h.texts("system"))


async def test_failed_query_is_reported_and_all_failing_pauses_required_research(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    down = BackendUnavailableError("SearXNG not reachable")
    search = FakeSearch({"ecb rate": down, "insee cpi": [INSEE]})
    h = harness(tmp_path, personas, search=search)
    await h.engine.step()
    assert any("Local search failed for 'ecb rate'" in t for t in h.texts("system"))
    assert h.state() is not SessionState.ERROR_RECOVERABLE  # one query still found hits

    h2 = harness(tmp_path / "b", personas, search=FakeSearch({"ecb rate": down, "insee cpi": down}))
    await h2.engine.step()
    assert h2.state() is SessionState.ERROR_RECOVERABLE  # required: retried with /step
    assert not any(isinstance(e, ResearchAdded) for e in h2.events)


async def test_no_hits_fail_the_search(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = harness(
        tmp_path,
        personas,
        search=FakeSearch({"ecb rate": [], "insee cpi": []}),
        research=ResearchMode.ON_DEMAND,
    )
    await h.engine.step()
    await h.engine.research("x")
    assert not any(isinstance(e, ErrorEvent) for e in h.events)  # the chat's generic line
    assert any(
        t == "Chair's search found nothing (ecb rate; insee cpi); the debate continues without it"
        for t in h.texts("system")
    )


async def test_no_hits_retry_once_with_shorter_queries(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    long = ["ESM lending capacity firepower 2026 official", "ECB rate 2026"]
    search = FakeSearch({q: [] for q in long})
    h = harness(
        tmp_path,
        personas,
        search=search,
        queries=[{"queries": long}],
        research=ResearchMode.ON_DEMAND,
    )
    await h.engine.step()
    await h.engine.research("x")
    assert [q for q, _ in search.queries][2:] == ["ESM lending capacity firepower", "ECB rate"]
    assert any(t.startswith("No result: Chair searches again") for t in h.texts("system"))
    [added] = [e for e in h.events if isinstance(e, ResearchAdded)]
    assert added.note.searches == [*long, "ESM lending capacity firepower", "ECB rate"]


@pytest.mark.parametrize(
    ("query", "short"),
    [
        ("ESM lending capacity firepower 2026 official", "ESM lending capacity firepower"),
        ("France 10-year OAT yield auction results October 2026", "France 10-year OAT yield"),
        ("ecb rate", "ecb rate"),
        ("official 2026", ""),
    ],
)
def test_shorter_query(query: str, short: str) -> None:
    assert shorter_query(query) == short


async def test_local_only_without_local_search_says_how_to_enable_it(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        local_responder([decision([("rigor", "x")])]),
        research=ResearchMode.ON_DEMAND,
        local_only=True,
    )
    await h.engine.step()
    await h.engine.research("x")
    system = h.texts("system")
    assert any("No web research" in t for t in system)  # the old local-only notice
    assert any("[research.local] enabled = true" in t for t in system)
    assert not any(r.tag.purpose.startswith("research") for r in h.requests())


def test_old_research_notes_load_as_claude() -> None:
    note = ResearchNote.model_validate({"focus": "f", "text": "t"})
    assert note.source == "claude"


@pytest.mark.parametrize(
    ("local", "require", "flag", "expected"),
    [
        (False, True, None, ResearchMode.OFF),  # M15: a notice, no research
        (True, True, None, ResearchMode.REQUIRED),
        (True, False, True, ResearchMode.AUTO),
        (True, False, None, ResearchMode.ON_DEMAND),
        (True, False, False, ResearchMode.OFF),
    ],
)
def test_resolve_mode_in_local_only(
    tmp_path: Path, local: bool, require: bool, flag: bool | None, expected: ResearchMode
) -> None:
    settings = Settings.model_validate(
        {"root": tmp_path, "research": {"local": {"enabled": local}}}
    )
    council = CouncilPreset(
        name="c",
        moderator="chair",
        members=["rigor"],
        protocol=ProtocolSettings(require_research=require),
    )
    assert resolve_mode(settings, council, flag, local_only=True) is expected


def test_resolve_mode_refuses_research_flag_without_local_search(tmp_path: Path) -> None:
    settings = Settings.model_validate({"root": tmp_path})
    council = CouncilPreset(name="c", moderator="chair", members=["rigor"])
    with pytest.raises(ResearchRequiredError, match=r"\[research.local\] enabled = true"):
        resolve_mode(settings, council, True, local_only=True)


async def test_lookups_drop_the_unsourced_flags_they_answer(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    unsourced = {"kind": "unsourced", "note": "Not in the facts."}
    update = {
        **decision([("rigor", "answer X")]),
        "research": ["ECB rate"],
        "flaws": [
            {"member": "rigor", "quote": "the deposit rate is 2.50%", **unsourced},
            {"member": "compounder", "quote": "gold is at $9,999/oz", **unsourced},
        ],
    }
    h = Harness(
        tmp_path,
        personas,
        local_responder([update]),
        research=ResearchMode.ON_DEMAND,
        local_only=True,
        config=LOCAL,
        search=FakeSearch(),
    )
    await h.engine.step()  # the opening
    await h.engine.step()  # round 1, then the update and its lookup
    rt = h.engine.meta.runtime
    assert [n.focus for n in rt.research] == ["ECB rate"]
    assert [f.member for f in rt.flaws if f.by == "moderator"] == ["compounder"]
    assert "Rigor's figure flagged as unsourced is in the new research; flag dropped" in (
        h.texts("system")
    )
    assert [(t.speaker, t.prompt) for t in rt.pending][-1] == ("rigor", "answer X")
    assert rt.pending[0].speaker == "compounder"


def test_facts_in_prompts_leave_out_missing_lines() -> None:
    note = ResearchNote(focus="pity", text=f"{MISSING}\n- another fact")
    assert without_missing(note.text) == f"{MISSING.splitlines()[0]}\n- another fact"
    text = context.facts_text([note])
    assert text is not None and "Missing" not in text and "another fact" in text
    assert missing_facts(note.text)  # the follow-up still reads the raw note


# --- [research] provider in normal sessions ---------------------------------------


def normal(
    tmp_path: Path,
    personas: dict[str, Persona],
    provider: str,
    *,
    search: FakeSearch | None = None,
    summaries: list[str] | None = None,
) -> Harness:
    """A normal (Claude) session with local search on and `provider` set."""
    return Harness(
        tmp_path,
        personas,
        local_responder([decision([("rigor", "x")])], [{"queries": ["a", "b"]}] * 2, summaries),
        research=ResearchMode.ON_DEMAND,
        fallback=True,
        config={"research": {"provider": provider, "local": {"enabled": True}}},
        search=search if search is not None else FakeSearch(),
    )


async def test_provider_local_searches_searxng_with_claude_writing_queries_and_summary(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    search = FakeSearch()
    h = normal(tmp_path, personas, "local", search=search)
    await h.engine.research("ECB rate")
    await h.engine.research("inflation")
    research = [r for r in h.claude.requests if r.tag.purpose.startswith("research")]
    assert [r.tag.purpose for r in research] == ["research_queries", "research"] * 2
    assert all(not r.tools for r in research)  # no web tools: SearXNG did the searching
    assert h.ollama.requests == []
    assert len(search.queries) == 4
    assert {n.source for n in h.engine.meta.runtime.research} == {"local"}
    consent = [t for t in h.texts("system") if "queries leave this PC" in t]
    assert len(consent) == 1  # at the first local search only


async def test_provider_local_follows_up_on_a_missing_line_in_a_normal_session(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        local_responder([decision([("rigor", "x")])], [{"queries": ["a"]}] * 2, [MISSING, SUMMARY]),
        research=ResearchMode.REQUIRED,
        config={"research": {"provider": "local", "local": {"enabled": True}}},
        search=FakeSearch(),
    )
    await h.engine.step()
    purposes = [r.tag.purpose for r in h.claude.requests]
    assert purposes[:5] == ["research_queries", "research"] * 2 + ["opening"]


async def test_provider_auto_uses_claude_then_searxng_once_claude_cannot_search(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    search = FakeSearch()
    h = normal(tmp_path, personas, "auto", search=search)
    await h.engine.research("ECB rate")
    assert [r.tag.purpose for r in h.claude.requests] == ["research"]
    assert h.claude.requests[0].tools  # Claude's own web search
    assert search.queries == []
    h.engine.claude_exhausted = True  # Chair now routes to the local fallback model
    await h.engine.research("inflation")
    assert [r.tag.purpose for r in h.ollama.requests] == ["research_queries", "research"]
    assert [n.source for n in h.engine.meta.runtime.research] == ["claude", "local"]
    assert not any("needs a Claude-backed moderator" in t for t in h.texts("system"))


@pytest.mark.parametrize("provider", ["claude", "auto", "local"])
async def test_local_only_sessions_ignore_the_provider_and_never_reach_claude(
    tmp_path: Path, personas: dict[str, Persona], provider: str
) -> None:
    config = {"research": {"provider": provider, "local": {"enabled": True}}}
    h = harness(tmp_path, personas, config=config, queries=[{"queries": ["a"]}] * 2)
    await h.engine.step()
    await h.engine.research("inflation")
    assert h.claude.requests == []
    assert {n.source for n in h.engine.meta.runtime.research} == {"local"}


@pytest.mark.parametrize("provider", ["local", "auto"])
def test_a_searxng_provider_needs_the_local_search_opt_in(provider: str) -> None:
    with pytest.raises(ValueError, match=r"\[research.local\] enabled = true"):
        Settings.model_validate({"research": {"provider": provider}})


def test_a_missing_line_copying_the_prompts_example_is_dropped() -> None:
    hit = SearchResult(title="t", url="https://a.example/x", snippet="Hard pity is at 80 pulls.")
    text = (
        "- Hard pity is at 80 pulls. https://a.example/x\n"
        f"- Missing: {prompts.LOCAL_MISSING_EXAMPLE}."
    )
    kept, dropped = check_summary(text, [hit])
    assert kept == "- Hard pity is at 80 pulls. https://a.example/x" and dropped == []
