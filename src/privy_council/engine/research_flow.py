"""Web research by the moderator: `/research` and its own searches."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError

from privy_council.backends.base import (
    BackendAuthError,
    BackendError,
    BackendUnavailableError,
    ChatMessage,
)
from privy_council.engine import context, prompts
from privy_council.engine.core import EngineCore
from privy_council.engine.local_research import (
    QUERIES_SCHEMA,
    NoSearchHits,
    check_summary,
    fallback_query,
    merge_results,
    missing_facts,
    parse_queries,
    render_results,
    shorter_queries,
)
from privy_council.engine.moderation import MAX_RESEARCH_REQUESTS, DecisionError, json_object
from privy_council.engine.notices import (
    NOTICE_ANSWER_RESEARCH,
    NOTICE_AUTO_RESEARCH,
    NOTICE_AUTO_RESEARCH_REPEAT,
    NOTICE_BUSY,
    NOTICE_CUT_OFF,
    NOTICE_LOCAL_DROPPED,
    NOTICE_LOCAL_FOLLOW_UP,
    NOTICE_LOCAL_NOTHING,
    NOTICE_LOCAL_QUERIES_FALLBACK,
    NOTICE_LOCAL_QUERY_FAILED,
    NOTICE_LOCAL_RETRY,
    NOTICE_LOCAL_SEARCH,
    NOTICE_LOCAL_SEARCHED,
    NOTICE_RESEARCH_FAILED,
    NOTICE_RESEARCH_NEEDS_CLAUDE,
    NOTICE_RESEARCH_NEEDS_LOCAL_SEARCH,
    NOTICE_RESEARCH_NO_SEARCH,
    NOTICE_RESEARCH_OFF,
    NOTICE_RESEARCH_READ,
    NOTICE_RESEARCH_SEARCHED,
    NOTICE_TERMS_RESEARCH,
)
from privy_council.engine.research import (
    WEB_TOOLS,
    ResearchMode,
    fetches,
    searches,
    searches_first,
)
from privy_council.engine.research_ledger import (
    earlier_queries,
    missing_items,
    not_findable,
    repeat_count,
    repeat_focus,
)
from privy_council.models.persona import BackendKind
from privy_council.models.research import AnswerLookup, LocalSearchRecord, SearchResult
from privy_council.models.session import ErrorEvent, Phase, ResearchAdded, ResearchNote, TurnStarted

log = logging.getLogger(__name__)

ANSWER_LOOKUP_SCHEMA: dict[str, Any] = AnswerLookup.model_json_schema()
ANSWER_LOOKUP_MAX_OUTPUT = 300  # one search focus
MAX_ANSWER_FOCUS_CHARS = 300
QUERIES_MAX_OUTPUT = 600  # a few short queries; the JSON never needs more


class ResearchFlow(EngineCore):
    """The moderator's web searches and their limits."""

    async def research(self, focus: str | None = None) -> None:
        """Have the moderator search the web; the facts are shared with every member."""
        if self._needs_topic():
            return
        problem = self._research_problem()
        if problem is not None:
            self._notice(problem)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        self.busy = True
        try:
            await self._research_safely(focus)
        except BackendAuthError as exc:
            # Outside a step nothing is left half done: just say why.
            self._emit(ErrorEvent(text=NOTICE_RESEARCH_FAILED.format(error=exc)))
        finally:
            self.busy = False

    def seed_research(self, notes: list[ResearchNote]) -> None:
        """Start from research done elsewhere (`council compare`): call before the opening.

        A session that has research of its own already keeps it.
        """
        rt = self.meta.runtime
        if rt.research or self.phase is not Phase.OPENING:
            return
        rt.research.extend(notes)
        self._save()
        if self.store:
            self.store.save_research(rt.research)
        for note in notes:
            self._emit(ResearchAdded(note=note))

    def _auto_research_ceiling(self, rounds: int) -> int:
        """Return the most searches the moderator could start on its own in `rounds`."""
        if self._research_unavailable() is not None:
            return 0
        research, cfg = self.meta.runtime.research, self.settings.research
        auto_left = cfg.auto_max_per_session - sum(n.auto for n in research)
        return max(min(auto_left, MAX_RESEARCH_REQUESTS * rounds), 0)

    async def _auto_research(self, focuses: list[str]) -> list[str]:
        """Run the moderator's own searches; return the ones relayed to the user instead."""
        relayed: list[str] = []
        for focus in focuses:
            research = self.meta.runtime.research
            if repeat_focus(focus, research):  # before the cap: a skip costs nothing
                count = repeat_count(focus, research)
                self._notice(
                    NOTICE_AUTO_RESEARCH_REPEAT.format(
                        nick=self.moderator.name, focus=focus, count=count
                    )
                )
                continue
            line = self._lookup_relay(focus)
            if line is not None:
                relayed.append(line)
                continue
            self._notice(NOTICE_AUTO_RESEARCH.format(nick=self.moderator.name, focus=focus))
            await self._research_safely(focus, auto=True)
        return relayed

    def _lookup_relay(self, focus: str) -> str | None:
        """Return the line asking the user about a search the moderator can't run, or None.

        Only the moderator's own searches are capped: past its allowance it asks, and
        the user's `/research` always runs.
        """
        cfg = self.settings.research
        if self._research_unavailable() is not None:
            return prompts.LOOKUP_ANSWER.format(focus=focus, reason=prompts.REASON_OFF)
        research = self.meta.runtime.research
        if sum(n.auto for n in research) >= cfg.auto_max_per_session:
            reason = prompts.REASON_AUTO_LIMIT.format(count=len(research))
        elif self.budget_warnings(1):
            reason = prompts.REASON_BUDGET
        else:
            return None
        self.lookups_for_user += 1  # only the user's /research can run it
        return prompts.LOOKUP_ASK.format(focus=focus, reason=reason)

    def _research_unavailable(self) -> Literal["off", "needs_claude", "needs_search"] | None:
        """Return why no search can run now, or None if one can."""
        if self.research_mode is ResearchMode.OFF:
            return "off"
        if self._local_search():
            return None if self.search is not None else "needs_search"
        route = self.registry.route(self.moderator, self.claude_exhausted)
        if route.kind is not BackendKind.CLAUDE_CLI:
            return "needs_claude"
        return None

    def _local_search(self) -> bool:
        """Whether the next search runs on SearXNG rather than Claude's web tools.

        Local-only sessions always search locally ([research] provider never brings
        Claude in: the registry refuses it anyway). Normal sessions follow `provider`;
        "auto" turns local when the Chair can't reach Claude (a usage limit, a local
        fallback), where "claude" would say research needs Claude.
        """
        provider = self.settings.research.provider
        if self.meta.local_only or provider == "local":
            return True
        claude = self.registry.route(self.moderator, self.claude_exhausted).kind
        return provider == "auto" and claude is not BackendKind.CLAUDE_CLI

    def _research_problem(self) -> str | None:
        """Return the notice explaining why research can't run now, or None if it can."""
        return {
            "off": NOTICE_RESEARCH_OFF,
            "needs_claude": NOTICE_RESEARCH_NEEDS_CLAUDE,
            "needs_search": NOTICE_RESEARCH_NEEDS_LOCAL_SEARCH,
            None: None,
        }[self._research_unavailable()]

    async def _first_research(self) -> None:
        """Search before the opening, so the framing and round 1 start from fresh facts.

        A required search raises instead (the step fails, `/step` retries), so the
        debate never starts without facts.
        """
        if not searches_first(self.research_mode) or self.meta.runtime.research:
            return
        problem = self._research_problem()
        if self.research_mode is ResearchMode.REQUIRED:
            if problem is not None:
                raise BackendUnavailableError(problem)
            await self._research_turn(None)
        elif problem is None:
            await self._research_safely(None)
        else:
            self._notice(problem)
            return
        await self._follow_up_research()

    async def _follow_up_research(self) -> None:
        """Search once more before round 1 for the facts the first note says are missing.

        Local search only: the Chair writes its first queries before it knows the
        game (or country, product), so "game pity 50/50 rules" finds nothing; the
        follow-up's queries are written with the first note in the prompt. Claude's
        own web search already searches until it is done. A failure never stops the
        debate.
        """
        rt = self.meta.runtime
        if not (rt.research and rt.research[-1].source == "local"):
            return
        if not self.settings.research.local.follow_up:
            return
        facts = missing_facts(rt.research[-1].text)
        if not facts:
            return
        self._notice(NOTICE_LOCAL_FOLLOW_UP.format(nick=self.moderator.name, facts=facts))
        await self._research_safely(facts)  # part of the first search: not the Chair's quota

    async def _terms_research(self, exchange: Sequence[str]) -> None:
        """Look up the measure the user's answer to the opening defines, before round 1.

        The first search runs before the user says what counts: on 2026-11-20 the index
        the answer named was already past its threshold, and round 1 priced it as open.
        `exchange` (the opening and the answer, as chat lines) goes in the prompt. Not
        the Chair's quota, like the first search; skipped when the budget is tight.
        """
        if self._research_unavailable() is not None or self.budget_warnings(1):
            return
        self._notice(NOTICE_TERMS_RESEARCH.format(nick=self.moderator.name))
        title = prompts.LABELS[self.meta.language]["research_terms"]
        await self._research_safely(prompts.RESEARCH_TERMS_FOCUS, title=title, history=exchange)

    async def _answer_research(self, exchange: Sequence[str]) -> None:
        """Look up what the user's answer changes, before the round that uses it.

        The moderator's update plans its searches before the user answers its
        questions, so a kind of event an answer adds was never searched (2026-11-24).
        One short call says what to look up, if anything. Not the Chair's quota;
        skipped when the budget is tight.
        """
        if self._research_unavailable() is not None or self.budget_warnings(2):
            return
        focus = await self._answer_focus(exchange)
        if not focus or repeat_focus(focus, self.meta.runtime.research):
            return
        self._notice(NOTICE_ANSWER_RESEARCH.format(nick=self.moderator.name, focus=focus))
        await self._research_safely(focus, history=exchange)

    async def _answer_focus(self, exchange: Sequence[str]) -> str:
        """Ask the moderator what an answer makes worth a search; "" if nothing."""
        task = prompts.ANSWER_LOOKUP_TASK.format(today=_today(), exchange="\n".join(exchange))
        prompt = self._research_prompt(prompts.ANSWER_LOOKUP_SYSTEM, task)
        try:
            result, _ = await self._complete(
                self.moderator,
                prompt,
                "answer_lookup",
                json_schema=ANSWER_LOOKUP_SCHEMA,
                timeout_s=self.settings.research.timeout_s,
                max_output=ANSWER_LOOKUP_MAX_OUTPUT,
            )
            lookup = AnswerLookup.model_validate(json_object(result))
        except BackendAuthError:
            raise
        except (BackendError, DecisionError, ValidationError) as exc:
            log.info("answer lookup failed: %s", type(exc).__name__)
            return ""
        if lookup.answer == "declines":
            return ""
        return " ".join(lookup.focus.split())[:MAX_ANSWER_FOCUS_CHARS]

    async def _research_safely(
        self,
        focus: str | None,
        *,
        auto: bool = False,
        title: str | None = None,
        history: Sequence[str] = (),
    ) -> None:
        """Run research; a failure (usage limit included) never stops the debate."""
        try:
            await self._research_turn(focus, auto=auto, title=title, history=history)
        except NoSearchHits as exc:
            self._notice(NOTICE_LOCAL_NOTHING.format(nick=self.moderator.name, queries=exc))
        except BackendAuthError:
            raise
        except BackendError as exc:
            self._emit(ErrorEvent(text=NOTICE_RESEARCH_FAILED.format(error=exc)))

    async def _research_turn(
        self,
        focus: str | None,
        *,
        auto: bool = False,
        title: str | None = None,
        history: Sequence[str] = (),
    ) -> None:
        """Moderator searches the web; the note joins every later prompt.

        With no focus asked for, the note is titled with a short label: the council's
        research_focus is an instruction, too long for a heading (2026-10-27 Discord).
        """
        title = title or focus or prompts.LABELS[self.meta.language]["research_opening"]
        focus = focus or self.council.research_focus.strip() or prompts.RESEARCH_DEFAULT_FOCUS
        self._emit(TurnStarted(speaker=self.moderator.slug, round=self.meta.runtime.round))
        if self._local_search():
            note = await self._local_research(focus, auto, history)
        else:
            note = await self._claude_research(focus, auto, history)
        self._add_note(note.model_copy(update={"focus": title}))

    def _add_note(self, note: ResearchNote) -> None:
        """Keep a research note: saved at once, shown, and in every later prompt."""
        rt = self.meta.runtime
        rt.research.append(note)
        self._save()
        if self.store:
            self.store.save_research(rt.research)
        self._emit(ResearchAdded(note=note))
        if note.truncated:
            self._notice(NOTICE_CUT_OFF.format(nick=self.moderator.name))

    def _research_prompt(
        self, system: str, task: str, *, facts: bool = True, history: Sequence[str] = ()
    ) -> context.Prompt:
        """Build a research prompt: no dossier, since the user's files could leak into queries."""
        return context.build_prompt(
            history=history,
            system=system,
            topic=self.meta.topic,
            facts=self._facts() if facts else None,  # earlier research: add, don't repeat
            facts_share=self.settings.research.context_share,
            task=task,
            budget_tokens=self._budget(self.moderator),
            state_text=context.state_text(self.meta.runtime.debate_state, self.nick),
        )

    async def _claude_research(
        self, focus: str, auto: bool, history: Sequence[str] = ()
    ) -> ResearchNote:
        """Search with the claude CLI's web tools."""
        cfg = self.settings.research
        read = list(dict.fromkeys(url for n in self.meta.runtime.research for url in n.fetched))
        read_before = prompts.RESEARCH_READ_BEFORE.format(urls="; ".join(read)) if read else ""
        task = prompts.RESEARCH_TASK.format(
            focus=focus,
            measure=self._opening_measure(),
            today=_today(),
            read_before=read_before + self._earlier_searches(),
        )
        prompt = self._research_prompt(prompts.RESEARCH_SYSTEM, task, history=history)
        result, _ = await self._complete(
            self.moderator,
            prompt,
            "research",
            tools=WEB_TOOLS,
            timeout_s=cfg.timeout_s,
            max_output=cfg.max_output_tokens,
        )
        queries, urls = searches(result), fetches(result)
        nick = self.moderator.name
        if queries:
            self._notice(NOTICE_RESEARCH_SEARCHED.format(nick=nick, queries="; ".join(queries)))
        # A WebFetch-only call still read the web: say which pages, not "without searching".
        if urls:
            self._notice(NOTICE_RESEARCH_READ.format(nick=nick, urls="; ".join(urls)))
        if not queries and not urls:
            self._notice(NOTICE_RESEARCH_NO_SEARCH.format(nick=nick))
        return ResearchNote(
            focus=focus,
            text=result.text.strip(),
            searches=queries,
            fetched=urls,
            truncated=result.truncated,
            auto=auto,
            missing=missing_items(result.text),
        )

    def _opening_measure(self) -> str:
        """Return the first search's lines: the question's measure and stated events."""
        if self.meta.runtime.research:
            return ""
        return prompts.RESEARCH_OPENING_MEASURE + prompts.RESEARCH_OPENING_PREMISE

    def _earlier_searches(self) -> str:
        """Return the research task's block on queries already run and facts not found."""
        notes = self.meta.runtime.research
        queries = earlier_queries(notes)
        if not queries:
            return ""
        missing = "; ".join(item for item, _ in not_findable(notes))
        block = prompts.RESEARCH_EARLIER.format(queries="; ".join(queries))
        return block + (prompts.RESEARCH_STILL_MISSING.format(missing=missing) if missing else "")

    async def _local_research(
        self, focus: str, auto: bool, history: Sequence[str] = ()
    ) -> ResearchNote:
        """Search with SearXNG: the moderator writes queries, then summarizes the hits.

        Bullets whose figures or URLs aren't in the hits are dropped (see
        `engine/local_research.py`), and the raw hits are kept for checking by hand.
        """
        rt = self.meta.runtime
        if not self.meta.local_only and not any(n.source == "local" for n in rt.research):
            self._notice(NOTICE_LOCAL_SEARCH)  # local-only sessions say it at the opening
        queries = await self._local_queries(focus, history)
        hits = await self._local_hits(queries)
        nick = self.moderator.name
        self._notice(
            NOTICE_LOCAL_SEARCHED.format(nick=nick, queries="; ".join(queries), count=len(hits))
        )
        if not hits and (retry := shorter_queries(queries)):
            self._notice(NOTICE_LOCAL_RETRY.format(nick=nick, queries="; ".join(retry)))
            queries, hits = [*queries, *retry], await self._local_hits(retry)
        if not hits:
            raise NoSearchHits("; ".join(queries))
        cfg = self.settings.research
        task = prompts.LOCAL_SUMMARY_TASK.format(
            focus=focus,
            today=_today(),
            results=render_results(hits),
            missing_example=prompts.LOCAL_MISSING_EXAMPLE,
            premise="" if rt.research else prompts.LOCAL_SUMMARY_PREMISE,
        )
        prompt = self._research_prompt(prompts.LOCAL_SUMMARY_SYSTEM, task, facts=False)
        result, _ = await self._complete(
            self.moderator,
            prompt,
            "research",
            timeout_s=cfg.timeout_s,
            max_output=cfg.max_output_tokens,
        )
        text, dropped = check_summary(result.text, hits, f"{self.meta.topic}\n{focus}")
        if dropped:
            self._notice(NOTICE_LOCAL_DROPPED.format(count=len(dropped)))
        if self.store:
            record = LocalSearchRecord(
                focus=focus, queries=queries, results=hits, summary=result.text, dropped=dropped
            )
            self.store.append_research_raw(record.model_dump_json())
        return ResearchNote(
            focus=focus,
            text=text or prompts.LOCAL_SUMMARY_EMPTY,
            searches=queries,
            truncated=result.truncated,
            auto=auto,
            source="local",
            missing=missing_items(text),
        )

    async def _local_queries(self, focus: str, history: Sequence[str] = ()) -> list[str]:
        """Ask the moderator for search queries, repairing once; else search the focus."""
        local = self.settings.research.local
        task = prompts.LOCAL_QUERIES_TASK.format(
            count=local.max_queries, focus=focus, measure=self._opening_measure(), today=_today()
        )
        prompt = self._research_prompt(prompts.LOCAL_QUERIES_SYSTEM, task, history=history)
        messages = [ChatMessage(role="user", content=prompt.user)]
        problem = ""
        for purpose in ("research_queries", "repair"):
            result, _ = await self._complete(
                self.moderator,
                prompt,
                purpose,
                messages=messages,
                json_schema=QUERIES_SCHEMA,
                timeout_s=self.settings.research.timeout_s,
                max_output=QUERIES_MAX_OUTPUT,
            )
            try:
                return parse_queries(result, local.max_queries)
            except DecisionError as exc:
                problem = str(exc)
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=result.text or "(empty)"),
                    ChatMessage(role="user", content=prompts.REPAIR_TASK.format(error=exc)),
                ]
        self._notice(
            NOTICE_LOCAL_QUERIES_FALLBACK.format(nick=self.moderator.name, problem=problem)
        )
        return [fallback_query(focus, self.meta.topic, self.council.research_focus.strip())]

    async def _local_hits(self, queries: list[str]) -> list[SearchResult]:
        """Run the queries one by one; a failed one is reported, all failing raises."""
        if self.search is None:
            raise BackendUnavailableError(NOTICE_RESEARCH_NEEDS_LOCAL_SEARCH)
        batches: list[list[SearchResult]] = []
        error: BackendError | None = None
        for query in queries:
            try:
                batches.append(await self.search.search(query, self.meta.language))
            except BackendError as exc:
                error = exc
                self._notice(NOTICE_LOCAL_QUERY_FAILED.format(query=query, error=exc), exc)
        if not batches and error is not None:
            raise error
        return merge_results(batches, self.settings.research.local.max_results)


def _today() -> str:
    """Return today's date (UTC) for the research prompts."""
    return f"{datetime.now(UTC):%Y-%m-%d}"
