"""The document reader (M21): fetch a public page, keep the passages that match, summarize.

A web fetch by Claude returns about the first 100k characters of a page, so a
filing's debt note further down was never read (2026-10-30, 2026-11-14). Here our
own code fetches the page behind a URL guard (`search/fetch.py`), cuts it into
passages, ranks them with BM25 against what to look for, and the moderator
summarizes the best ones in one call with no tools. Bullets the passages don't back
are dropped, as for local search.

The moderator may only ask for a URL already in the fresh facts; the owner's `/read`
takes any URL. Discord has no `/read`.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from privy_council.backends.base import BackendAuthError, BackendError, BackendRateLimitError
from privy_council.engine import prompts
from privy_council.engine.facts import blocks, on_domains
from privy_council.engine.local_research import check_summary, missing_line, unbacked_numbers
from privy_council.engine.moderation import MAX_DOCUMENT_REQUESTS, DecisionError, json_object
from privy_council.engine.notices import (
    NOTICE_BUSY,
    NOTICE_DOCUMENT_DROPPED,
    NOTICE_DOCUMENT_FAILED,
    NOTICE_DOCUMENT_NEEDS_LOCAL,
    NOTICE_DOCUMENT_NO_MATCH,
    NOTICE_DOCUMENT_NOT_IN_FACTS,
    NOTICE_DOCUMENT_OFF,
    NOTICE_DOCUMENT_READ,
    NOTICE_DOCUMENT_READING,
    NOTICE_DOCUMENT_SITE_REFUSES,
    NOTICE_DOCUMENT_USAGE,
)
from privy_council.engine.research_flow import ResearchFlow, _today
from privy_council.engine.research_ledger import (
    cited_url,
    missing_items,
    not_findable,
    page_for_repeat,
    readable_pages,
    repeat_focus,
)
from privy_council.models.moderator import DocumentRequest
from privy_council.models.persona import BackendKind
from privy_council.models.research import DocumentRecord, ReadPick, SearchResult
from privy_council.models.session import ErrorEvent, ResearchNote, TurnStarted
from privy_council.search import bm25
from privy_council.search.extract import chunks, html_blocks, text_blocks
from privy_council.search.fetch import FetchedPage, fetch_page

log = logging.getLogger(__name__)

READ_PICK_SCHEMA: dict[str, Any] = ReadPick.model_json_schema()
READ_PICK_MAX_OUTPUT = 300  # a page number and a few words
MAX_LOOK_FOR_CHARS = 200


class DocumentFlow(ResearchFlow):
    """Reading one page for a figure the web search couldn't reach."""

    async def read(self, arg: str) -> None:
        """The owner's `/read <url> <what to look for>`: any public page, through the guard."""
        if self._needs_topic():
            return
        url, _, look_for = arg.strip().partition(" ")
        if not url or not look_for.strip():
            self._notice(NOTICE_DOCUMENT_USAGE)
            return
        problem = self._document_problem()
        if problem is not None:
            self._notice(problem)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        self.busy = True
        try:
            await self._read_safely(url, look_for.strip(), auto=False)
        except BackendAuthError as exc:
            self._emit(ErrorEvent(text=NOTICE_DOCUMENT_FAILED.format(error=exc)))
        finally:
            self.busy = False

    async def _auto_documents(self, requests: list[DocumentRequest]) -> list[str]:
        """Read the pages the moderator asked for; return the reads relayed to the user."""
        relayed: list[str] = []
        for request in requests[:MAX_DOCUMENT_REQUESTS]:
            if self._document_problem() is not None:
                continue
            nick = self.moderator.name
            if not cited_url(request.url, self.meta.runtime.research):
                self._notice(NOTICE_DOCUMENT_NOT_IN_FACTS.format(nick=nick, url=request.url))
                continue
            if on_domains(request.url, self.settings.research.documents.unreadable_domains):
                self._notice(NOTICE_DOCUMENT_SITE_REFUSES.format(nick=nick, url=request.url))
                continue
            reason = self._read_cap_reason()
            if reason is not None:
                relayed.append(
                    prompts.READ_ASK.format(
                        url=request.url, look_for=request.look_for, reason=reason
                    )
                )
                self.lookups_for_user += 1
                continue
            await self._read_safely(request.url, request.look_for, auto=True)
        return relayed

    def _reads_for_repeats(self, focuses: list[str]) -> list[DocumentRequest]:
        """Return a page read for the first lookup that will be skipped as a repeat.

        Call it before the lookups run. The search was tried twice with nothing new;
        an official page those searches cite, never read in full, is what's left.
        """
        if not self._documents_on():
            return []
        research = self.meta.runtime.research
        cfg = self.settings.research
        for focus in focuses:
            if not repeat_focus(focus, research):
                continue
            url = page_for_repeat(
                focus, research, cfg.primary_domains, cfg.documents.unreadable_domains
            )
            if url is not None:
                return [DocumentRequest(url=url, look_for=focus)]
        return []

    async def _picked_reads(self) -> list[DocumentRequest]:
        """Ask a local moderator, in one short call, whether a cited page holds a missing figure.

        Only for an update that asked for no read: a local Chair never used
        `documents` (2026-11-19), a Claude Chair does when it fits (2026-11-16 replays).
        Any failure means no read: the debate goes on as the update planned.
        """
        if not self._documents_on() or not self._local_moderator():
            return []
        research = self.meta.runtime.research
        cfg = self.settings.research
        pages = [
            url
            for url, read in readable_pages(
                research, cfg.primary_domains, cfg.documents.unreadable_domains
            )
            if not read
        ]
        missing = [item for item, _ in not_findable(research)]
        if not pages or not missing:
            return []
        task = prompts.READ_PICK_TASK.format(
            missing="\n".join(prompts.READ_PICK_MISSING.format(item=m) for m in missing),
            pages="\n".join(
                prompts.READ_PICK_PAGE.format(index=i, url=u) for i, u in enumerate(pages, 1)
            ),
        )
        prompt = self._research_prompt(prompts.READ_PICK_SYSTEM, task)
        try:
            result, _ = await self._complete(
                self.moderator,
                prompt,
                "read_pick",
                json_schema=READ_PICK_SCHEMA,
                timeout_s=cfg.timeout_s,
                max_output=READ_PICK_MAX_OUTPUT,
            )
            pick = ReadPick.model_validate(json_object(result))
        except (BackendAuthError, BackendRateLimitError):
            raise
        except (BackendError, DecisionError, ValidationError) as exc:
            log.info("read pick failed: %s", exc)
            return []
        log.info("read pick: page %d of %d", pick.page, len(pages))
        if not 1 <= pick.page <= len(pages):
            return []
        look_for = " ".join(pick.look_for.split())[:MAX_LOOK_FOR_CHARS] or missing[0]
        return [DocumentRequest(url=pages[pick.page - 1], look_for=look_for)]

    def _local_moderator(self) -> bool:
        """Whether the moderator runs on a local model now (a local Chair or a fallback)."""
        route = self.registry.route(self.moderator, self.claude_exhausted)
        return route.kind is BackendKind.OLLAMA

    def _readable_block(self) -> str:
        """Return the official pages the moderator may read, for its task ("" if none)."""
        if not self._documents_on():
            return ""
        cfg = self.settings.research
        pages = readable_pages(
            self.meta.runtime.research, cfg.primary_domains, cfg.documents.unreadable_domains
        )
        block = prompts.render_readable_pages(pages)
        return f"\n\n{block}" if block else ""

    def _document_problem(self) -> str | None:
        """Return why no page can be read now, or None if one can."""
        cfg = self.settings.research
        if not cfg.documents.enabled:
            return NOTICE_DOCUMENT_OFF
        if self.meta.local_only and not cfg.local.enabled:
            return NOTICE_DOCUMENT_NEEDS_LOCAL
        if self._research_unavailable() == "off":
            return self._research_problem()
        return None

    def _documents_on(self) -> bool:
        """Whether the moderator may ask for a page read (for its task's rule)."""
        return self._document_problem() is None

    def _read_cap_reason(self) -> str | None:
        """Return why the moderator can't read on its own now (its allowance), or None."""
        research = self.meta.runtime.research
        if sum(n.auto for n in research) >= self.settings.research.auto_max_per_session:
            return prompts.REASON_AUTO_LIMIT.format(count=len(research))
        if self.budget_warnings(1):
            return prompts.REASON_BUDGET
        return None

    async def _read_safely(self, url: str, look_for: str, *, auto: bool) -> None:
        """Read a page; a failure (refused, unreachable, timeout) never stops the debate."""
        try:
            await self._read_document(url, look_for, auto=auto)
        except BackendAuthError:
            raise
        except BackendError as exc:
            self._emit(ErrorEvent(text=NOTICE_DOCUMENT_FAILED.format(error=exc)))

    async def _read_document(self, url: str, look_for: str, *, auto: bool) -> None:
        """Fetch, pick the passages, summarize them, and add the note."""
        nick = self.moderator.name
        self._emit(TurnStarted(speaker=self.moderator.slug, round=self.meta.runtime.round))
        self._notice(NOTICE_DOCUMENT_READING.format(nick=nick, url=url, look_for=look_for))
        page = await fetch_page(url, self.settings.research.documents)
        passages = self._passages(page, look_for)
        if not passages:
            self._notice(NOTICE_DOCUMENT_NO_MATCH.format(nick=nick, url=url, look_for=look_for))
            return
        chars = sum(len(p) for p in passages)
        self._notice(
            NOTICE_DOCUMENT_READ.format(nick=nick, url=url, count=len(passages), chars=chars)
        )
        text = await self._summarize_page(url, page, look_for, passages)
        self._add_note(
            ResearchNote(
                focus=look_for,
                text=text or prompts.DOCUMENT_SUMMARY_EMPTY,
                fetched=[url],
                auto=auto,
                source="document",
                missing=missing_items(text),
            )
        )

    def _passages(self, page: FetchedPage, look_for: str) -> list[str]:
        """Return the page's passages that best match `look_for`, best first."""
        cfg = self.settings.research.documents
        html = page.content_type != "text/plain"
        blocks = html_blocks(page.text) if html else text_blocks(page.text)
        cut = chunks(blocks)
        rendered = [c.render() for c in cut]
        labels = [c.label() for c in cut]
        picked = bm25.top(look_for, rendered, cfg.passages, cfg.passage_chars, labels)
        return [rendered[i][: cfg.passage_chars] for i in picked]

    async def _summarize_page(
        self, url: str, page: FetchedPage, look_for: str, passages: list[str]
    ) -> str:
        """Summarize the passages in one call with no tools; drop unbacked bullets."""
        numbered = "\n\n".join(
            prompts.DOCUMENT_PASSAGE.format(index=i, text=p) for i, p in enumerate(passages, 1)
        )
        task = prompts.DOCUMENT_TASK.format(
            url=url, look_for=look_for, today=_today(), passages=numbered
        )
        prompt = self._research_prompt(prompts.RESEARCH_SYSTEM, task, facts=False)
        cfg = self.settings.research
        result, _ = await self._complete(
            self.moderator,
            prompt,
            "read",
            timeout_s=cfg.timeout_s,
            max_output=cfg.max_output_tokens,
        )
        hits = [SearchResult(title="", url=url, snippet=p) for p in passages]
        hits.append(SearchResult(title="", url=page.url))  # after redirects
        context = f"{self.meta.topic}\n{look_for}"
        text, dropped = check_summary(result.text, hits, context)
        text, dropped = _strict(text, "\n".join([context, url, *passages]), dropped)
        if dropped:
            self._notice(NOTICE_DOCUMENT_DROPPED.format(count=len(dropped)))
        if self.store:
            record = DocumentRecord(
                url=page.url,
                look_for=look_for,
                passages=passages,
                summary=result.text,
                dropped=dropped,
            )
            self.store.append_document_raw(record.model_dump_json())
        return text


def _strict(text: str, sources: str, dropped: list[str]) -> tuple[str, list[str]]:
    """Drop the bullets with a number the passages don't hold (bare numbers included).

    A "Missing:" line is never dropped (it may name a period the page lacks), and
    it is judged apart from the bullet it follows, which `blocks` would join it to.
    """
    kept: list[str] = []
    missing: list[str] = []
    for block in blocks(text):
        lines = block.split("\n")
        missing += [line for line in lines if missing_line(line)]
        fact = "\n".join(line for line in lines if not missing_line(line))
        if fact.strip():
            (dropped if unbacked_numbers(fact, sources) else kept).append(fact)
    return "\n".join([*kept, *missing]), dropped
