"""Memory at close and the `/keep` and `/forget` commands (brief §10)."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from privy_council.backends.base import BackendError, CompletionResult
from privy_council.engine import context
from privy_council.engine.core import EngineCore
from privy_council.engine.moderation import resolve_speaker
from privy_council.engine.notices import (
    FORGET_ALL,
    FORGET_PERSONA,
    FORGET_SHARED,
    NOTICE_BUSY,
    NOTICE_FORGET_CONFIRM,
    NOTICE_FORGET_EMPTY,
    NOTICE_FORGET_USAGE,
    NOTICE_FORGOT,
    NOTICE_KEEP_NOTHING,
    NOTICE_KEEP_REWRITE_USAGE,
    NOTICE_KEEP_USAGE,
    NOTICE_MEMORY_FAILED,
    NOTICE_MEMORY_MERGE_BAD,
    NOTICE_MEMORY_MERGED,
    NOTICE_MEMORY_NOTHING,
    NOTICE_MEMORY_OFF,
    NOTICE_MEMORY_READ_ONLY,
    NOTICE_MEMORY_SAVED,
    NOTICE_MEMORY_SILENT,
    NOTICE_SHARED_FAILED,
    NOTICE_SHARED_KEPT,
    NOTICE_SHARED_MERGE_BAD,
    NOTICE_SHARED_NONE,
    NOTICE_SHARED_NONE_KEPT,
    NOTICE_SHARED_REWRITTEN,
    NOTICE_SHARED_SAVED,
    NOTICE_SHARED_UNQUOTED,
)
from privy_council.engine.scheduler import schedule
from privy_council.memory.distill import (
    MemoryOutputError,
    MemoryWriter,
    SessionDigest,
    parse_selection,
)
from privy_council.models.council import MemoryMode
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import MemoryProposed, MemoryReview, TurnStarted


class MemoryFlow(EngineCore):
    """Private notes, shared-fact proposals and their approval."""

    def estimate_close_calls(self) -> int:
        """Estimate Claude calls for `/close`: synthesis, memory notes, shared proposal."""

        def on_claude(persona: Persona) -> bool:
            """Return whether the persona currently routes to Claude."""
            return self.registry.route(persona, self.claude_exhausted).kind is (
                BackendKind.CLAUDE_CLI
            )

        writers = self._note_writers()
        shared = self._proposes_facts()
        return on_claude(self.moderator) * (1 + shared) + sum(on_claude(p) for p in writers)

    @property
    def memory_review_pending(self) -> bool:
        """Return whether shared-memory changes await the user's `/keep`."""
        return self.meta.runtime.memory_review.pending

    async def keep(self, arg: str) -> None:
        """Approve proposed shared facts (`1 3`, `all`, `none`) or a merged shared file."""
        review = self.meta.runtime.memory_review
        if not review.pending:
            self._notice(NOTICE_KEEP_NOTHING)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        if review.facts:
            chosen = parse_selection(arg, len(review.facts))
            if chosen is None:
                self._notice(NOTICE_KEEP_USAGE)
                return
            await self._keep_facts([review.facts[i] for i in chosen])
            return
        choice = parse_selection(arg, 1)
        if choice is None:
            self._notice(NOTICE_KEEP_REWRITE_USAGE)
            return
        if choice and review.rewrite is not None:
            self.memory.rewrite_shared(review.rewrite)
            self._notice(NOTICE_SHARED_REWRITTEN)
        else:
            self._notice(NOTICE_SHARED_KEPT)
        review.rewrite = None
        self._save()

    def forget(self, arg: str) -> None:
        """Wipe memory (`@nick`, `shared` or `all`); the same command twice confirms it."""
        target = self._forget_target(arg)
        if target is None:
            self._notice(NOTICE_FORGET_USAGE)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        what = self._forget_label(target)
        if self._forget_pending != target:
            self._forget_pending = target
            self._notice(NOTICE_FORGET_CONFIRM.format(what=what, arg=arg.strip()))
            return
        self._forget_pending = None
        if target == "all":
            existed = self.memory.reset_all()
        elif target == "shared":
            existed = self.memory.reset_shared()
        else:
            existed = self.memory.reset_private(target)
        self._notice((NOTICE_FORGOT if existed else NOTICE_FORGET_EMPTY).format(what=what))

    def _memory_writer(self) -> MemoryWriter:
        """Return a writer whose calls go through `_complete` (routing, fallback, ledger)."""

        async def complete(
            persona: Persona,
            prompt: context.Prompt,
            purpose: str,
            *,
            json_schema: dict[str, Any] | None = None,
            max_output: int | None = None,
        ) -> CompletionResult:
            """Call the persona's backend, dropping the fallback flag."""
            result, _ = await self._complete(
                persona, prompt, purpose, json_schema=json_schema, max_output=max_output
            )
            return result

        return MemoryWriter(self.memory, self.settings.memory, complete, self._budget)

    async def _remember(self, synthesis: str) -> None:
        """Distill memory at close: private notes per persona, shared-fact proposals.

        Claude calls run concurrently, Ollama ones in sequence grouped by model.
        Every failure is a notice; memory never blocks the close.
        """
        if self.council.memory is not MemoryMode.ON:
            off = self.council.memory is MemoryMode.OFF
            self._notice(NOTICE_MEMORY_OFF if off else NOTICE_MEMORY_READ_ONLY)
            return
        writer = self._memory_writer()
        digest = SessionDigest(
            council=self.council.name,
            topic=self.meta.topic,
            day=datetime.now().astimezone().date(),
            synthesis=synthesis,
            messages=list(self.messages),
            state_text=context.state_text(self.meta.runtime.debate_state, self.nick),
            nick_of=self.nick,
            lang=self.meta.language,
        )
        facts: list[str] = []
        writers = self._note_writers()
        for persona in self._speakers():
            if persona.meta.memory and persona not in writers:
                self._notice(NOTICE_MEMORY_SILENT.format(nick=persona.name))
        jobs: list[tuple[Persona, Callable[[], Awaitable[None]]]] = [
            (p, functools.partial(self._remember_notes, writer, p, digest)) for p in writers
        ]
        if self._proposes_facts():
            jobs.append(
                (self.moderator, functools.partial(self._propose_facts, writer, digest, facts))
            )
        plan = schedule(jobs, lambda job: self._route_placement(job[0]))

        async def sequential() -> None:
            """Run the local jobs one after another."""
            for _, job in plan.sequential:
                await job()

        await asyncio.gather(*(job() for _, job in plan.concurrent), sequential())
        if facts:
            await self._file_facts(writer, facts)

    async def _remember_notes(
        self, writer: MemoryWriter, persona: Persona, digest: SessionDigest
    ) -> None:
        """Write one persona's notes, merging them if they outgrew the cap."""
        nick = persona.name
        self._emit(TurnStarted(speaker=persona.slug, round=self.meta.runtime.round))
        try:
            count = await writer.write_notes(persona, digest)
            self._notice(
                NOTICE_MEMORY_SAVED.format(nick=nick, count=count)
                if count
                else NOTICE_MEMORY_NOTHING.format(nick=nick)
            )
            if writer.private_over_cap(persona):
                merged = await writer.consolidate(persona)
                cap = self.settings.memory.private_cap_bytes
                notice = NOTICE_MEMORY_MERGED if merged else NOTICE_MEMORY_MERGE_BAD
                self._notice(notice.format(nick=nick, cap=cap))
        except BackendError as exc:
            self._notice(NOTICE_MEMORY_FAILED.format(nick=nick, error=exc))

    async def _propose_facts(
        self, writer: MemoryWriter, digest: SessionDigest, out: list[str]
    ) -> None:
        """Collect the moderator's shared-fact proposals that pass the quote check."""
        try:
            facts, unquoted = await writer.propose_facts(self.moderator, digest)
        except (BackendError, MemoryOutputError) as exc:
            self._notice(NOTICE_SHARED_FAILED.format(error=exc))
            return
        if unquoted:
            self._notice(NOTICE_SHARED_UNQUOTED.format(count=unquoted))
        if not facts:
            self._notice(NOTICE_SHARED_NONE)
        out.extend(facts)

    async def _file_facts(self, writer: MemoryWriter, facts: list[str]) -> None:
        """Queue facts for `/keep`, or save them at once when approval is off."""
        if self.settings.memory.require_shared_approval:
            self.meta.runtime.memory_review = MemoryReview(facts=facts)
            self._save()  # offered once the session is closed
            return
        self.memory.append_shared(facts)
        self._notice(NOTICE_SHARED_SAVED.format(count=len(facts)))
        await self._merge_shared(writer)

    async def _keep_facts(self, kept: list[str]) -> None:
        """Save the approved facts, then propose a merge if user.md outgrew its cap."""
        review = self.meta.runtime.memory_review
        review.facts = []
        self.memory.append_shared(kept)
        self._save()
        if not kept:
            self._notice(NOTICE_SHARED_NONE_KEPT)
            return
        self._notice(NOTICE_SHARED_SAVED.format(count=len(kept)))
        self.busy = True
        try:
            await self._merge_shared(self._memory_writer())
        finally:
            self.busy = False

    async def _merge_shared(self, writer: MemoryWriter) -> None:
        """Consolidate user.md past its cap: proposed for `/keep`, or applied if approval is off."""
        if not writer.shared_over_cap():
            return
        try:
            merged = await writer.propose_rewrite(self.moderator)
        except BackendError as exc:
            self._notice(NOTICE_SHARED_FAILED.format(error=exc))
            return
        if merged is None:
            self._notice(NOTICE_SHARED_MERGE_BAD.format(cap=self.settings.memory.shared_cap_bytes))
            return
        if not self.settings.memory.require_shared_approval:
            self.memory.rewrite_shared(merged)
            self._notice(NOTICE_SHARED_REWRITTEN)
            return
        self.meta.runtime.memory_review.rewrite = merged
        self._save()
        self._emit(MemoryProposed(rewrite=merged))

    def _forget_target(self, arg: str) -> str | None:
        """Resolve a `/forget` argument to "all", "shared" or a persona slug."""
        key = arg.strip().lower()
        if key in ("all", "shared"):
            return key
        if not key:
            return None
        everyone = {p.slug: p for p in self._speakers()}
        slug = resolve_speaker(key, everyone)
        if slug is None and key.lstrip("@") in self.memory.personas_with_notes():
            slug = key.lstrip("@")  # a persona outside this council
        return slug

    def _forget_label(self, target: str) -> str:
        """Describe what a `/forget` target deletes."""
        if target == "all":
            return FORGET_ALL
        if target == "shared":
            return FORGET_SHARED
        return FORGET_PERSONA.format(nick=self.nick(target))

    def _note_writers(self) -> list[Persona]:
        """Return the memory personas that spoke this session.

        A persona with no messages has nothing of its own to remember; asked anyway,
        a local model invented the position it "argued" (live check, 2026-10-17).
        """
        if self.council.memory is not MemoryMode.ON:
            return []
        spoke = {m.speaker for m in self.messages}
        return [p for p in self._speakers() if p.meta.memory and p.slug in spoke]

    def _proposes_facts(self) -> bool:
        """Return whether the close will ask the moderator for shared facts."""
        return self.council.memory is MemoryMode.ON and self._user_spoke()

    def _user_spoke(self) -> bool:
        """Return whether the owner wrote anything in the session besides the question."""
        return any(m.speaker == "user" and m.author is None for m in self.messages)
