"""Running a comparison: each variant's debate in turn, with no user in the loop.

A variant debates the fixed number of rounds, takes a vote, then gets its
synthesis. The first variant's web research is handed to the others, and the
moderator doesn't search mid-debate, so every variant starts from the same facts.
A stop (Ctrl-C or the STOP file) lands on a turn boundary; `resume` goes on from there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum

from privy_council.backends.base import BackendError
from privy_council.backends.registry import BackendRegistry
from privy_council.compare import judge
from privy_council.compare.spec import apply_variant
from privy_council.compare.store import CompareState, CompareStore, VariantRun, VariantStatus
from privy_council.config import Settings
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.research import ResearchMode
from privy_council.engine.resume import ResumeError
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    ErrorEvent,
    Event,
    MessageEvent,
    Phase,
    ResearchAdded,
    RoundStarted,
    SessionState,
    TurnStarted,
    VoteTaken,
)
from privy_council.storage.sessions import SESSIONS_DIR, SYNTHESIS_MD, SessionStore

Progress = Callable[[str], None]
STOP_POLL_S = 2.0


class Outcome(StrEnum):
    """How a `CompareRunner.run()` ended."""

    DONE = "done"
    STOPPED = "stopped"
    FAILED = "failed"


def compare_settings(settings: Settings) -> Settings:
    """Return the settings variants run with: no searches started by the moderator."""
    research = settings.research.model_copy(update={"auto_max_per_session": 0})
    return settings.model_copy(update={"research": research})


def variant_research(state: CompareState, index: int) -> ResearchMode:
    """Return a variant's research mode: only the first one searches."""
    mode = ResearchMode(state.research)
    if index == 0 or mode is ResearchMode.OFF:
        return mode
    return ResearchMode.ON_DEMAND  # seeded with the shared facts instead; nobody types /research


class CompareRunner:
    """Runs the variants of one comparison that aren't done yet."""

    def __init__(
        self,
        *,
        settings: Settings,
        preset: CouncilPreset,
        personas: Mapping[str, Persona],
        registry: BackendRegistry,
        store: CompareStore,
        progress: Progress,
    ) -> None:
        """Prepare a run; nothing happens until `run()`."""
        self.settings = compare_settings(settings)
        self.preset = preset
        self.personas = dict(personas)
        self.registry = registry
        self.store = store
        self.state = store.load()
        self.progress = progress
        self.engine: DebateEngine | None = None
        self.stopping = False
        self._last_error: str | None = None
        self._started: dict[str, datetime] = {}

    def request_stop(self) -> None:
        """Stop at the next turn boundary; the call in flight finishes first."""
        if self.stopping:
            return
        self.stopping = True
        if self.engine is not None:
            self.engine.request_pause()
        self.progress("Stopping after the current turn…")

    async def run(self) -> Outcome:
        """Run the remaining variants; returns DONE once every variant is."""
        self.store.stop_file.unlink(missing_ok=True)  # left over from an earlier stop
        watcher = asyncio.create_task(self._watch_stop_file())
        try:
            for index, run in enumerate(self.state.variants):
                if run.status is VariantStatus.DONE:
                    continue
                if self.stopping:
                    return Outcome.STOPPED
                outcome = await self._run_variant(index, run)
                if outcome is not Outcome.DONE:
                    return outcome
            return Outcome.DONE
        finally:
            watcher.cancel()
            self.store.stop_file.unlink(missing_ok=True)

    async def _watch_stop_file(self) -> None:
        """Stop when the user creates the STOP file."""
        while not self.stopping:
            if self.store.stop_file.exists():
                self.request_stop()
                return
            await asyncio.sleep(STOP_POLL_S)

    async def _run_variant(self, index: int, run: VariantRun) -> Outcome:
        """Run (or resume) one variant's debate to its synthesis."""
        council, personas = apply_variant(
            self.preset, self.personas, run.variant, self.state.rounds
        )
        self._last_error = None
        self.progress(f"== {run.variant.name} ({run.variant.label()})")
        try:
            engine = self._engine(index, run, council, personas)
        except ResumeError as exc:
            return self._settle(run, Outcome.FAILED, str(exc))
        self.progress(f"   session {run.session}")
        self.engine = engine
        try:
            outcome = await self._drive(index, engine)
        finally:
            self.engine = None
        return self._settle(run, outcome, self._last_error)

    def _settle(self, run: VariantRun, outcome: Outcome, error: str | None) -> Outcome:
        """Record how a variant's run ended."""
        if outcome is Outcome.DONE:
            run.status, run.error = VariantStatus.DONE, None
        elif outcome is Outcome.FAILED:
            run.status, run.error = VariantStatus.FAILED, error
            self.progress(f"!! {run.variant.name} stopped on an error: {error}")
        self.store.save(self.state)
        return outcome

    def _engine(
        self,
        index: int,
        run: VariantRun,
        council: CouncilPreset,
        personas: dict[str, Persona],
    ) -> DebateEngine:
        """Open the variant's session, or create it (seeded with the shared research)."""
        nicks = {slug: p.name for slug, p in personas.items()}
        data_dir = self.settings.data_dir
        if run.session is not None:
            store = SessionStore.open(data_dir / SESSIONS_DIR / run.session, nicks)
            engine = DebateEngine.resume(
                settings=self.settings,
                council=council,
                personas=personas,
                registry=self.registry,
                store=store,
            )
            engine.subscribe(self._on_event(engine))
            engine.announce_resume()
        else:
            store = SessionStore.create(
                data_dir, council.name, self.state.topic, datetime.now(UTC), nicks
            )
            engine = DebateEngine(
                settings=self.settings,
                council=council,
                personas=personas,
                registry=self.registry,
                topic=self.state.topic,
                store=store,
                research=variant_research(self.state, index),
                lang=self.state.lang,
            )
            engine.subscribe(self._on_event(engine))
            if index > 0 and self.state.shared_research:
                engine.seed_research(self.state.shared_research)
        run.session, run.status = store.id, VariantStatus.RUNNING
        self.store.save(self.state)
        return engine

    async def _drive(self, index: int, engine: DebateEngine) -> Outcome:
        """Step the debate to its end; one failed step is retried once."""
        failures = 0
        while engine.phase is not Phase.DONE:
            if self.stopping:
                return Outcome.STOPPED
            if engine.phase is Phase.SYNTHESIS and not _voted(engine):
                await engine.vote()
                if not _voted(engine):  # no vote possible (nothing debated): synthesize anyway
                    engine.meta.runtime.last_vote_round = engine.meta.runtime.round
                continue
            await engine.step()
            if index == 0:
                self._share_research(engine)
            if engine.state is SessionState.ERROR_RECOVERABLE:
                failures += 1
                if failures > 1:
                    return Outcome.FAILED
                self.progress("   retrying the failed step once…")
            else:
                failures = 0
        return Outcome.DONE

    def _share_research(self, engine: DebateEngine) -> None:
        """Keep the first variant's research for the others, once its opening ran."""
        if self.state.shared_research is not None or engine.phase is Phase.OPENING:
            return
        self.state.shared_research = list(engine.meta.runtime.research)
        self.store.save(self.state)

    def _on_event(self, engine: DebateEngine) -> Callable[[Event], None]:
        """Return a listener printing one progress line per round, turn and error."""

        def listen(event: Event) -> None:
            """Print the event, if it is progress worth a line."""
            line = self._progress_line(engine, event)
            if line is not None:
                self.progress(line)

        return listen

    def _progress_line(self, engine: DebateEngine, event: Event) -> str | None:
        """Return the progress line for an event, or None."""
        if isinstance(event, TurnStarted):
            self._started[event.speaker] = event.ts
        elif isinstance(event, RoundStarted):
            return f"-- round {event.round} --"
        elif isinstance(event, ResearchAdded):
            return f"   research: {event.note.focus}"
        elif isinstance(event, VoteTaken):
            return "   vote taken"
        elif isinstance(event, ErrorEvent):
            self._last_error = event.text
            return f"!! {event.text}"
        elif isinstance(event, MessageEvent) and event.message.speaker in self._started:
            started = self._started.pop(event.message.speaker)
            seconds = (event.ts - started).total_seconds()
            return f"   {engine.nick(event.message.speaker)}: {seconds:.0f} s"
        return None


def _voted(engine: DebateEngine) -> bool:
    """Return whether the final vote was taken (after the last round)."""
    rt = engine.meta.runtime
    return rt.last_vote_round == rt.round and rt.round > 0


def syntheses(state: CompareState, settings: Settings) -> dict[str, str]:
    """Return variant name → synthesis, for the variants that have one."""
    found = {}
    for run in state.variants:
        if run.session is None:
            continue
        path = settings.data_dir / SESSIONS_DIR / run.session / SYNTHESIS_MD
        if path.is_file():
            found[run.variant.name] = path.read_text(encoding="utf-8")
    return found


async def run_judge(store: CompareStore, settings: Settings, registry: BackendRegistry) -> str:
    """Have Claude compare the syntheses once; the verdict is saved (raises BackendError)."""
    state = store.load()
    texts = syntheses(state, settings)
    if len(texts) < 2:
        raise BackendError("the judge needs at least two finished variants")
    request = judge.judge_request(
        compare_id=store.id,
        topic=state.topic,
        research=state.shared_research or [],
        syntheses=texts,
        language=state.lang or "en",
        model=settings.claude.default_model,
    )
    verdict = await judge.judge(registry.backend(BackendKind.CLAUDE_CLI), request)
    store.save_judge(verdict)
    state.judged = True
    store.save(state)
    return verdict
