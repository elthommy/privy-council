"""The debate protocol: opening, blind round, moderated cross-examination, synthesis.

`DebateEngine` is UI-agnostic. It emits events to listeners (the session store
first, so every event is on disk before any UI sees it) and is driven by
`step()`, `say()` and `close()`. Each `step()` runs one phase and pauses.
Progress is saved at turn boundaries, so a step interrupted by a Claude usage
limit can be retried without re-running the turns that already succeeded.

Shared plumbing lives in `core.py`; research, memory, the roster, `/vote` +
`/summary` and the dossier in their own flows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from privy_council.backends.base import (
    BackendError,
    BackendRateLimitError,
    ChatMessage,
    CompletionResult,
)
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine import context, dynamics, prompts
from privy_council.engine.dossier_flow import DossierFlow
from privy_council.engine.flaw_flow import FlawFlow
from privy_council.engine.memory_flow import MemoryFlow
from privy_council.engine.moderation import (
    cap_questions,
    decide,
    drop_answered,
    drop_asked,
    drop_quoted_facts,
    guest_question,
    keep_user_facts,
    moderator_schema,
    resolve_speaker,
    round_robin,
    to_pending,
    with_flaw_turns,
)
from privy_council.engine.notices import (
    ASK_LINE,
    NOTICE_ALREADY_CLOSED,
    NOTICE_BUSY,
    NOTICE_CLOSED,
    NOTICE_CONTINUED,
    NOTICE_ENDED,
    NOTICE_ENDED_EARLY,
    NOTICE_LANGUAGE,
    NOTICE_LOCAL_ONLY,
    NOTICE_LOCAL_ONLY_SEARCH,
    NOTICE_MODERATOR_FALLBACK,
    NOTICE_MODERATOR_PROBLEM,
    NOTICE_MORE,
    NOTICE_NEXT,
    NOTICE_NO_ANSWER,
    NOTICE_OPENING_FAILED,
    NOTICE_PAUSED,
    NOTICE_PAUSING,
    NOTICE_QUESTIONS_PAUSE,
    NOTICE_RATE_LIMIT,
    NOTICE_REACHED_MAX,
    NOTICE_REASON_LOOPING,
    NOTICE_REASON_MODERATOR,
    NOTICE_REASON_SETTLED,
    NOTICE_RESUMED_CLOSING,
    NOTICE_RUN_AT_END,
    NOTICE_SILENT_ADDED,
    NOTICE_STEP_FAILED,
    NOTICE_SYNTHESIS_MODEL_LIMITED,
    NOTICE_UNKNOWN_MEMBER,
)
from privy_council.engine.research_flow import ResearchFlow
from privy_council.engine.resume import check_resumable, resume_notice
from privy_council.engine.review_flow import ReviewFlow
from privy_council.engine.roster_flow import RosterFlow
from privy_council.engine.scheduler import schedule
from privy_council.memory.store import MemoryStore
from privy_council.models.council import CouncilPreset, RebuttalStyle
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.persona import BackendKind, Persona, Role
from privy_council.models.session import (
    BudgetWarning,
    ErrorEvent,
    MemoryProposed,
    PendingTurn,
    Phase,
    RoundStarted,
    SessionState,
    StateUpdated,
    TurnStarted,
)
from privy_council.storage.sessions import SessionStore
from privy_council.util.language import LANGUAGE_NAMES, LANGUAGES

MODERATOR_SCHEMA: dict[str, Any] = moderator_schema()
# The people's lines repeated in the update's task (shared chats): about 1k tokens.
PEOPLE_LINES_MAX_CHARS = 4000
# Output floor for the moderator's JSON updates. The cap counts streamed characters,
# and a full debate state reached Chair's 4000-token cap in several live sessions
# (cut off, then a repair call).
MODERATE_MIN_OUTPUT_TOKENS = 8000

PHASE_STATES = {
    Phase.OPENING: SessionState.OPENING,
    Phase.ROUND: SessionState.ROUND_RUNNING,
    Phase.SYNTHESIS: SessionState.SYNTHESIZING,
}


class DebateEngine(ResearchFlow, MemoryFlow, RosterFlow, ReviewFlow, DossierFlow, FlawFlow):
    """Runs one council session; see the module docstring."""

    @classmethod
    def resume(
        cls,
        *,
        settings: Settings,
        council: CouncilPreset,
        personas: Mapping[str, Persona],
        registry: BackendRegistry,
        store: SessionStore,
        memory: MemoryStore | None = None,
    ) -> DebateEngine:
        """Rebuild a saved session's engine (raises ResumeError); then `announce_resume()`."""
        meta = store.load_meta()
        check_resumable(meta, personas, registry.local_only)
        return cls(
            settings=settings,
            council=council,
            personas=personas,
            registry=registry,
            topic=meta.topic,
            store=store,
            memory=memory,
            restored=(meta, store.events()),
        )

    def announce_resume(self) -> None:
        """Tell listeners where a resumed session stands (call after subscribing).

        A session that died while writing memory is closed here: re-running the
        writers could file the same notes twice.
        """
        rt = self.meta.runtime
        if self.state is SessionState.CLOSING:
            rt.phase, rt.pending = Phase.DONE, []
            self._notice(NOTICE_RESUMED_CLOSING)
            self._set_state(SessionState.CLOSED)
            return
        self._save()
        self._notice(resume_notice(self.meta))
        if self.state is SessionState.CLOSED and rt.memory_review.pending:
            self._emit(
                MemoryProposed(facts=rt.memory_review.facts, rewrite=rt.memory_review.rewrite)
            )

    async def step(self) -> None:
        """Run the next phase (opening, one round + moderator update, or synthesis)."""
        if self._needs_topic():
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        self._pause_requested = False
        self._continue_after_early_stop()
        await self._step()

    async def run(self, rounds: int) -> int:
        """Run up to `rounds` debate rounds, stopping on pause, error or debate end.

        A pending opening runs first without counting. Returns completed rounds.
        """
        if self._needs_topic():
            return 0
        if self.busy:
            self._notice(NOTICE_BUSY)
            return 0
        self._pause_requested = False
        self._continue_after_early_stop()
        done = 0
        while done < rounds and not self._pause_requested:
            if self.phase in (Phase.SYNTHESIS, Phase.DONE):
                if not self.meta.runtime.early_stop:  # an early stop already said why
                    self._notice(NOTICE_RUN_AT_END)
                break
            counts = self.phase is Phase.ROUND
            self._asked_user = False
            await self._step()
            if self.state is not SessionState.PAUSED:
                break
            if counts and not self.meta.runtime.round_in_progress:
                done += 1
            if self._asked_user:  # questions mustn't scroll away under a /run
                self._notice(NOTICE_QUESTIONS_PAUSE.format(nick=self.moderator.name))
                break
        if self._pause_requested:
            self._notice(NOTICE_PAUSED)
        return done

    def request_pause(self) -> None:
        """Ask a running step to stop at the next turn boundary (never mid-call)."""
        if self.busy and not self._pause_requested:
            self._pause_requested = True
            self._notice(NOTICE_PAUSING)

    async def ask(self, who: str, question: str, author: str | None = None) -> None:
        """Put a direct question to one member: a single out-of-round turn.

        `author` names the person asking in a shared chat; None is the owner.
        """
        if self._needs_topic():
            return
        slug = resolve_speaker(who, self.members)
        if slug is None:
            self._notice(NOTICE_UNKNOWN_MEMBER.format(who=who))
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        persona = self.members[slug]
        self._post("user", ASK_LINE.format(nick=persona.name, question=question), author=author)
        self.busy = True
        try:
            await self._ask_turn(persona, question)
        except BackendError as exc:
            self._emit(ErrorEvent(text=NOTICE_NO_ANSWER.format(nick=persona.name, error=exc)))
        finally:
            self.busy = False

    def estimate_claude_calls(self, rounds: int) -> int:
        """Estimate Claude calls for `rounds` rounds, at most (a ceiling for warnings).

        Claude members + moderator each round, plus the searches the moderator may
        still start on its own (MAX_RESEARCH_REQUESTS per update, within its allowance).
        """
        speakers = [*self.members.values(), self.moderator]
        per_round = sum(
            self.registry.route(p, self.claude_exhausted).kind is BackendKind.CLAUDE_CLI
            for p in speakers
        )
        return per_round * rounds + self._auto_research_ceiling(rounds)

    async def _step(self) -> None:
        """Run one phase, marking the engine busy meanwhile."""
        self.busy = True
        try:
            await self._step_unguarded()
        finally:
            self.busy = False

    async def _step_unguarded(self) -> None:
        """Run the next phase and settle errors into ERROR_RECOVERABLE."""
        phase = self.phase
        if phase is Phase.DONE:
            self._notice(NOTICE_ALREADY_CLOSED)
            return
        self._set_state(PHASE_STATES[phase])
        try:
            await self._run_phase(phase)
        except BackendRateLimitError as exc:
            when = f", resets {exc.retry_at.astimezone():%a %H:%M}" if exc.retry_at else ""
            self._fail(NOTICE_RATE_LIMIT.format(when=when))
        except BackendError as exc:
            self._fail(NOTICE_STEP_FAILED.format(phase=phase.value.capitalize(), error=exc))

    async def close(self) -> None:
        """Skip to the synthesis and close the session."""
        if self.phase is Phase.DONE:
            self._notice(NOTICE_ALREADY_CLOSED)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        if self.machine.can(SessionState.SYNTHESIZING):
            rt = self.meta.runtime
            rt.phase, rt.early_stop, rt.pending = Phase.SYNTHESIS, False, []
        await self.step()  # if not (the opening hasn't run), this runs the opening

    @property
    def max_rounds(self) -> int:
        """Return this session's round limit: the council's, plus the `/more` rounds."""
        return self.council.protocol.max_rounds + self.meta.runtime.extra_rounds

    def more(self, rounds: int) -> None:
        """Add `rounds` past the limit, reopening a debate that reached it (`/more N`)."""
        if self.phase is Phase.DONE:
            self._notice(NOTICE_ALREADY_CLOSED)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        rt = self.meta.runtime
        rt.extra_rounds += rounds
        if rt.phase is Phase.SYNTHESIS and not rt.early_stop:  # an early stop resumes on /step
            # The last update's plan was dropped (and told the moderator it was the last).
            rt.phase, rt.stop_reason, rt.stalled_rounds = Phase.ROUND, None, 0
            rt.pending = round_robin(list(self.members), self.meta.language)
        self._save()
        self._notice(NOTICE_MORE.format(count=rounds, limit=self.max_rounds))

    def say(self, text: str, author: str | None = None) -> None:
        """Record a user interjection; the moderator integrates it at the next update.

        `author` names the person in a shared chat; None is the owner.
        """
        self._post("user", text, author=author)

    async def _run_phase(self, phase: Phase) -> None:
        """Dispatch one phase and settle the resulting state."""
        if phase is Phase.OPENING:
            await self._opening()
        elif phase is Phase.ROUND:
            await self._round()
        else:
            await self._synthesis()
            return
        self._set_state(SessionState.PAUSED)

    async def _opening(self) -> None:
        """Search first if the session researches; moderator frames; plan round 1."""
        rt = self.meta.runtime
        lang = self.meta.language
        other = next(code for code in LANGUAGES if code != lang)
        self._notice(NOTICE_LANGUAGE.format(name=LANGUAGE_NAMES[lang], other=other))
        if self.meta.local_only:
            searches = self._research_unavailable() is None
            self._notice(NOTICE_LOCAL_ONLY_SEARCH if searches else NOTICE_LOCAL_ONLY)
        await self._first_research()
        prompt = context.build_prompt(
            system=self._moderator_system(),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=prompts.OPENING_TASK,
            budget_tokens=self._budget(self.moderator),
            memory=self._memory(self.moderator),
            memory_share=self.settings.memory.context_share,
            history=context.render_history(
                context.blind_messages(self.messages, self.moderator.slug), self.nick
            ),
        )
        try:
            result, fallback = await self._complete(self.moderator, prompt, "opening", stream=True)
            self._post_result(self.moderator, result, fallback)
        except BackendRateLimitError:
            raise
        except BackendError as exc:
            self._notice(NOTICE_OPENING_FAILED.format(error=exc))
        rt.phase = Phase.ROUND
        rt.pending = [PendingTurn(speaker=s) for s in self.members]
        self._save()

    async def _round(self) -> None:
        """Run the pending turns of a round, then the moderator update."""
        rt = self.meta.runtime
        if not rt.round_in_progress:
            self._budget_check()
            rt.round += 1
            rt.round_in_progress = True
            self._emit(RoundStarted(round=rt.round, label=self._round_label()))
            self._save()
        if self._is_blind():
            await self._run_blind(list(rt.pending))
        else:
            for turn in list(rt.pending):
                if self._pause_requested:
                    break
                await self._member_turn(turn)
        if self._pause_requested:
            return  # the round stays in progress; the next step resumes it
        await self._moderate()
        rt.round_in_progress = False
        self._save()

    async def _run_blind(self, turns: list[PendingTurn]) -> None:
        """Run independent turns: Claude concurrently, Ollama sequentially by model.

        Devil's advocates go last and see the blind takes: with
        nothing to oppose, they used to guess the majority wrong.
        """
        advocates = [t for t in turns if self._is_advocate(t)]
        plan = schedule([t for t in turns if t not in advocates], self._placement)

        async def sequential() -> None:
            """Run the local turns one after another, grouped by model."""
            for turn in plan.sequential:
                if self._pause_requested:
                    return
                await self._member_turn(turn)

        outcomes = await asyncio.gather(
            *(self._member_turn(t) for t in plan.concurrent),
            sequential(),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        for turn in advocates:
            if self._pause_requested:
                return
            await self._member_turn(turn)

    def _is_advocate(self, turn: PendingTurn) -> bool:
        """Return whether a turn belongs to a devil's advocate."""
        return self.members[turn.speaker].meta.role is Role.DEVIL_ADVOCATE

    async def _member_turn(self, turn: PendingTurn) -> None:
        """Run one member turn; a failed turn is skipped, except on a usage limit."""
        persona = self.members[turn.speaker]
        rt = self.meta.runtime
        floor = prompts.floor_line(persona.name, turn.prompt) if turn.prompt else None
        last = next((m for m in reversed(self.messages) if m.speaker != "system"), None)
        # A re-run turn (after a crash or a usage limit) already has its floor line;
        # system lines such as the resume notice may sit in between.
        if floor and not (last and last.speaker == self.moderator.slug and last.text == floor):
            self._post(self.moderator.slug, floor)
        self._emit(TurnStarted(speaker=persona.slug, round=rt.round))
        prompt = self._member_prompt(persona, turn)
        try:
            result, fallback = await self._complete(persona, prompt, "turn", stream=True)
            self._post_result(persona, result, fallback)
            self._check_turn(persona, prompt, result.text)
        except BackendRateLimitError:
            raise
        except BackendError as exc:
            self._notice(NOTICE_NO_ANSWER.format(nick=persona.name, error=exc))
        rt.pending.remove(turn)
        self._save()

    def _member_prompt(self, persona: Persona, turn: PendingTurn) -> context.Prompt:
        """Build a member's prompt: blind rounds see only the moderator and the user."""
        rt = self.meta.runtime
        lang = self.meta.language
        blind = self._is_blind()
        if blind and self._is_advocate(turn):
            history = context.advocate_messages(
                self.messages, self.moderator.slug, rt.round, persona.slug
            )
            task = prompts.task(prompts.BLIND_DEVIL_ADVOCATE_TASK, lang)
        elif blind:
            history = context.blind_messages(self.messages, self.moderator.slug)
            task = prompts.task(prompts.BLIND_TASK, lang)
        else:
            history = context.recent_messages(
                self.messages, self.settings.engine.context_last_k_messages
            )
            task = (
                prompts.task(
                    prompts.CROSS_TASK,
                    lang,
                    round=rt.round,
                    moderator=self.moderator.name,
                    prompt=turn.prompt,
                    example=self._example_nick(persona),
                )
                if turn.prompt
                else prompts.task(prompts.OPEN_ROUND_TASK, lang, round=rt.round)
            )
        return context.build_prompt(
            system=self._member_system(persona),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=task,
            budget_tokens=self._budget(persona),
            memory=self._memory(persona),
            memory_share=self.settings.memory.context_share,
            state_text=None if blind else context.state_text(rt.debate_state, self.nick),
            history=context.render_history(history, self.nick),
        )

    async def _moderate(self) -> None:
        """Ask the moderator to update the debate state and plan the next round."""
        rt = self.meta.runtime
        last_round = rt.round >= self.max_rounds
        stats = dynamics.member_stats(self.messages, list(self.members), rt.round)
        request = self._moderate_request(stats, last_round)

        async def call(messages: list[ChatMessage], purpose: str) -> CompletionResult:
            """Send the moderator conversation in JSON mode."""
            result, _ = await self._complete(
                self.moderator,
                request,
                purpose,
                messages=messages,
                json_schema=self._moderator_schema(),
                max_output=max(self._max_output(self.moderator), MODERATE_MIN_OUTPUT_TOKENS),
            )
            return result

        decision, problems = await decide(call, request.user, self.members)
        for problem in problems:
            self._notice(NOTICE_MODERATOR_PROBLEM.format(problem=problem))
        if decision is not None:
            decision = self._settle_questions(decision)
        rt.moderated_upto = len(self.messages)
        if decision is not None:
            self._record_flaws(self._moderator_flaws(decision))
        planned = self._apply_decision(decision, stats, last_round)
        if decision is not None:
            # Once the debate ends, new facts would reach only the synthesis, with no
            # member left to challenge them (an early stop's research too: rare to resume).
            ending = rt.phase is Phase.SYNTHESIS
            searched = len(rt.research)
            relayed = [] if ending else await self._auto_research(decision.research)
            if self._drop_answered_flaws(rt.research[searched:]):
                rt.pending = self._with_flaw_turns(planned)
                self._notice(NOTICE_NEXT.format(speakers=self._speaker_list(rt.pending)))
            summon = self._summon_ask(decision.summon)
            self._forward_questions([*decision.questions_for_user, *relayed, *summon])
            rt.asked_questions.extend(decision.questions_for_user)

    def _moderate_request(
        self, stats: list[dynamics.MemberStats], last_round: bool
    ) -> context.Prompt:
        """Build the moderator's update prompt for the round that just ended."""
        rt = self.meta.runtime
        new_messages = [m for m in self.messages[rt.moderated_upto :] if m.speaker != "system"]
        return context.build_prompt(
            system=self._moderator_system(),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=self._moderate_task(stats, last_round),
            budget_tokens=self._budget(self.moderator),
            memory=self._memory(self.moderator),
            memory_share=self.settings.memory.context_share,
            state_text=context.state_text(rt.debate_state, self.nick),
            history=context.render_history(new_messages, self.nick),
            history_title=prompts.TITLE_ROUND_MESSAGES,
        )

    def _moderate_task(self, stats: list[dynamics.MemberStats], last_round: bool) -> str:
        """Return the moderator's update task, with the engine's debate dynamics."""
        rt = self.meta.runtime
        protocol = self.council.protocol
        stall_limit = protocol.stall_rounds if protocol.auto_stop_on_consensus else 0
        task = prompts.MODERATE_TASK.format(
            questions_cap=prompts.QUESTIONS_CAP_SHARED
            if self.guests
            else prompts.QUESTIONS_CAP_ONE,
            dynamics=prompts.render_dynamics(
                stats, self.nick, rt.stalled_rounds, stall_limit, self._round_figure_flaws()
            ),
            round=rt.round,
            max_rounds=self.max_rounds,
            stop_rule=prompts.stop_rule(
                protocol.auto_stop_on_consensus, last_round, first_round=rt.round == 1
            ),
            research_rule=(
                prompts.RESEARCH_RULE_LAST
                if last_round
                else prompts.RESEARCH_RULE_OFF
                if self._research_unavailable() in ("off", "needs_claude")
                else prompts.RESEARCH_RULE_ON
            ),
            summon_rule=prompts.summon_rule([] if last_round else self._summon_candidates()),
            asked_rule=prompts.asked_rule(rt.asked_questions),
            flaw_rule=prompts.flaw_rule(
                self.settings.engine.max_flaw_prompts_per_round, last_round
            ),
        )
        people = [
            prompts.chat_line(m, self.nick)
            for m in self.messages[rt.moderated_upto :]
            if m.speaker == "user"
        ]
        if not (self.guests and people):  # one person: the prompt stays as it was
            return task
        # Capped: the task is never trimmed, and a flood of lines would push out the rest.
        kept, _ = context.fit_messages(people, PEOPLE_LINES_MAX_CHARS)
        return f"{task}\n\n" + prompts.PEOPLE_LINES_TASK.format(lines="\n".join(kept))

    def _settle_questions(self, decision: ModeratorDecision) -> ModeratorDecision:
        """Drop questions asked before and cap each person's; in a shared chat, mind the guests.

        Answered open questions are dropped, and a guest question the moderator only
        wrote down is forwarded when it asked no guest anything (a local Chair asks
        the owner only, M16 checks).
        """
        rt = self.meta.runtime
        fresh = drop_asked(decision.questions_for_user, rt.asked_questions, self.user_name)
        questions = cap_questions(fresh, self.user_name, self.guests)
        if not self.guests:
            return decision.model_copy(update={"questions_for_user": questions})
        state = drop_answered(rt.debate_state, decision.debate_state, self._spoke_since())
        extra = guest_question(questions, state.open_questions, self.guests, rt.asked_questions)
        update = {"questions_for_user": questions + extra, "debate_state": state}
        return decision.model_copy(update=update)

    def _moderator_schema(self) -> dict[str, Any]:
        """Return the update's JSON schema: room for each person's questions."""
        if not self.guests:
            return MODERATOR_SCHEMA
        return moderator_schema(1 + len(self.guests))

    def _spoke_since(self) -> set[str]:
        """Return the people who wrote since the moderator's last update."""
        rt = self.meta.runtime
        return {
            m.author or self.user_name
            for m in self.messages[rt.moderated_upto :]
            if m.speaker == "user"
        }

    def _forward_questions(self, questions: list[str]) -> None:
        """Post the moderator's questions for the user as one highlighted message."""
        if not questions:
            return
        lines = [self._addressed(q) for q in questions]
        self._post(self.moderator.slug, "\n".join(lines), to_user=True)
        self._asked_user = True

    def _addressed(self, question: str) -> str:
        """Return a question for the user line, @owner first unless it names a person."""
        people = [self.user_name, *self.guests] if self.guests else []
        if any(question.startswith(f"@{name}") for name in people):
            return question
        return prompts.USER_QUESTION_LINE.format(nick=self.user_name, question=question)

    def _apply_decision(
        self,
        decision: ModeratorDecision | None,
        stats: list[dynamics.MemberStats],
        last_round: bool,
    ) -> list[PendingTurn]:
        """Store the new debate state, then plan the next round or stop.

        Returns the planned turns before the flaw turns, to re-plan if flags drop.
        """
        rt = self.meta.runtime
        if decision is None:
            self._notice(NOTICE_MODERATOR_FALLBACK)
        else:
            self._record_judgment(decision, stats)
        planned = self._plan(decision, stats)
        turns = self._with_flaw_turns(planned)
        reason = self._stop_reason(decision, last_round)
        if reason is not None:
            rt.phase, rt.stop_reason, rt.early_stop = Phase.SYNTHESIS, reason, not last_round
            rt.pending = [] if last_round else turns
            notice = NOTICE_ENDED if last_round else NOTICE_ENDED_EARLY
            self._notice(notice.format(reason=reason.rstrip(" .")))  # the line adds its own
            return planned
        rt.pending = turns
        self._notice(NOTICE_NEXT.format(speakers=self._speaker_list(turns)))
        return planned

    def _record_judgment(
        self, decision: ModeratorDecision, stats: list[dynamics.MemberStats]
    ) -> None:
        """Store the moderator's state and judgment, count stalled rounds, tell the UI.

        Round 1 never counts as stalled: nobody has challenged anybody yet.
        """
        rt = self.meta.runtime
        state = keep_user_facts(rt.debate_state, decision.debate_state)
        sources = "\n".join(t for t in (self._facts(), self._dossier()) if t)
        rt.debate_state = drop_quoted_facts(rt.debate_state, state, sources)
        rt.consensus = decision.consensus
        rt.new_arguments, rt.decision_open = decision.new_arguments, decision.decision_open
        progress = rt.round == 1 or not dynamics.stalled(
            decision.new_arguments, decision.decision_open, stats, decision.position_changes
        )
        rt.stalled_rounds = 0 if progress else rt.stalled_rounds + 1
        if self.store:
            self.store.save_debate_state(rt.debate_state)
        self._emit(
            StateUpdated(
                state=rt.debate_state,
                consensus=rt.consensus,
                stalled_rounds=rt.stalled_rounds,
                new_arguments=rt.new_arguments,
                decision_open=rt.decision_open,
            )
        )

    def _plan(
        self, decision: ModeratorDecision | None, stats: list[dynamics.MemberStats]
    ) -> list[PendingTurn]:
        """Return the next round's turns: the moderator's plan plus members left out too long."""
        turns = to_pending(decision.next_turns) if decision else []
        if not turns or self.council.protocol.rebuttal_style is RebuttalStyle.ROUND_ROBIN:
            return round_robin(list(self.members), self.meta.language)
        limit = self.council.protocol.max_silent_rounds
        for slug in dynamics.silent_members(stats, {t.speaker for t in turns}, limit):
            turns.append(
                PendingTurn(speaker=slug, prompt=prompts.SILENT_TURN_PROMPT[self.meta.language])
            )
            count = next(s.silent_rounds for s in stats if s.slug == slug)
            self._notice(NOTICE_SILENT_ADDED.format(nick=self.nick(slug), count=count))
        return turns

    def _with_flaw_turns(self, turns: list[PendingTurn]) -> list[PendingTurn]:
        """Return the planned turns with the round's flaw checks in front."""
        cap = self.settings.engine.max_flaw_prompts_per_round
        flaws = self._round_moderator_flaws()
        return with_flaw_turns(turns, flaws, cap, self.meta.language)

    def _stop_reason(self, decision: ModeratorDecision | None, last_round: bool) -> str | None:
        """Return why the debate ends after this update, or None to go on.

        The moderator stops on consensus; the engine on `stall_rounds` stalled
        rounds in a row (loop detection). Both need auto_stop_on_consensus, and
        neither stops the debate early after round 1, before any cross-examination.
        """
        protocol = self.council.protocol
        rt = self.meta.runtime
        early_ok = protocol.auto_stop_on_consensus and rt.round > 1
        if decision is not None and decision.stop and (early_ok or last_round):
            default = NOTICE_REACHED_MAX if last_round else NOTICE_REASON_MODERATOR
            return decision.stop_reason or default
        if early_ok and not last_round and 0 < protocol.stall_rounds <= rt.stalled_rounds:
            if decision is not None and not decision.decision_open:
                return NOTICE_REASON_SETTLED
            return NOTICE_REASON_LOOPING.format(count=rt.stalled_rounds)
        return NOTICE_REACHED_MAX if last_round else None

    def _speaker_list(self, turns: list[PendingTurn]) -> str:
        """Return the nicks of the planned speakers, comma-separated."""
        return ", ".join(self.nick(t.speaker) for t in turns)

    def _continue_after_early_stop(self) -> None:
        """Go back to debating after an early stop: the moderator's plan runs next."""
        rt = self.meta.runtime
        if not (rt.early_stop and rt.phase is Phase.SYNTHESIS):
            return
        rt.phase, rt.early_stop, rt.stalled_rounds, rt.stop_reason = Phase.ROUND, False, 0, None
        if not rt.pending:
            rt.pending = round_robin(list(self.members), self.meta.language)
        self._notice(NOTICE_CONTINUED.format(speakers=self._speaker_list(rt.pending)))
        self._save()

    async def _synthesis(self) -> None:
        """Moderator writes the final synthesis; the session closes."""
        rt = self.meta.runtime
        prompt = self._synthesis_prompt()
        for text in self.budget_warnings(self.estimate_close_calls()):
            self._emit(BudgetWarning(text=text))
        result, fallback = await self._synthesis_call(prompt)
        self._post_result(self.moderator, result, fallback)
        if self.store:
            self.store.save_synthesis(result.text)
        self._set_state(SessionState.CLOSING)
        await self._remember(result.text)
        rt.phase, rt.pending = Phase.DONE, []
        self._notice(NOTICE_CLOSED)
        self._set_state(SessionState.CLOSED)
        review = rt.memory_review
        if review.pending:
            self._emit(MemoryProposed(facts=review.facts, rewrite=review.rewrite))

    def _synthesis_prompt(self) -> context.Prompt:
        """Build the synthesis prompt: the task, questions asked, latest vote, errors caught."""
        rt = self.meta.runtime
        task = prompts.task(
            prompts.SYNTHESIS_TASK,
            self.meta.language,
            reason=rt.stop_reason or prompts.SYNTHESIS_REASON_DEFAULT,
        )
        asked = prompts.synthesis_asked(rt.asked_questions)
        vote = prompts.synthesis_vote(rt.last_vote, rt.last_vote_round, self.nick)
        # A member who answered a challenge either withdrew the error or kept it, and the
        # moderator's next update, which read the answer, flags it again if it still stands.
        # Super listed withdrawn errors even when shown the answer (2026-10-26). A later
        # flag with the same quote repeats the old turn, not the answer (2026-10-27).
        flagged = [f for f in rt.flaws if f.by == "moderator"]
        answered = {(f.member, f.excerpt) for f in flagged if self._answered(f)}
        flaws = [f for f in flagged if (f.member, f.excerpt) not in answered]
        errors = prompts.synthesis_errors(flaws, self.nick, self.meta.language)
        parts = (task, asked, vote, errors)
        return self._wrapup_prompt("\n\n".join(p for p in parts if p))

    async def _synthesis_call(self, prompt: context.Prompt) -> tuple[CompletionResult, bool]:
        """Write the synthesis with synthesis_model, falling back to the usual model."""
        override = self.moderator.meta.synthesis_model
        if override is None:
            return await self._complete(self.moderator, prompt, "synthesis", stream=True)
        try:
            return await self._complete(
                self.moderator, prompt, "synthesis", model=override, stream=True
            )
        except BackendRateLimitError:
            # Stronger models often have tighter limits; the usual model may still work.
            self._notice(
                NOTICE_SYNTHESIS_MODEL_LIMITED.format(
                    model=override, fallback=self.moderator.meta.model
                )
            )
            return await self._complete(self.moderator, prompt, "synthesis", stream=True)

    def _budget_check(self) -> None:
        """Warn when the coming round's Claude calls would cross a soft cap."""
        rt = self.meta.runtime
        speakers = [self.members[t.speaker] for t in rt.pending] + [self.moderator]
        estimate = sum(
            self.registry.route(p, self.claude_exhausted).kind is BackendKind.CLAUDE_CLI
            for p in speakers
        )
        for text in self.budget_warnings(estimate):
            self._emit(BudgetWarning(text=text))

    async def _ask_turn(self, persona: Persona, question: str) -> None:
        """Answer a direct user question with the debate state and recent messages."""
        self._emit(TurnStarted(speaker=persona.slug, round=self.meta.runtime.round))
        recent = context.recent_messages(
            self.messages, self.settings.engine.context_last_k_messages
        )
        prompt = context.build_prompt(
            system=self._member_system(persona),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=prompts.ASK_TASK.format(question=question),
            budget_tokens=self._budget(persona),
            memory=self._memory(persona),
            memory_share=self.settings.memory.context_share,
            state_text=context.state_text(self.meta.runtime.debate_state, self.nick),
            history=context.render_history(recent, self.nick),
        )
        result, fallback = await self._complete(persona, prompt, "ask", stream=True)
        self._post_result(persona, result, fallback)
