"""Taking stock mid-debate: `/vote` and `/summary` (brief §11).

Neither changes the protocol's position: a vote is a table for the user (the
latest one also feeds the final synthesis), a summary is one Chair message.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from privy_council.backends.base import BackendError, CompletionResult
from privy_council.engine import context, prompts
from privy_council.engine.core import EngineCore
from privy_council.engine.moderation import DecisionError, json_object
from privy_council.engine.notices import (
    NOTICE_BUSY,
    NOTICE_NOTHING_YET,
    NOTICE_SESSION_CLOSED,
    NOTICE_SUMMARY_DONE,
    NOTICE_SUMMARY_FAILED,
    NOTICE_VOTE_INVALID,
)
from privy_council.engine.scheduler import schedule
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import (
    BudgetWarning,
    ErrorEvent,
    Phase,
    TurnEnded,
    TurnStarted,
    VoteTaken,
)
from privy_council.models.vote import Vote, VoteReply


def vote_schema() -> dict[str, Any]:
    """Return the JSON schema sent with vote calls, with the length rule on `position`."""
    schema = VoteReply.model_json_schema()
    schema["properties"]["position"]["description"] = prompts.VOTE_POSITION_RULE
    return schema


VOTE_SCHEMA = vote_schema()


def parse_vote(speaker: str, result: CompletionResult, fallback: bool) -> Vote:
    """Turn a member's JSON reply into a vote row; an unusable reply becomes an error row."""
    try:
        reply = VoteReply.model_validate(json_object(result))
    except DecisionError as exc:
        return Vote(speaker=speaker, error=NOTICE_VOTE_INVALID.format(problem=exc))
    except ValidationError as exc:
        problem = "; ".join(e["msg"] for e in exc.errors())
        return Vote(speaker=speaker, error=NOTICE_VOTE_INVALID.format(problem=problem))
    position = " ".join(reply.position.split())
    return Vote(speaker=speaker, position=position, confidence=reply.confidence, fallback=fallback)


class ReviewFlow(EngineCore):
    """`/vote` and `/summary`: out-of-round calls that leave the debate where it was."""

    async def vote(self) -> None:
        """Ask every member for a final position and confidence; emit them as one table."""
        if not self._reviewable("vote"):
            return
        members = list(self.members.values())
        self._warn_budget(members)
        self.busy = True
        try:
            votes = await self._collect_votes(members)
        finally:
            self.busy = False
        rt = self.meta.runtime
        rt.last_vote, rt.last_vote_round = votes, rt.round
        self._save()
        self._emit(VoteTaken(round=rt.round, votes=votes))

    async def summary(self) -> None:
        """Have the moderator write an interim synthesis; the debate goes on."""
        if not self._reviewable("summarize"):
            return
        self._warn_budget([self.moderator])
        self.busy = True
        try:
            self._emit(TurnStarted(speaker=self.moderator.slug, round=self.meta.runtime.round))
            task = prompts.task(
                prompts.SUMMARY_TASK, self.meta.language, round=self.meta.runtime.round
            )
            result, fallback = await self._complete(
                self.moderator, self._wrapup_prompt(task), "summary", stream=True
            )
        except BackendError as exc:
            self._emit(ErrorEvent(text=NOTICE_SUMMARY_FAILED.format(error=exc)))
            return
        finally:
            self.busy = False
        self._post_result(self.moderator, result, fallback)
        self._notice(NOTICE_SUMMARY_DONE)

    def _wrapup_prompt(self, task: str) -> context.Prompt:
        """Build a moderator prompt over the whole debate (synthesis, summary)."""
        return context.build_prompt(
            system=self._moderator_system(),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=task,
            budget_tokens=self._budget(self.moderator),
            memory=self._memory(self.moderator),
            memory_share=self.settings.memory.context_share,
            state_text=context.state_text(self.meta.runtime.debate_state, self.nick),
            history=context.render_history(
                context.recent_messages(self.messages, len(self.messages)), self.nick
            ),
        )

    def _reviewable(self, what: str) -> bool:
        """Check that there is a debate to take stock of, saying why not."""
        if self._needs_topic():
            return False
        rt = self.meta.runtime
        if self.phase is Phase.DONE:
            self._notice(NOTICE_SESSION_CLOSED)
        elif self.busy:
            self._notice(NOTICE_BUSY)
        elif rt.round == 0 or (rt.round == 1 and rt.round_in_progress):
            self._notice(NOTICE_NOTHING_YET.format(what=what))
        else:
            return True
        return False

    def _warn_budget(self, personas: list[Persona]) -> None:
        """Emit budget warnings for the Claude calls these personas would make."""
        calls = sum(
            self.registry.route(p, self.claude_exhausted).kind is BackendKind.CLAUDE_CLI
            for p in personas
        )
        for text in self.budget_warnings(calls):
            self._emit(BudgetWarning(text=text))

    async def _collect_votes(self, members: list[Persona]) -> list[Vote]:
        """Run the vote calls: Claude concurrently, Ollama one model at a time."""
        plan = schedule(members, self._route_placement)
        votes: dict[str, Vote] = {}

        async def sequential() -> None:
            """Ask the local members one after another, grouped by model."""
            for persona in plan.sequential:
                votes[persona.slug] = await self._vote_of(persona)

        async def concurrent(persona: Persona) -> None:
            """Ask one Claude member."""
            votes[persona.slug] = await self._vote_of(persona)

        await asyncio.gather(*(concurrent(p) for p in plan.concurrent), sequential())
        return [votes[p.slug] for p in members]

    async def _vote_of(self, persona: Persona) -> Vote:
        """Ask one member for its vote; a failure becomes an error row, never an exception."""
        self._emit(TurnStarted(speaker=persona.slug, round=self.meta.runtime.round))
        prompt = context.build_prompt(
            system=self._member_system(persona),
            topic=self.meta.topic,
            facts=self._facts(),
            facts_share=self.settings.research.context_share,
            dossier=self._dossier(),
            dossier_share=self.settings.dossier.context_share,
            task=prompts.VOTE_TASK,
            budget_tokens=self._budget(persona),
            memory=self._memory(persona),
            memory_share=self.settings.memory.context_share,
            state_text=context.state_text(self.meta.runtime.debate_state, self.nick),
            history=context.render_history(
                context.recent_messages(
                    self.messages, self.settings.engine.context_last_k_messages
                ),
                self.nick,
            ),
        )
        try:
            result, fallback = await self._complete(
                persona, prompt, "vote", json_schema=VOTE_SCHEMA
            )
        except BackendError as exc:
            return Vote(speaker=persona.slug, error=str(exc))
        finally:
            self._emit(TurnEnded(speaker=persona.slug))
        return parse_vote(persona.slug, result, fallback)
