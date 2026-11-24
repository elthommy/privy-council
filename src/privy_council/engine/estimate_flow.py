"""Estimate lines in forecast councils (M17): the task rule, the ledger, the split.

Councils with `estimate: probability` make every member end a turn with a parsable
probability line. The engine keeps those numbers per member, checks them by
arithmetic (engine/ledger.py), and hands them to the moderator and the synthesis
so neither has to re-read numbers from the prose. A devil's advocate argues a role
and gives no estimate.
"""

from __future__ import annotations

from privy_council.engine import context, ledger, prompts
from privy_council.engine.flaw_flow import FlawFlow
from privy_council.engine.notices import NOTICE_SPLIT
from privy_council.models.estimate import EstimateRecord
from privy_council.models.flaw import ESTIMATE_KINDS, REPAIR_KINDS, Flaw
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.persona import Persona, Role
from privy_council.models.session import PendingTurn


class EstimateFlow(FlawFlow):
    """Ask for, read and report the members' estimate lines."""

    def _estimating(self) -> bool:
        """Whether this council's members give estimate lines."""
        return self.council.protocol.estimate == "probability"

    def _devils(self) -> list[str]:
        """Return the slugs of the devil's advocates in the channel."""
        return [s for s, p in self.members.items() if p.meta.role is Role.DEVIL_ADVOCATE]

    def _estimate_task(self, persona: Persona) -> str:
        """Return the estimate rule to append to a member's task ("" outside forecasts)."""
        if not self._estimating():
            return ""
        devil = persona.meta.role is Role.DEVIL_ADVOCATE
        rule = prompts.estimate_rule(self.meta.language, devil, self.meta.runtime.split)
        return f"\n\n{rule}"

    def _turn_flaws(self, persona: Persona, prompt: context.Prompt, text: str) -> list[Flaw]:
        """Run the engine's checks, minus the figure check on the estimate lines."""
        if self._estimating():
            text = ledger.blank_lines(text)
        return super()._turn_flaws(persona, prompt, text)

    def _draft_flaws(self, persona: Persona, prompt: context.Prompt, text: str) -> list[Flaw]:
        """Add a draft's estimate-line flaws (M19), keeping the ledger as it is."""
        flaws = super()._draft_flaws(persona, prompt, text)
        return flaws + [f for f in self._read_estimate(persona, text)[1] if f.kind in REPAIR_KINDS]

    def _read_estimate(
        self, persona: Persona, text: str
    ) -> tuple[EstimateRecord | None, list[Flaw]]:
        """Parse a turn's estimate lines against the member's last ones, storing nothing."""
        if not self._estimating() or persona.meta.role is Role.DEVIL_ADVOCATE:
            return None, []
        rt = self.meta.runtime
        own = rt.estimates.get(persona.slug, [])
        return ledger.read_turn(
            text,
            member=persona.slug,
            round_=rt.round,
            previous=own[-1] if own else None,
            split_set=rt.split is not None,
            reframed=self._user_spoke_since(persona.slug),
        )

    def _user_spoke_since(self, slug: str) -> bool:
        """Whether the user wrote between the member's turn just posted and its previous one.

        An answer to the moderator often redefines the question, and members then
        move their numbers for that reason (2026-11-14 run).
        """
        own = 0
        for message in reversed(self.messages):
            if message.speaker == slug:
                own += 1
                if own == 2:
                    return False
            elif message.speaker == "user" and own == 1:
                return True
        return False

    def _track_estimate(self, persona: Persona, text: str, *, check: bool = True) -> None:
        """Read a member turn's estimate lines into the ledger; flag them if `check`.

        `/ask` answers are read but not flagged: they come outside the rounds.
        """
        record, flaws = self._read_estimate(persona, text)
        if record is not None:
            rt = self.meta.runtime
            rt.estimates[persona.slug] = [*rt.estimates.get(persona.slug, []), record]
        if check:
            self._record_flaws(flaws)

    def _set_split(self, decision: ModeratorDecision) -> None:
        """Keep the moderator's split event, the first time it sets one."""
        rt = self.meta.runtime
        event = " ".join((decision.split or "").split())[: prompts.SPLIT_MAX_CHARS]
        if not self._estimating() or rt.split is not None or not event:
            return
        rt.split = event
        self._notice(NOTICE_SPLIT.format(nick=self.moderator.name, event=event))

    def _round_estimate_flaws(self) -> list[Flaw]:
        """Return this round's engine flags on estimate lines, if the moderator sees them."""
        if not self.settings.engine.estimate_flaws_to_moderator:
            return []
        rt = self.meta.runtime
        return [
            f
            for f in rt.flaws
            if f.by == "engine" and f.round == rt.round and f.kind in ESTIMATE_KINDS
        ]

    def _estimates_block(self) -> str:
        """Return the estimate table for the moderator's dynamics ("" outside forecasts)."""
        if not self._estimating():
            return ""
        table = prompts.render_estimates(
            self.meta.runtime.estimates,
            list(self.members),
            self._devils(),
            self.nick,
            self.meta.runtime.split,
            self._round_estimate_flaws(),
        )
        return f"\n\n{table}"

    def _late_facts_turns(self) -> list[PendingTurn]:
        """Return a last turn for each member whose estimate predates the newest facts.

        Not after round 1 alone: a blind prompt shows no earlier turn to react to.
        """
        rt = self.meta.runtime
        on = self.settings.engine.late_facts_turns and self._estimating() and rt.round > 1
        if not on:
            return []
        devils = self._devils()
        turns = []
        for slug in self.members:
            if slug in devils or not rt.estimates.get(slug):
                continue
            spoke = max(m.ts for m in self.messages if m.speaker == slug)
            late = list(dict.fromkeys(n.focus for n in rt.research if n.ts > spoke))
            if late:
                facts = prompts.LATE_FACTS_SEPARATOR.join(late)
                prompt = prompts.LATE_FACTS_PROMPT[self.meta.language].format(facts=facts)
                turns.append(PendingTurn(speaker=slug, prompt=prompt))
        return turns

    def _synthesis_estimates(self) -> list[str]:
        """Return the synthesis's estimate table and devil's-advocate rule, if any."""
        if not self._estimating():
            return []
        records = self.meta.runtime.estimates
        devils = self._devils()
        voters = [s for s in self.members if s not in devils]
        last = [records[s][-1].estimate.value for s in voters if records.get(s)]
        table = prompts.synthesis_estimates(records, voters, self.nick, ledger.aggregate(last))
        rules = [prompts.SYNTHESIS_DEVIL.format(nick=self.nick(s)) for s in devils]
        return [p for p in (table, *rules) if p]
