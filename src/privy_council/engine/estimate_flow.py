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
from privy_council.models.flaw import ESTIMATE_KINDS, Flaw
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.persona import Persona, Role


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

    def _check_turn(self, persona: Persona, prompt: context.Prompt, text: str) -> None:
        """Run the engine's checks, minus the figure check on the estimate lines."""
        if self._estimating():
            text = ledger.blank_lines(text)
        super()._check_turn(persona, prompt, text)

    def _track_estimate(self, persona: Persona, text: str, *, check: bool = True) -> None:
        """Read a member turn's estimate lines into the ledger; flag them if `check`.

        `/ask` answers are read but not flagged: they come outside the rounds.
        """
        if not self._estimating() or persona.meta.role is Role.DEVIL_ADVOCATE:
            return
        rt = self.meta.runtime
        own = rt.estimates.get(persona.slug, [])
        record, flaws = ledger.read_turn(
            text,
            member=persona.slug,
            round_=rt.round,
            previous=own[-1] if own else None,
            split_set=rt.split is not None,
        )
        if record is not None:
            rt.estimates[persona.slug] = [*own, record]
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
