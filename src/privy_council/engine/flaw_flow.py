"""Flaws in member turns (M11): the engine's checks, the moderator's flaw reports,
and the floor prompts that make members check them.

Flag only: a flawed turn's text is never changed, on disk or on screen.
"""

from __future__ import annotations

from collections.abc import Sequence

from privy_council.engine import checks, context, prompts
from privy_council.engine.core import EngineCore
from privy_council.engine.notices import NOTICE_FLAW_ANSWERED
from privy_council.models.flaw import FIGURE_KINDS, Flaw, FlawKind
from privy_council.models.moderator import ModeratorDecision, ReportedFlaw
from privy_council.models.persona import Persona, Role
from privy_council.models.session import FlawsFound, ResearchNote


class FlawFlow(EngineCore):
    """Find, record and act on flaws in member turns."""

    def _addressable(self) -> list[str]:
        """Return the nicks and slugs a member may @-address: the channel and the user."""
        people = [self.moderator, *self.members.values()]
        names = [self.user_name, "user", *self.guests]
        return [*names, *(p.name for p in people), *(p.slug for p in people)]

    def _devil_line(self, persona: Persona) -> str | None:
        """Return the line a devil's advocate owes this turn (the blind round), or None."""
        if self._is_blind() and persona.meta.role is Role.DEVIL_ADVOCATE:
            return prompts.LABELS[self.meta.language]["devil"]
        return None

    def _check_turn(self, persona: Persona, prompt: context.Prompt, text: str) -> None:
        """Run the engine's checks on a member turn against the prompt it was given."""
        sources = context.sources_text(prompt.system, prompt.user)
        given = context.given_text(prompt.user)
        self._turn_sources[persona.slug, self.meta.runtime.round] = (sources, given)
        flaws = checks.check_turn(
            text,
            member=persona.slug,
            nick=persona.name,
            sources=sources,
            given=given,
            known=self._addressable(),
            devil_line=self._devil_line(persona),
            round_=self.meta.runtime.round,
        )
        self._record_flaws(flaws)

    def _record_flaws(self, flaws: list[Flaw]) -> None:
        """Keep flaws in the session and log them in the transcript."""
        if not flaws:
            return
        rt = self.meta.runtime
        rt.flaws.extend(flaws)
        self._emit(FlawsFound(round=rt.round, flaws=flaws))

    def _moderator_flaws(self, decision: ModeratorDecision) -> list[Flaw]:
        """Turn the moderator's flaw reports for the round that just ended into flaws."""
        rnd = self.meta.runtime.round
        return [
            Flaw(
                member=f.member,
                kind=FlawKind(f.kind),
                excerpt=f.quote,
                note=f.note,
                round=rnd,
                by="moderator",
            )
            for f in decision.flaws
            if not self._withdrawn(f, rnd)
        ]

    def _withdrawn(self, flaw: ReportedFlaw, rnd: int) -> bool:
        """Whether a moderator's flag doesn't hold by its own note or the engine's checks.

        Unsourced: the figures are given or the member's own arithmetic. Other kinds:
        the note says the figures hold ("shown only roughly", 2026-10-31 debate: the
        member then "corrected" a right figure) and disputes nothing. Arithmetic:
        the note says it holds ("35 x 160 = 5,600 is correct"); a note that also calls
        something wrong needs the quote's figures cleared too (2026-10-27 Discord debate).
        """
        if flaw.kind == FlawKind.UNSOURCED.value:
            return self._cleared(flaw.member, flaw.quote, rnd)
        if not checks.note_confirms(flaw.note):
            return False
        if flaw.kind != FlawKind.ARITHMETIC.value:
            return not checks.note_disputes(flaw.note)
        return not checks.note_disputes(flaw.note) or self._cleared(flaw.member, flaw.quote, rnd)

    def _cleared(self, member: str, quote: str, rnd: int) -> bool:
        """Whether a quote's figures are in the member's prompt or its own arithmetic.

        The moderator called "unsourced" a figure from the research notes and a
        member's 60 x 160 (2026-10-27 Discord debate). Unknown after a resume: kept.
        """
        given = self._turn_sources.get((member, rnd))
        turns = [m.text for m in self.messages if m.speaker == member and m.round == rnd]
        if given is None or not turns:
            return False
        return checks.figures_cleared(quote, turns[-1], *given)

    def _drop_answered_flaws(self, notes: Sequence[ResearchNote]) -> bool:
        """Drop the round's "unsourced" flags whose figures the new notes give; True if any.

        The moderator flags figures in the same decision that asks for its lookups,
        so a lookup can source a figure it just called unsourced (2026-10-26 gacha run).
        """
        text = "\n".join(n.text for n in notes)
        if not text:
            return False
        rt = self.meta.runtime
        answered = [
            f
            for f in self._round_moderator_flaws()
            if f.kind is FlawKind.UNSOURCED and checks.figures_found(f.excerpt, text)
        ]
        if not answered:
            return False
        rt.flaws = [f for f in rt.flaws if f not in answered]
        for flaw in answered:
            self._notice(NOTICE_FLAW_ANSWERED.format(nick=self.nick(flaw.member)))
        return True

    def _answered(self, flaw: Flaw) -> bool:
        """Whether the member replied to the moderator's challenge of a flaw.

        The challenge is the moderator's line quoting the flaw in a later round (its
        floor turn); the reply is the member's next message after it.
        """
        challenged = False
        for message in self.messages:
            if message.round <= flaw.round:
                continue
            if message.speaker == self.moderator.slug and flaw.excerpt in message.text:
                challenged = True
            elif challenged and message.speaker == flaw.member:
                return True
        return False

    def _round_moderator_flaws(self) -> list[Flaw]:
        """Return the moderator's flaw reports for the current round, in its order."""
        rt = self.meta.runtime
        return [f for f in rt.flaws if f.by == "moderator" and f.round == rt.round]

    def _round_figure_flaws(self) -> list[Flaw]:
        """Return the figures the engine flagged in the current round, for the moderator."""
        rt = self.meta.runtime
        return [
            f
            for f in rt.flaws
            if f.by == "engine" and f.round == rt.round and f.kind in FIGURE_KINDS
        ]

    def _example_nick(self, persona: Persona) -> str:
        """Return another member's nick, to show how to address one (not a literal "@nick")."""
        others = [p.name for p in self.members.values() if p.slug != persona.slug]
        return others[0] if others else self.moderator.name
