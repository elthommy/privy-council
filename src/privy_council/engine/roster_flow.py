"""Changing who sits in the channel: `/summon` and `/kick` (brief §11)."""

from __future__ import annotations

from privy_council.engine import prompts
from privy_council.engine.core import EngineCore
from privy_council.engine.moderation import resolve_speaker
from privy_council.engine.notices import (
    NOTICE_BUSY,
    NOTICE_JOINED,
    NOTICE_JOINED_LATE,
    NOTICE_KICK_LAST,
    NOTICE_KICK_MODERATOR,
    NOTICE_KICK_USAGE,
    NOTICE_LEFT,
    NOTICE_SESSION_CLOSED,
    NOTICE_SUMMON_MEMBER,
    NOTICE_SUMMON_MODERATOR,
    NOTICE_SUMMON_PROPOSAL_IGNORED,
    NOTICE_SUMMON_UNKNOWN,
    NOTICE_SUMMON_USAGE,
    NOTICE_UNKNOWN_MEMBER,
)
from privy_council.models.moderator import SummonProposal
from privy_council.models.persona import Persona, Role
from privy_council.models.session import PendingTurn, Phase, RosterChanged

# Roles that run the debate rather than take part in it.
_CHAIR_ROLES = (Role.MODERATOR, Role.SCRIBE)


class RosterFlow(EngineCore):
    """Add and remove members between turns; the roster is saved in session.json."""

    def summon(self, who: str) -> None:
        """Add a persona to the channel; it speaks in the current or next round."""
        slug = self._summon_target(who)
        if slug is None:
            return
        persona = self.personas[slug]
        rt = self.meta.runtime
        self.members[slug] = persona
        self.meta.members.append(slug)
        self.meta.record_model(persona)
        if rt.phase in (Phase.OPENING, Phase.ROUND) or rt.early_stop:
            rt.pending.append(self._join_turn(slug))
        self._save()
        self._emit(RosterChanged(members=list(self.meta.members)))
        late = rt.phase is Phase.SYNTHESIS and not rt.early_stop
        self._notice((NOTICE_JOINED_LATE if late else NOTICE_JOINED).format(nick=persona.name))

    def kick(self, who: str) -> None:
        """Remove a member; its pending turns are dropped, its messages stay."""
        if not self._roster_editable(who, NOTICE_KICK_USAGE):
            return
        if resolve_speaker(who, {self.moderator.slug: self.moderator}):
            self._notice(NOTICE_KICK_MODERATOR.format(nick=self.moderator.name))
            return
        slug = resolve_speaker(who, self.members)
        if slug is None:
            self._notice(NOTICE_UNKNOWN_MEMBER.format(who=who))
            return
        if len(self.members) == 1:
            self._notice(NOTICE_KICK_LAST)
            return
        rt = self.meta.runtime
        del self.members[slug]
        self.meta.members.remove(slug)
        rt.pending = [t for t in rt.pending if t.speaker != slug]
        self._save()
        self._emit(RosterChanged(members=list(self.meta.members)))
        self._notice(NOTICE_LEFT.format(nick=self.nick(slug)))

    def _roster_editable(self, who: str, usage: str) -> bool:
        """Check the common preconditions of `/summon` and `/kick`, saying why not."""
        if not who.strip():
            self._notice(usage)
            return False
        if self.phase is Phase.DONE:
            self._notice(NOTICE_SESSION_CLOSED)
            return False
        if self.busy:
            self._notice(NOTICE_BUSY)
            return False
        return True

    def _summon_target(self, who: str) -> str | None:
        """Resolve a `/summon` argument to a persona slug that can join, or None (noticed)."""
        if not self._roster_editable(who, NOTICE_SUMMON_USAGE):
            return None
        slug = resolve_speaker(who, self.personas)
        if slug is None:
            available = sorted(p.slug for p in self._outsiders())
            self._notice(
                NOTICE_SUMMON_UNKNOWN.format(who=who, available=", ".join(available) or "none")
            )
            return None
        persona = self.personas[slug]
        if slug in self.members:
            self._notice(NOTICE_SUMMON_MEMBER.format(nick=persona.name))
            return None
        if persona.meta.role in _CHAIR_ROLES:
            self._notice(NOTICE_SUMMON_MODERATOR.format(nick=persona.name))
            return None
        return slug

    def _outsiders(self) -> list[Persona]:
        """Return the personas that could join: not in the channel, not a chair role."""
        return [
            p
            for p in self.personas.values()
            if p.slug not in self.members and p.meta.role not in _CHAIR_ROLES
        ]

    def _summon_candidates(self) -> list[Persona]:
        """Return the personas the moderator may propose: outsiders not proposed yet."""
        proposed = self.meta.runtime.proposed_summons
        return [p for p in self._outsiders() if p.slug not in proposed]

    def _summon_ask(self, proposal: SummonProposal | None) -> list[str]:
        """Turn the moderator's summon proposal into a line for the user, at most once each.

        The user decides: the line says how to `/summon`. Proposals made
        once the debate is over for good are dropped.
        """
        rt = self.meta.runtime
        if proposal is None or (rt.phase is Phase.SYNTHESIS and not rt.early_stop):
            return []
        slug = resolve_speaker(proposal.persona, self.personas)
        if slug in rt.proposed_summons or slug in self.members:
            return []
        if slug is None or slug not in {p.slug for p in self._outsiders()}:
            self._notice(NOTICE_SUMMON_PROPOSAL_IGNORED.format(who=proposal.persona))
            return []
        rt.proposed_summons.append(slug)
        self._save()
        persona = self.personas[slug]
        reason = proposal.reason.strip().rstrip(".")  # the line adds its own period
        return [prompts.SUMMON_ASK.format(nick=persona.name, reason=reason, slug=slug)]

    def _join_turn(self, slug: str) -> PendingTurn:
        """Return a newcomer's turn: blind while round 1 hasn't finished, else a catch-up."""
        rt = self.meta.runtime
        blind_round_ahead = rt.round == 0 or (rt.round == 1 and rt.round_in_progress)
        if blind_round_ahead and self.council.protocol.blind_first_round:
            return PendingTurn(speaker=slug)
        return PendingTurn(speaker=slug, prompt=prompts.JOIN_PROMPT[self.meta.language])
