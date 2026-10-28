"""Session-level models: messages, debate state and engine events."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from privy_council.models.flaw import Flaw
from privy_council.models.persona import Persona
from privy_council.models.vote import Vote
from privy_council.util.language import Language


def _now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class SessionState(StrEnum):
    """Engine state machine states (brief §13)."""

    CREATED = "CREATED"
    OPENING = "OPENING"
    ROUND_RUNNING = "ROUND_RUNNING"
    PAUSED = "PAUSED"
    SYNTHESIZING = "SYNTHESIZING"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR_RECOVERABLE = "ERROR_RECOVERABLE"


class Message(BaseModel):
    """One chat line: a persona turn, a user interjection or a system notice."""

    model_config = ConfigDict(frozen=True)

    speaker: str  # persona slug, "user" or "system"
    text: str
    round: int = Field(default=0, ge=0)
    ts: datetime = Field(default_factory=_now)
    fallback: bool = False  # produced by the local fallback model instead of Claude
    truncated: bool = False  # cut by the hard output cap; the text is partial
    to_user: bool = False  # the moderator asks the user something (highlighted)
    # Who wrote a "user" message when several people share the chat; None is the owner.
    author: str | None = None

    def display_nick(self, nick_of: Callable[[str], str]) -> str:
        """Return the nick to show for this message: its author, else the speaker's."""
        return self.author or nick_of(self.speaker)


class Consensus(StrEnum):
    """How far the members agree, as judged by the moderator after each round."""

    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class Disagreement(BaseModel):
    """A point of contention and who stands where on it."""

    topic: str
    sides: dict[str, str] = Field(default_factory=dict)  # slug -> stance


class DebateState(BaseModel):
    """Compact moderator-maintained summary; what small models see instead of the transcript."""

    # What the user said about their own situation; kept all session, so a detail
    # survives after the disagreement that carried it is gone.
    user_facts: list[str] = Field(default_factory=list)
    positions: dict[str, str] = Field(default_factory=dict)  # slug -> current position
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class Phase(StrEnum):
    """What the next `/step` does (the protocol position, unlike SessionState)."""

    OPENING = "opening"
    ROUND = "round"
    SYNTHESIS = "synthesis"
    DONE = "done"


class PendingTurn(BaseModel):
    """A turn still to run in the current or next round."""

    model_config = ConfigDict(frozen=True)

    speaker: str  # persona slug
    prompt: str = ""  # the moderator's targeted instruction; empty in the blind round


class ResearchNote(BaseModel):
    """Facts the moderator found on the web, shared with every member."""

    model_config = ConfigDict(frozen=True)

    focus: str  # what was asked for: the user's /research query, or the default focus
    text: str
    searches: list[str] = Field(default_factory=list)  # the web queries actually run
    ts: datetime = Field(default_factory=_now)
    truncated: bool = False
    auto: bool = False  # started by the moderator on its own (counts toward auto_max)
    # Who searched: the claude CLI's web tools, or SearXNG (local-only sessions).
    source: Literal["claude", "local"] = "claude"


class DossierFile(BaseModel):
    """A file the user attached; `text` is what prompts get (the file or its summary)."""

    name: str  # the copy's name in the session's dossier/ directory
    text: str
    tokens: int = Field(ge=0)  # estimated size of the original file
    summarized: bool = False  # text is the moderator's summary (plus a CSV's first rows)
    stats: str | None = None  # a CSV's exact totals, computed by the engine
    ts: datetime = Field(default_factory=_now)


class MemoryReview(BaseModel):
    """Shared-memory changes waiting for the user's `/keep` (brief §10: approval)."""

    facts: list[str] = Field(default_factory=list)  # proposed bullets, numbered from 1
    rewrite: str | None = None  # proposed merged user.md, when it outgrew its cap

    @property
    def pending(self) -> bool:
        """Return whether anything awaits a decision."""
        return bool(self.facts) or self.rewrite is not None


class EngineRuntime(BaseModel):
    """Protocol progress, saved at turn boundaries so a failed step can be retried."""

    phase: Phase = Phase.OPENING
    round: int = Field(default=0, ge=0)
    round_in_progress: bool = False
    # Turns of the current round not yet answered (or of the next round, once planned).
    pending: list[PendingTurn] = Field(default_factory=list)
    debate_state: DebateState = Field(default_factory=DebateState)
    research: list[ResearchNote] = Field(default_factory=list)
    dossier: list[DossierFile] = Field(default_factory=list)
    # Number of transcript messages the moderator has already folded into the state.
    moderated_upto: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    memory_review: MemoryReview = Field(default_factory=MemoryReview)
    # The latest `/vote` and the round it was taken in, for the final synthesis.
    last_vote: list[Vote] = Field(default_factory=list)
    last_vote_round: int = Field(default=0, ge=0)
    # Dynamic moderation (M7): the moderator's latest consensus call, how many rounds
    # in a row brought nothing new, and the personas it already proposed to summon.
    consensus: Consensus = Consensus.NONE
    stalled_rounds: int = Field(default=0, ge=0)
    # The moderator's last round judgment: what was new, and whether the decision is open.
    new_arguments: list[str] | None = None
    decision_open: bool = True
    proposed_summons: list[str] = Field(default_factory=list)
    # Questions the moderator put to the user, listed in its task so it never re-asks.
    asked_questions: list[str] = Field(default_factory=list)
    # The debate stopped before max_rounds (consensus or looping): `pending` keeps the
    # moderator's plan, and /step or /run resumes debating instead of the synthesis.
    early_stop: bool = False
    # Rounds added past the council's max_rounds with `/more N`.
    extra_rounds: int = Field(default=0, ge=0)
    # Flaws found in member turns (M11), by the engine's checks and by the moderator.
    flaws: list[Flaw] = Field(default_factory=list)


class SessionMeta(BaseModel):
    """Contents of `session.json`: what the session is and where it stands."""

    id: str
    council: str
    topic: str
    moderator: str
    members: list[str]
    created_at: datetime = Field(default_factory=_now)
    state: SessionState = SessionState.CREATED
    runtime: EngineRuntime = Field(default_factory=EngineRuntime)
    prompts_version: str = ""
    # `ResearchMode` value chosen at `council new`, so a resume keeps it.
    research: Literal["off", "on_demand", "auto", "required"] = "on_demand"
    # Debate language (M10): None until the question is known, then detected from it
    # or set by `--lang` / `/lang`. Sessions from before M10 have no field: English.
    lang: Language | None = None
    # "backend:model" each participant ran on, in order, so a later persona edit doesn't
    # rewrite history (M13). Sessions from before M13 have none.
    models: dict[str, list[str]] = Field(default_factory=dict)
    # Local-only mode (M15): no call may reach Claude. Set at creation; resume keeps it.
    local_only: bool = False

    def record_model(self, persona: Persona) -> None:
        """Note the model a participant runs on now, if it differs from the last one."""
        seat = f"{persona.meta.backend.value}:{persona.meta.model}"
        seen = self.models.setdefault(persona.slug, [])
        if not seen or seen[-1] != seat:
            seen.append(seat)

    @property
    def language(self) -> Language:
        """Return the debate language, English until it is decided."""
        return self.lang or "en"


class _EventBase(BaseModel):
    """Common fields of every engine event."""

    model_config = ConfigDict(frozen=True)

    ts: datetime = Field(default_factory=_now)


class MessageEvent(_EventBase):
    """A new chat message was produced."""

    kind: Literal["message"] = "message"
    message: Message


class RoundStarted(_EventBase):
    """A protocol round began."""

    kind: Literal["round_started"] = "round_started"
    round: int = Field(ge=0)
    label: str = ""


class StateUpdated(_EventBase):
    """The moderator produced a new debate state."""

    kind: Literal["state_updated"] = "state_updated"
    state: DebateState
    consensus: Consensus | None = None  # None in transcripts written before M7
    stalled_rounds: int = Field(default=0, ge=0)
    new_arguments: list[str] | None = None
    decision_open: bool | None = None  # None in older transcripts


class BudgetWarning(_EventBase):
    """Claude usage is approaching or past a soft cap."""

    kind: Literal["budget_warning"] = "budget_warning"
    text: str


class ErrorEvent(_EventBase):
    """A recoverable error, with an actionable message for the UI."""

    kind: Literal["error"] = "error"
    text: str
    recoverable: bool = True


class SessionStateChanged(_EventBase):
    """The engine state machine moved to a new state."""

    kind: Literal["session_state"] = "session_state"
    state: SessionState


class TurnStarted(_EventBase):
    """A persona started thinking (the UI can show a status)."""

    kind: Literal["turn_started"] = "turn_started"
    speaker: str
    round: int = Field(ge=0)


class ResearchAdded(_EventBase):
    """The moderator added web research to the shared context."""

    kind: Literal["research"] = "research"
    note: ResearchNote


class MemoryProposed(_EventBase):
    """Shared-memory changes await the user's approval (`/keep`)."""

    kind: Literal["memory_proposal"] = "memory_proposal"
    facts: list[str] = Field(default_factory=list)
    rewrite: str | None = None


class TextDelta(_EventBase):
    """A chunk of a reply being typed (the TUI typing effect); never saved to disk."""

    kind: Literal["delta"] = "delta"
    speaker: str
    text: str
    restart: bool = False  # first chunk of a (re)try: drop what was shown before


class VoteTaken(_EventBase):
    """The members voted (`/vote`); shown as a table, fed to the final synthesis."""

    kind: Literal["vote"] = "vote"
    round: int = Field(ge=0)  # the round the debate was in
    votes: list[Vote]


class DossierAdded(_EventBase):
    """The user attached a file to the dossier (`/attach`)."""

    kind: Literal["dossier"] = "dossier"
    file: DossierFile


class FlawsFound(_EventBase):
    """Flaws found in member turns (M11); hidden in the chat, `/flaws` lists them."""

    kind: Literal["flaws"] = "flaws"
    round: int = Field(ge=0)
    flaws: list[Flaw]


class RosterChanged(_EventBase):
    """A persona joined or left the channel (`/summon`, `/kick`)."""

    kind: Literal["roster"] = "roster"
    members: list[str]  # member slugs after the change


Event = Annotated[
    MessageEvent
    | RoundStarted
    | StateUpdated
    | BudgetWarning
    | ErrorEvent
    | SessionStateChanged
    | TurnStarted
    | ResearchAdded
    | MemoryProposed
    | RosterChanged
    | VoteTaken
    | DossierAdded
    | FlawsFound
    | TextDelta,
    Field(discriminator="kind"),
]
