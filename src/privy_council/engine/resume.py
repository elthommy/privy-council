"""Helpers to rebuild a `DebateEngine` from a session directory (`council resume`).

`session.json` holds the protocol position, saved at every turn boundary, and
`transcript.jsonl` holds every message. A turn that was in flight when the app
died is still in `runtime.pending`, so it simply runs again: at most that one
turn is lost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from privy_council.engine.notices import (
    NOTICE_RESUMED_AFTER_ROUND,
    NOTICE_RESUMED_CLOSED,
    NOTICE_RESUMED_EARLY_STOP,
    NOTICE_RESUMED_ERROR,
    NOTICE_RESUMED_MID_ROUND,
    NOTICE_RESUMED_OPENING,
    NOTICE_RESUMED_SYNTHESIS,
)
from privy_council.models.persona import Persona
from privy_council.models.session import (
    BudgetWarning,
    DossierAdded,
    ErrorEvent,
    Event,
    Message,
    MessageEvent,
    Phase,
    ResearchAdded,
    RoundStarted,
    SessionMeta,
    SessionState,
    StateUpdated,
    VoteTaken,
)

# States a session can't be resumed in as such: the call that ran is gone.
_INTERRUPTED = {
    SessionState.OPENING: SessionState.CREATED,
    SessionState.ROUND_RUNNING: SessionState.PAUSED,
    SessionState.SYNTHESIZING: SessionState.PAUSED,
}

# Events worth showing again when a front end replays the history.
_REPLAYED = (
    MessageEvent,
    RoundStarted,
    ResearchAdded,
    BudgetWarning,
    ErrorEvent,
    StateUpdated,
    VoteTaken,
    DossierAdded,
)


class ResumeError(Exception):
    """The session can't be resumed; the message says why and what to do."""


def resume_state(stored: SessionState) -> SessionState:
    """Map the state saved on disk to the state a resumed engine starts in."""
    return _INTERRUPTED.get(stored, stored)


def replay_events(events: Sequence[Event]) -> list[Event]:
    """Return the transcript events a front end shows again on resume."""
    return [e for e in events if isinstance(e, _REPLAYED)]


def messages_of(events: Sequence[Event]) -> list[Message]:
    """Return the chat messages, in order (the engine's `messages`)."""
    return [e.message for e in events if isinstance(e, MessageEvent)]


def resume_notice(meta: SessionMeta) -> str:
    """Return the system line telling the user where the resumed session stands."""
    rt = meta.runtime
    if meta.state is SessionState.ERROR_RECOVERABLE:
        return NOTICE_RESUMED_ERROR
    if meta.state is SessionState.CLOSED:
        return NOTICE_RESUMED_CLOSED
    if rt.phase is Phase.OPENING:
        return NOTICE_RESUMED_OPENING
    if rt.phase is Phase.SYNTHESIS:
        return NOTICE_RESUMED_EARLY_STOP if rt.early_stop else NOTICE_RESUMED_SYNTHESIS
    if rt.round_in_progress:
        return NOTICE_RESUMED_MID_ROUND.format(round=rt.round, count=len(rt.pending))
    return NOTICE_RESUMED_AFTER_ROUND.format(round=rt.round)


def check_resumable(
    meta: SessionMeta, personas: Mapping[str, Persona], local_only: bool = False
) -> None:
    """Raise ResumeError if the session can't be resumed here.

    That is: it's closed, its personas are gone, or it was created in the other
    mode (local-only is sticky, M15).
    """
    if meta.local_only != local_only:
        mode = "local-only" if meta.local_only else "normal"
        raise ResumeError(f"Session {meta.id} is a {mode} session; resume it in that mode")
    if meta.state is SessionState.CLOSED and not meta.runtime.memory_review.pending:
        raise ResumeError(f"Session {meta.id} is closed; its synthesis is in synthesis.md")
    missing = [s for s in [meta.moderator, *meta.members] if s not in personas]
    if missing:
        raise ResumeError(
            f"Session {meta.id} needs persona(s) {', '.join(missing)}, "
            "which are missing or invalid (council personas validate)"
        )
