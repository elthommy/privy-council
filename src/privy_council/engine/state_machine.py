"""Session state machine: the legal transitions between `SessionState`s."""

from __future__ import annotations

from privy_council.models.session import SessionState as S

# A running state can fail into ERROR_RECOVERABLE; retrying re-enters a running
# state. After a round the engine always pauses: `/step` semantics.
TRANSITIONS: dict[S, frozenset[S]] = {
    S.CREATED: frozenset({S.OPENING}),
    S.OPENING: frozenset({S.PAUSED, S.ERROR_RECOVERABLE}),
    S.PAUSED: frozenset({S.ROUND_RUNNING, S.SYNTHESIZING}),
    S.ROUND_RUNNING: frozenset({S.PAUSED, S.ERROR_RECOVERABLE}),
    S.SYNTHESIZING: frozenset({S.CLOSING, S.ERROR_RECOVERABLE}),
    S.CLOSING: frozenset({S.CLOSED}),
    S.CLOSED: frozenset(),
    S.ERROR_RECOVERABLE: frozenset({S.OPENING, S.ROUND_RUNNING, S.SYNTHESIZING}),
}


class InvalidTransitionError(Exception):
    """Raised on a transition the state machine doesn't allow (an engine bug)."""


class StateMachine:
    """Tracks the current session state and enforces legal transitions."""

    def __init__(self, state: S = S.CREATED) -> None:
        """Start in `state` (CREATED for a new session)."""
        self.state = state

    def can(self, target: S) -> bool:
        """Return whether moving to `target` is allowed."""
        return target in TRANSITIONS[self.state]

    def to(self, target: S) -> None:
        """Move to `target`, or raise InvalidTransitionError."""
        if not self.can(target):
            raise InvalidTransitionError(f"{self.state} -> {target} is not allowed")
        self.state = target
