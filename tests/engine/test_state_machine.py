from __future__ import annotations

import pytest

from privy_council.engine.state_machine import InvalidTransitionError, StateMachine
from privy_council.models.session import SessionState as S


def test_happy_path() -> None:
    sm = StateMachine()
    for target in [S.OPENING, S.PAUSED, S.ROUND_RUNNING, S.PAUSED, S.SYNTHESIZING, S.CLOSING]:
        sm.to(target)
    sm.to(S.CLOSED)
    assert sm.state is S.CLOSED


def test_error_is_recoverable_into_running_states() -> None:
    sm = StateMachine(S.ROUND_RUNNING)
    sm.to(S.ERROR_RECOVERABLE)
    assert sm.can(S.ROUND_RUNNING) and sm.can(S.SYNTHESIZING) and sm.can(S.OPENING)
    assert not sm.can(S.CLOSED)


@pytest.mark.parametrize(
    ("start", "target"),
    [(S.CREATED, S.ROUND_RUNNING), (S.PAUSED, S.CLOSED), (S.CLOSED, S.OPENING)],
)
def test_illegal_transitions_raise(start: S, target: S) -> None:
    with pytest.raises(InvalidTransitionError):
        StateMachine(start).to(target)
