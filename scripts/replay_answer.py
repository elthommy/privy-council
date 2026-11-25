"""Replay the lookup an answer triggers between rounds, on a saved session.

The engine is rebuilt as it stood after the round's update, with the Chair's question
and an answer; then the engine asks the moderator what to look up. One short Claude
call per run (local with --local-only), plus one web search with --search.

    uv run python scripts/replay_answer.py SESSION_DIR --round 1 --root SCRATCH \\
        [--answer "I leave it to you"] [--runs 3] [--local-only] [--search]

Without --answer, the session's own answer is used. Use a --root whose data dir is a
scratch dir: calls go into its usage.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from replay_moderation import _engine

from privy_council.config import Settings, load_settings
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.resume import messages_of
from privy_council.models.session import (
    Phase,
    ResearchAdded,
    RoundStarted,
    SessionState,
    StateUpdated,
)
from privy_council.storage.sessions import SessionStore


def build(settings: Settings, session: Path, round_: int, local_only: bool) -> DebateEngine:
    """Rebuild the engine as it stood just before round `round_ + 1` started."""
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    starts = [
        i for i, e in enumerate(events) if isinstance(e, RoundStarted) and e.round == round_ + 1
    ]
    before = events[: starts[0]]
    update = max(i for i, e in enumerate(before) if isinstance(e, StateUpdated))
    state = before[update]
    assert isinstance(state, StateUpdated)
    out = meta.model_copy(deep=True)
    out.state = SessionState.PAUSED
    out.local_only = out.local_only or local_only
    rt = out.runtime
    rt.phase, rt.round, rt.round_in_progress, rt.pending = Phase.ROUND, round_, False, []
    rt.debate_state = state.state
    rt.moderated_upto = len(messages_of(before[: update + 1]))
    rt.research = [e.note for e in before if isinstance(e, ResearchAdded)]
    return _engine(settings, out, before)


def replace_answer(engine: DebateEngine, answer: str) -> None:
    """Put `answer` in place of the user's answers since the update."""
    start = engine.meta.runtime.moderated_upto
    own = [m for m in engine.messages[start:] if m.speaker == "user"]
    for message in own[1:]:
        engine.messages.remove(message)
    if own:
        index = engine.messages.index(own[0])
        engine.messages[index] = own[0].model_copy(update={"text": answer})


async def main() -> None:
    """Replay the lookup choice, and the search if asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--round", type=int, default=1, help="The round whose update asked")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--answer", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.root)
    for _ in range(args.runs):
        engine = build(settings, args.session, args.round, args.local_only)
        if args.answer:
            replace_answer(engine, args.answer)
        exchange = engine._answer_exchange()
        print("\n".join(exchange) or "(no answer to a question)")
        focus = await engine._answer_focus(exchange)
        print(f"=> focus: {focus!r}\n")
        if focus and args.search:
            await engine._research_safely(focus, history=exchange)
            print(engine.meta.runtime.research[-1].text)


if __name__ == "__main__":
    asyncio.run(main())
