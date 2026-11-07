"""Replay `/vote` at the end of a saved session with the current prompts.

Rebuilds the engine from the transcript, as it stood when the session ended,
and asks every member for its vote again, to compare confidences and positions
with the saved vote. A local-only session is reseated as it ran (no Claude
call); otherwise members on Claude spend quota, so `--allow-claude` is needed.
Memory is left empty, so later notes can't leak in.

    uv run python scripts/replay_vote.py SESSION_DIR [--runs 2] [--allow-claude]
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.engine.budget import Ledger
from privy_council.engine.protocol import DebateEngine
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import apply_local_only
from privy_council.memory.store import MemoryStore
from privy_council.models.persona import BackendKind
from privy_council.models.session import VoteTaken
from privy_council.storage.sessions import SessionStore


def build_engine(settings: Settings, session: Path, allow_claude: bool) -> DebateEngine:
    """Rebuild the session's engine at its end, with empty memory."""
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    council = councils[meta.council]
    if meta.local_only:
        settings, personas = apply_local_only(settings, personas, [council.moderator])
    claude = [s for s in council.members if personas[s].meta.backend is BackendKind.CLAUDE_CLI]
    if claude and not allow_claude:
        raise SystemExit(f"members on Claude ({', '.join(claude)}): pass --allow-claude")
    return DebateEngine(
        settings=settings,
        council=council,
        personas=personas,
        registry=BackendRegistry(
            settings, Ledger.in_data_dir(settings.data_dir), local_only=meta.local_only
        ),
        topic=meta.topic,
        memory=MemoryStore(Path(tempfile.mkdtemp(prefix="council-vote-replay-"))),
        restored=(meta, events),
    )


def saved_votes(session: Path) -> list[VoteTaken]:
    """Return the votes taken in the saved session."""
    return [e for e in SessionStore.open(session, {}).events() if isinstance(e, VoteTaken)]


async def replay(settings: Settings, session: Path, runs: int, allow_claude: bool) -> None:
    """Print the saved votes, then each replayed vote."""
    for taken in saved_votes(session):
        print(f"== saved vote, round {taken.round}")
        for v in taken.votes:
            print(f"  {v.speaker:12} {v.confidence}%  {v.position}")
    for n in range(1, runs + 1):
        engine = build_engine(settings, session, allow_claude)
        votes = await engine._collect_votes(list(engine.members.values()))
        await engine.registry.aclose()
        print(f"== replay {n}")
        for v in votes:
            print(f"  {v.speaker:12} {v.confidence}%  {v.position or v.error}")


def main() -> None:
    """Parse the arguments and replay the vote."""
    parser = argparse.ArgumentParser(description="Replay /vote at the end of a session.")
    parser.add_argument("session", type=Path, help="Session directory")
    parser.add_argument("--runs", type=int, default=1, help="Votes to replay")
    parser.add_argument("--allow-claude", action="store_true", help="Let Claude members vote")
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    args = parser.parse_args()
    asyncio.run(replay(load_settings(args.root), args.session, args.runs, args.allow_claude))


if __name__ == "__main__":
    main()
