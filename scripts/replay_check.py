"""Replay the moderator's check of the members' notes at the close of a saved session.

Reads the notes each member wrote at that session's close (the bullets tagged
with its day and council, from `notes.md` or the file given by `--suffix`, e.g.
a `.bak-...` copy holding the notes as first written), sends them through the
real check on the moderator's backend, and prints every change. The moderator
on Claude spends one call per run, so `--allow-claude` is needed.

    uv run python scripts/replay_check.py SESSION_DIR [--suffix .bak-1107] [--runs 2] \\
        [--allow-claude]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from replay_turn import _engine

from privy_council.config import Settings, load_settings
from privy_council.memory.distill import SessionDigest
from privy_council.memory.store import bullet_lines
from privy_council.models.persona import BackendKind
from privy_council.storage.sessions import SessionStore


def saved_drafts(settings: Settings, council: str, day: str, suffix: str) -> dict[str, list[str]]:
    """Return each persona's notes from that day and council, as the check gets them.

    A persona's `notes.md<suffix>` is read when it exists, else its `notes.md`.
    """
    stamp = f"{day} ({council}): "
    drafts: dict[str, list[str]] = {}
    for folder in sorted((settings.data_dir / "memory" / "personas").iterdir()):
        path = folder / f"notes.md{suffix}"
        path = path if path.exists() else folder / "notes.md"
        if not path.exists():
            continue
        lines = bullet_lines(path.read_text())
        notes = [line.removeprefix(stamp) for line in lines if line.startswith(stamp)]
        if notes:
            drafts[folder.name] = notes
    return drafts


async def replay(settings: Settings, session: Path, suffix: str, runs: int, claude: bool) -> None:
    """Run the check `runs` times and print the changes."""
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    engine = _engine(settings, meta, events)
    if engine.moderator.meta.backend is BackendKind.CLAUDE_CLI and not claude:
        raise SystemExit("the moderator is on Claude: pass --allow-claude")
    digest = SessionDigest(
        council=engine.council.name,
        topic=meta.topic,
        day=meta.created_at.date(),
        synthesis="",
        messages=list(engine.messages),
        state_text=None,
        nick_of=engine.nick,
        lang=meta.language,
    )
    drafts = saved_drafts(settings, meta.council, meta.created_at.date().isoformat(), suffix)
    for slug, notes in drafts.items():
        for note in notes:
            print(f"  {slug:11} {note}")
    writer = engine._memory_writer()
    for n in range(1, runs + 1):
        checked, fixed, dropped = await writer.check_notes(engine.moderator, digest, drafts)
        print(f"== run {n}: {fixed} fixed, {dropped} dropped")
        for slug, notes in drafts.items():
            for note in notes:
                if note not in checked.get(slug, []):
                    print(f"  {slug:11} WAS {note}")
            for note in checked.get(slug, []):
                if note not in notes:
                    print(f"  {slug:11} NOW {note}")
    await engine.registry.aclose()


def main() -> None:
    """Parse the arguments and replay the check."""
    parser = argparse.ArgumentParser(description="Replay the check of the members' notes.")
    parser.add_argument("session", type=Path, help="Session directory")
    parser.add_argument("--suffix", default="", help="Read notes.md<suffix> instead")
    parser.add_argument("--runs", type=int, default=1, help="Checks to run")
    parser.add_argument("--allow-claude", action="store_true", help="Let a Claude moderator run")
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    args = parser.parse_args()
    settings = load_settings(args.root)
    asyncio.run(replay(settings, args.session, args.suffix, args.runs, args.allow_claude))


if __name__ == "__main__":
    main()
