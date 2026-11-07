"""Replay the end-of-session private-notes call on local models (no Claude calls).

Rebuilds the notes prompt a persona got at the close of a saved session (the
synthesis, the final debate state, its own and the user's lines, its earlier
notes) and sends it to each model, to check what the notes credit to the user.
The persona's notes from that session's day and council are left out, as they
were written by that very call.

    uv run python scripts/replay_notes.py SESSION_DIR --persona rigor --persona optimist \\
        --models nemotron-3.5-super qwen3.6:27b --runs 3 [--flag "accepted|will not"]

A `=value` suffix on a model sets Ollama's `think` option, as in replay_turn.py.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from replay_turn import _engine, _report, _run, model_think

from privy_council.backends.base import ChatMessage, CompletionResult
from privy_council.config import Settings, load_settings
from privy_council.engine import context
from privy_council.memory.distill import NOTES_MAX_OUTPUT, SessionDigest
from privy_council.memory.store import MemoryStore
from privy_council.models.persona import Persona
from privy_council.storage.sessions import SessionStore


class _Captured(Exception):
    """Raised by the fake call once the prompt is built."""

    def __init__(self, system: str, user: str) -> None:
        """Keep the prompt as the engine would send it."""
        super().__init__("captured")
        self.system, self.user = system, user


def _memory_copy(settings: Settings, council: str, day: str) -> MemoryStore:
    """Copy the memory dir, minus the bullets written on that session's day and council."""
    root = Path(tempfile.mkdtemp(prefix="council-notes-replay-"))
    shutil.copytree(settings.data_dir / "memory", root, dirs_exist_ok=True)
    stamp = f"- {day} ({council}):"
    for notes in root.glob("personas/*/notes.md"):
        kept = [line for line in notes.read_text().splitlines() if not line.startswith(stamp)]
        notes.write_text("\n".join(kept) + "\n" if kept else "")
    return MemoryStore(root)


def build_notes_prompt(settings: Settings, session: Path, slug: str) -> tuple[str, str, float]:
    """Return (system, user, temperature) of a persona's notes call at the session's close."""
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    engine = _engine(settings, meta, events)
    engine.memory = _memory_copy(settings, meta.council, meta.created_at.date().isoformat())
    persona: Persona = engine.members.get(slug) or engine.moderator
    if persona.slug != slug:
        raise SystemExit(f"{slug} is not in this session")
    digest = SessionDigest(
        council=engine.council.name,
        topic=meta.topic,
        day=meta.created_at.date(),
        synthesis=(session / "synthesis.md").read_text(),
        messages=list(engine.messages),
        state_text=context.state_text(meta.runtime.debate_state, engine.nick),
        nick_of=engine.nick,
        lang=meta.language,
    )
    writer = engine._memory_writer()

    async def capture(
        persona: Persona,
        prompt: context.Prompt,
        purpose: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_output: int | None = None,
    ) -> CompletionResult:
        """Localize the prompt like the engine, then stop."""
        del persona, json_schema, max_output
        chat = [ChatMessage(role="user", content=prompt.user)]
        system, localized = engine._localized(prompt.system, chat, purpose)
        raise _Captured(system, localized[-1].content)

    writer.complete = capture
    try:
        asyncio.run(writer.write_notes(persona, digest))
    except _Captured as got:
        return got.system, got.user, persona.meta.temperature
    raise SystemExit("the notes call was never made")


def main() -> None:
    """Replay each persona's notes call on each model and print the answers."""
    parser = argparse.ArgumentParser(description="Replay end-of-session notes on local models.")
    parser.add_argument("session", type=Path, help="Session directory (data/sessions/...)")
    parser.add_argument("--persona", action="append", required=True, help="Persona slug")
    parser.add_argument("--models", nargs="+", required=True, help="model[=think]")
    parser.add_argument("--runs", type=int, default=1, help="Answers per model and persona")
    parser.add_argument("--flag", default=None, help="Regex counted in each answer")
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--show-prompt", action="store_true", help="Print each prompt in full")
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    args = parser.parse_args()
    settings = load_settings(args.root)
    flag = re.compile(args.flag, re.IGNORECASE) if args.flag else None
    prompts = {slug: build_notes_prompt(settings, args.session, slug) for slug in args.persona}
    for slug, (system, user, _) in prompts.items():
        print(f"\n===== {slug}: prompt {len(system) + len(user)} chars")
        print(f"{system}\n\n{user}" if args.show_prompt else user[-800:])
    for raw in args.models:
        model, think = model_think(settings, raw)
        for slug, (system, user, temperature) in prompts.items():
            request = (system, user, temperature, NOTES_MAX_OUTPUT)
            for n in range(1, args.runs + 1):
                _report(f"{model} [{slug}]", n, _run(model, think, request, args.num_ctx), flag)


if __name__ == "__main__":
    main()
