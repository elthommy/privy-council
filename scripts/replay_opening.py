"""Replay a saved session's opening on local models (no Claude calls, no search).

The engine is rebuilt from the session as it stood before the Chair's opening (its
first research notes kept, memory empty), and the opening prompt it would send goes
to each model through Ollama. Prints each opening and a few counts: lettered
readings in the questions to the user, @-mentions of members, and a `--flag` regex.

    uv run python scripts/replay_opening.py SESSION_DIR --models qwen3.6:latest \\
        --runs 3 [--root DIR] [--flag "disjunct|\\bOR\\b"] [--local-only]

`--local-only` seats the Chair as a local-only session does, so the prompt is
trimmed to the local model's context like in that session.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from replay_turn import _run, model_think

from privy_council.backends.base import ChatMessage
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.engine.budget import Ledger
from privy_council.engine.protocol import DebateEngine
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import apply_local_only
from privy_council.memory.store import MemoryStore
from privy_council.models.session import MessageEvent, Phase, SessionState
from privy_council.storage.sessions import SessionStore

READING = re.compile(r"\(([a-d])\)")


def build_opening(settings: Settings, session: Path, local_only: bool) -> tuple[str, str, float]:
    """Rebuild (system, user, temperature) of the session's opening call."""
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    council = councils[meta.council]
    cut = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, MessageEvent) and e.message.speaker == council.moderator
    )
    opened = events[cut]
    assert isinstance(opened, MessageEvent)
    out = meta.model_copy(deep=True)
    out.state = SessionState.PAUSED
    rt = out.runtime
    rt.phase, rt.round, rt.round_in_progress, rt.pending = Phase.OPENING, 0, False, []
    rt.research = [n for n in rt.research if n.ts <= opened.message.ts]
    if local_only:
        settings, personas = apply_local_only(settings, personas, {council.moderator})
    engine = DebateEngine(
        settings=settings,
        council=council,
        personas=personas,
        registry=BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir)),
        topic=meta.topic,
        memory=MemoryStore(Path(tempfile.mkdtemp(prefix="council-replay-"))),
        restored=(out, events[:cut]),
    )
    prompt = engine._opening_prompt()
    chat = [ChatMessage(role="user", content=prompt.user)]
    system, localized = engine._localized(prompt.system, chat, "opening")
    return system, localized[-1].content, engine.moderator.meta.temperature


def _counts(text: str, nicks: list[str], flag: re.Pattern[str] | None) -> str:
    """Return the counts printed after each opening."""
    readings = len(READING.findall(text))
    mentions = sum(text.count(f"@{n}") for n in nicks)
    flagged = len(flag.findall(text)) if flag else 0
    return f"readings (a)-(d): {readings} · @members: {mentions} · flag: {flagged}"


def main() -> None:
    """Replay the opening on each model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Session directory (data/sessions/...)")
    parser.add_argument("--models", nargs="+", default=["qwen3.6:latest"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--flag", default=None, help="Regex counted in each opening")
    parser.add_argument("--num-ctx", type=int, default=24576)
    parser.add_argument("--max-output", type=int, default=3000)
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.root or Path.cwd())
    system, user, temperature = build_opening(settings, args.session, args.local_only)
    personas, _ = load_personas(settings.personas_dir)
    nicks = [p.name for p in personas.values()]
    flag = re.compile(args.flag) if args.flag else None
    print(f"prompt: {len(system) + len(user)} chars")
    for raw in args.models:
        model, think = model_think(settings, raw)
        for n in range(1, args.runs + 1):
            request = (system, user, temperature, args.max_output)
            response = _run(model, think, request, args.num_ctx)
            text = response.get("message", {}).get("content", "")
            print(f"\n===== {raw} run {n} ({response['wall_s']:.0f} s)\n{text}")
            print(f"----- {_counts(text, nicks, flag)}")


if __name__ == "__main__":
    main()
