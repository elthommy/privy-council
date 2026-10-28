"""Replay member turns of a saved session on other local models (no Claude calls).

Compares models on the exact prompt a member got: for each chosen turn, the
engine is rebuilt from the transcript as it stood just before that member spoke
(debate state, recent messages, the moderator's floor line, language line), and
the prompt it would send goes to each model through Ollama. Memory is left
empty, so later sessions' notes can't leak in.

    uv run python scripts/replay_turn.py SESSION_DIR --turn coach:2 --turn vitality:3 \\
        --models qwen3.6:latest qwen3.6:27b=false mistral-small3.2:24b --runs 2 \\
        [--flag "quotidien|chaque jour"] [--root DIR] [--num-ctx 12288]

A `=value` suffix sets Ollama's `think` option (true/false, or low/medium/high).
`--flag` is a regex counted in each answer, e.g. a detail the model got wrong.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from privy_council.backends.base import ChatMessage
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.engine.budget import Ledger
from privy_council.engine.protocol import DebateEngine
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.memory.store import MemoryStore
from privy_council.models.session import (
    DebateState,
    Event,
    MessageEvent,
    PendingTurn,
    Phase,
    SessionMeta,
    SessionState,
    StateUpdated,
)
from privy_council.storage.sessions import SessionStore

OLLAMA = "http://localhost:11434/api/chat"


def _think(raw: str | None) -> bool | str | None:
    """Parse the `=value` suffix into Ollama's think option."""
    if raw is None:
        return None
    return {"true": True, "false": False}.get(raw.lower(), raw.lower())


def _cut(events: list[Event], speaker: str, round_: int) -> int:
    """Return the index of the member's first message in that round."""
    for i, e in enumerate(events):
        if isinstance(e, MessageEvent) and (e.message.speaker, e.message.round) == (
            speaker,
            round_,
        ):
            return i
    raise SystemExit(f"no message from {speaker} in round {round_}")


def _floor_prompt(events: list[Event], moderator: str, name: str) -> str:
    """Return the moderator's floor prompt for the member, if it was the last line."""
    for e in reversed(events):
        if not isinstance(e, MessageEvent) or e.message.speaker == "system":
            continue
        text = e.message.text
        if e.message.speaker == moderator and re.match(rf"@{re.escape(name)}[,:]\s", text):
            return text.split(maxsplit=1)[1]
        return ""
    return ""


def _meta_at(meta: SessionMeta, events: list[Event], round_: int) -> SessionMeta:
    """Return the session meta as it stood mid-round, before the member's turn."""
    out = meta.model_copy(deep=True)
    out.state = SessionState.PAUSED
    rt = out.runtime
    rt.phase, rt.round, rt.round_in_progress, rt.pending = Phase.ROUND, round_, True, []
    states = [e.state for e in events if isinstance(e, StateUpdated)]
    rt.debate_state = states[-1] if states else DebateState()
    return out


def _engine(settings: Settings, meta: SessionMeta, events: list[Event]) -> DebateEngine:
    """Build an engine on the cut transcript, with no store and empty memory."""
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    registry = BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir))
    memory = MemoryStore(Path(tempfile.mkdtemp(prefix="council-replay-")))
    return DebateEngine(
        settings=settings,
        council=councils[meta.council],
        personas=personas,
        registry=registry,
        topic=meta.topic,
        memory=memory,
        restored=(meta, events),
    )


@dataclass(frozen=True)
class ReplayTurn:
    """A member turn rebuilt from a saved session, ready to send to other models."""

    spec: str  # speaker:round
    system: str
    user: str
    temperature: float
    max_output: int
    nick: str
    known: list[str]  # nicks and slugs the member may address
    devil_line: str | None  # the line a devil's advocate owes in round 1
    original: str  # what the member actually said


def build_turn(settings: Settings, session: Path, spec: str) -> ReplayTurn:
    """Rebuild the prompt of a `speaker:round` turn as the engine sent it."""
    speaker, _, round_text = spec.partition(":")
    store = SessionStore.open(session, {})
    meta, events = store.load_meta(), store.events()
    round_ = int(round_text)
    cut = _cut(events, speaker, round_)
    before = events[:cut]
    engine = _engine(settings, _meta_at(meta, before, round_), before)
    persona = engine.members[speaker]
    turn = PendingTurn(
        speaker=speaker, prompt=_floor_prompt(before, engine.moderator.slug, persona.name)
    )
    prompt = engine._member_prompt(persona, turn)
    chat = [ChatMessage(role="user", content=prompt.user)]
    system, localized = engine._localized(prompt.system, chat, "turn")
    original = events[cut]
    assert isinstance(original, MessageEvent)
    return ReplayTurn(
        spec=spec,
        system=system,
        user=localized[-1].content,
        temperature=persona.meta.temperature,
        max_output=engine._max_output(persona),
        nick=persona.name,
        known=engine._addressable(),
        devil_line=engine._devil_line(persona),
        original=original.message.text,
    )


def _request(settings: Settings, session: Path, spec: str) -> tuple[str, str, float, int]:
    """Build (system, user, temperature, max_output) for a `speaker:round` turn."""
    turn = build_turn(settings, session, spec)
    return turn.system, turn.user, turn.temperature, turn.max_output


def _run(
    model: str, think: bool | str | None, request: tuple[str, str, float, int], num_ctx: int
) -> dict[str, Any]:
    """Send one non-streaming chat call; return Ollama's response plus wall time."""
    system, user, temperature, max_output = request
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "10m",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_output},
    }
    if think is not None:
        body["think"] = think
    start = time.monotonic()
    response: dict[str, Any] = httpx.post(OLLAMA, json=body, timeout=1800).json()
    response["wall_s"] = time.monotonic() - start
    return response


def _report(model: str, n: int, r: dict[str, Any], flag: re.Pattern[str] | None) -> None:
    """Print one answer with its timing and flag count."""
    if "error" in r:
        print(f"\n--- {model} #{n}: ERROR {r['error']}")
        return
    text = r["message"]["content"].strip()
    rate = r["eval_count"] / (r["eval_duration"] / 1e9)
    flagged = f" | flag hits: {len(flag.findall(text))}" if flag else ""
    print(
        f"\n--- {model} #{n}: {r['wall_s']:.0f}s (load {r['load_duration'] / 1e9:.0f}s, "
        f"{r['eval_count']} tokens at {rate:.1f} tok/s){flagged}\n{text}"
    )


def main() -> None:
    """Replay each turn on each model and print the answers side by side."""
    parser = argparse.ArgumentParser(description="Replay member turns on local models.")
    parser.add_argument("session", type=Path, help="Session directory (data/sessions/...)")
    parser.add_argument("--turn", action="append", required=True, help="speaker:round")
    parser.add_argument("--models", nargs="+", required=True, help="model[=think]")
    parser.add_argument("--runs", type=int, default=1, help="Answers per model and turn")
    parser.add_argument("--flag", default=None, help="Regex counted in each answer")
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    args = parser.parse_args()
    settings = load_settings(args.root)
    flag = re.compile(args.flag, re.IGNORECASE) if args.flag else None
    requests = {spec: _request(settings, args.session, spec) for spec in args.turn}
    for spec in args.turn:
        print(f"\n===== turn {spec}: prompt {sum(len(p) for p in requests[spec][:2])} chars")
        print(requests[spec][1][-600:])
    # Model-major order: each model loads once for all its turns.
    for raw in args.models:
        model, _, think = raw.partition("=")
        for spec in args.turn:
            for n in range(1, args.runs + 1):
                result = _run(model, _think(think or None), requests[spec], args.num_ctx)
                _report(f"{model} [{spec}]", n, result, flag)


if __name__ == "__main__":
    main()
