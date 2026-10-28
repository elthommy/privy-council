"""Bench local models as the moderator, Chair (M15 step 0; no Claude calls).

Replays saved sessions with Chair reseated on each local model, as
`replay_moderation.py` does for one round: the engine is rebuilt from the
transcript as it stood, and only Chair's call is sent. Three kinds of item:

- round: a real round update, scored against what Claude's Chair decided then
  (consensus, flaws reported) and read by hand (next turns, floor prompts);
- plant: the same, after a known error was written into a member's turn; it is
  caught when a flaw names that member and quotes its marker, misattributed
  when a flaw quotes the marker under another member;
- synthesis: the final synthesis, saved next to Claude's to be read by hand.

The manifest names real sessions and quotes their turns, so it stays in
data/bench/ (gitignored). Every answer is appended to --out as soon as it
arrives: after a stop (Ctrl-C, or the GPU wanted back), the same command
skips what is done.

    uv run python -u scripts/chair_bench.py --manifest data/bench/chair_manifest.toml \\
        --models qwen3.6:27b gpt-oss:120b --out data/bench/runs/chair.jsonl [--num-ctx 24576]
    uv run python scripts/chair_bench.py --manifest ... --dry-run     # prompt sizes, no calls
    uv run python scripts/chair_bench.py --summary data/bench/runs/chair.jsonl
    uv run python scripts/chair_bench.py --dump data/bench/runs/chair.jsonl > read.md

Manifest:

    [[round]]
    session = "data/bench/sessions/..."
    round = 2

    [[plant]]
    session = "data/sessions/..."
    round = 3
    speaker = "value"
    find = "3-year cost/year: **$42**"     # exact text of the speaker's turn
    replace = "3-year cost/year: **$62**"
    marker = "62"                          # what a correct flaw quotes
    note = "125 / 3 is 42, not 62"

    [[synthesis]]
    session = "data/bench/sessions/..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from replay_moderation import _meta_at

from privy_council.backends.base import ChatMessage, CompletionResult
from privy_council.backends.registry import BackendRegistry
from privy_council.compare.spec import SeatChange, reseat
from privy_council.config import Settings, load_settings
from privy_council.engine import dynamics
from privy_council.engine.budget import Ledger
from privy_council.engine.moderation import decide
from privy_council.engine.protocol import MODERATE_MIN_OUTPUT_TOKENS, DebateEngine
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.memory.store import MemoryStore
from privy_council.models.council import MemoryMode
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.persona import BackendKind
from privy_council.models.session import (
    Event,
    MessageEvent,
    SessionMeta,
    SessionState,
    SessionStateChanged,
    StateUpdated,
)
from privy_council.storage.sessions import SYNTHESIS_MD, SessionStore
from privy_council.util.tokens import estimate_tokens

BENCH_TIMEOUT_S = 1800.0  # Super spends minutes on a 12k+ prompt before its first token
CLAUDE = "claude"  # the "model" of the reference rows (what Claude's Chair did then)
CLAUDE_PREFIX = "claude:"  # --models claude:sonnet runs Chair live on Claude (uses quota)


@dataclass(frozen=True)
class Plant:
    """A known error written into one member's turn."""

    speaker: str
    find: str
    replace: str
    marker: str
    note: str = ""


@dataclass(frozen=True)
class Item:
    """One Chair call to replay."""

    kind: str  # "round", "plant" or "synthesis"
    session: Path
    round: int = 0
    plants: tuple[Plant, ...] = field(default=())

    @property
    def id(self) -> str:
        """Return a stable id, unique within a manifest."""
        return f"{self.kind}:{self.session.name}:{self.round}"


def load_manifest(path: Path) -> list[Item]:
    """Read the manifest; plants on the same round become one item."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    items = [Item("round", Path(r["session"]), int(r["round"])) for r in data.get("round", [])]
    planted: dict[tuple[str, int], list[Plant]] = defaultdict(list)
    for p in data.get("plant", []):
        plant = Plant(p["speaker"], p["find"], p["replace"], p["marker"], p.get("note", ""))
        planted[(p["session"], int(p["round"]))].append(plant)
    items += [Item("plant", Path(s), r, tuple(ps)) for (s, r), ps in planted.items()]
    items += [Item("synthesis", Path(s["session"])) for s in data.get("synthesis", [])]
    return items


def plant_errors(events: list[Event], round_: int, plants: tuple[Plant, ...]) -> list[Event]:
    """Return the events with each plant's text replaced in its speaker's turn."""
    out = list(events)
    for plant in plants:
        hits = [
            i
            for i, e in enumerate(out)
            if isinstance(e, MessageEvent)
            and e.message.speaker == plant.speaker
            and e.message.round == round_
            and plant.find in e.message.text
        ]
        if len(hits) != 1:
            raise SystemExit(f"plant {plant.speaker} r{round_}: {plant.find!r} found {len(hits)}x")
        event = out[hits[0]]
        assert isinstance(event, MessageEvent)
        text = event.message.text.replace(plant.find, plant.replace)
        out[hits[0]] = event.model_copy(
            update={"message": event.message.model_copy(update={"text": text})}
        )
    return out


def bench_settings(settings: Settings, model: str, num_ctx: int | None) -> Settings:
    """Return settings with the bench timeout and, if given, the model's num_ctx."""
    ollama = settings.ollama
    models = dict(ollama.models)
    if num_ctx is not None and model in models:
        models[model] = models[model].model_copy(update={"num_ctx": num_ctx})
    elif num_ctx is not None:
        raise SystemExit(f"{model} has no [ollama.models] entry to set num_ctx on")
    update = {"models": models, "timeout_s": max(ollama.timeout_s, BENCH_TIMEOUT_S)}
    return settings.model_copy(update={"ollama": ollama.model_copy(update=update)})


def build_engine(
    settings: Settings, model: str | None, meta: SessionMeta, events: list[Event], rounds: int
) -> DebateEngine:
    """Rebuild a saved session's engine with Chair on `model` (None: as saved), memory off.

    `rounds` is the round limit: the saved session's own total, so a replayed
    round is the last one only if it was then (the meta at a round says less).
    """
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    council = councils[meta.council]
    protocol = council.protocol.model_copy(update={"max_rounds": rounds})
    council = council.model_copy(update={"protocol": protocol, "memory": MemoryMode.OFF})
    if model is not None:
        # "claude:<model>" reruns Claude's Chair, e.g. as a baseline on the plants.
        backend, name = BackendKind.OLLAMA, model
        if model.startswith(CLAUDE_PREFIX):
            backend, name = BackendKind.CLAUDE_CLI, model.removeprefix(CLAUDE_PREFIX)
        change = SeatChange(slug=meta.moderator, model=name, backend=backend)
        personas[meta.moderator] = reseat(personas[meta.moderator], change)
    return DebateEngine(
        settings=settings,
        council=council,
        personas=personas,
        registry=BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir)),
        topic=meta.topic,
        memory=MemoryStore(Path(tempfile.mkdtemp(prefix="council-chair-bench-"))),
        restored=(meta, events),
    )


def round_state(item: Item) -> tuple[SessionMeta, list[Event], SessionMeta]:
    """Return (meta at the round's end, events until its update, the saved final meta)."""
    store = SessionStore.open(item.session, {})
    final, events = store.load_meta(), store.events()
    if item.plants:
        events = plant_errors(events, item.round, item.plants)
    updates = [i for i, e in enumerate(events) if isinstance(e, StateUpdated)]
    meta = _meta_at(final, events, item.round, 0)
    # Flaws from this round on are what the replayed Chair must find for itself.
    meta.runtime.flaws = [
        f
        for f in final.runtime.flaws
        if f.round < item.round or (f.round == item.round and f.by == "engine")
    ]
    return meta, events[: updates[item.round - 1]], final


def synthesis_state(item: Item) -> tuple[SessionMeta, list[Event]]:
    """Return the saved meta and the events before the synthesis was written."""
    store = SessionStore.open(item.session, {})
    meta, events = store.load_meta(), store.events()
    cut = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, SessionStateChanged) and e.state is SessionState.SYNTHESIZING
    )
    return meta, events[:cut]


# --- Replaying -----------------------------------------------------------------------


async def replay_round(engine: DebateEngine, round_: int) -> dict[str, Any]:
    """Send Chair's round update; return the decision, attempts, time and trimming."""
    stats = dynamics.member_stats(engine.messages, list(engine.members), round_)
    request = engine._moderate_request(stats, round_ >= engine.council.protocol.max_rounds)

    results: list[CompletionResult] = []

    async def call(messages: list[ChatMessage], purpose: str) -> CompletionResult:
        """Send the moderator conversation in JSON mode, as the engine does."""
        result, _ = await engine._complete(
            engine.moderator,
            request,
            purpose,
            messages=messages,
            json_schema=engine._moderator_schema(),
            max_output=MODERATE_MIN_OUTPUT_TOKENS,
        )
        results.append(result)
        return result

    start = time.monotonic()
    decision, problems = await decide(call, request.user, engine.members)
    return {
        "seconds": round(time.monotonic() - start, 1),
        **usage_fields(results),
        "json": "failed" if decision is None else ("repaired" if problems else "first"),
        "problems": problems,
        "prompt_tokens": estimate_tokens(request.system) + estimate_tokens(request.user),
        "dropped_messages": request.dropped_messages,
        "state_clipped": request.state_clipped,
        **(decision_fields(decision) if decision else {}),
    }


def usage_fields(results: list[CompletionResult]) -> dict[str, Any]:
    """Return the tokens the calls read and wrote, and the model's own seconds."""
    usages = [r.usage for r in results if r.usage is not None]
    return {
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "truncated": any(r.truncated for r in results),
        "model_seconds": round(sum(r.latency_s for r in results), 1),
    }


def decision_fields(decision: ModeratorDecision) -> dict[str, Any]:
    """Return what the bench keeps of a decision."""
    return {
        "consensus": decision.consensus.value,
        "stop": decision.stop,
        "flaws": [
            {"member": f.member, "kind": f.kind, "quote": f.quote, "note": f.note}
            for f in decision.flaws
        ],
        "next_turns": [{"speaker": t.speaker, "prompt": t.prompt} for t in decision.next_turns],
        "new_arguments": decision.new_arguments,
        "questions_for_user": decision.questions_for_user,
    }


async def replay_synthesis(engine: DebateEngine) -> dict[str, Any]:
    """Write the synthesis on the replayed Chair; return its text and time."""
    prompt = engine._synthesis_prompt()
    start = time.monotonic()
    result, _ = await engine._synthesis_call(prompt)
    return {
        "seconds": round(time.monotonic() - start, 1),
        **usage_fields([result]),
        "text": result.text.strip(),
        "truncated": result.truncated,
        "prompt_tokens": estimate_tokens(prompt.system) + estimate_tokens(prompt.user),
        "dropped_messages": prompt.dropped_messages,
    }


async def run_item(settings: Settings, model: str, item: Item) -> dict[str, Any]:
    """Replay one item on one model; a backend error is recorded, not raised."""
    try:
        if item.kind == "synthesis":
            meta, events = synthesis_state(item)
            engine = build_engine(settings, model, meta, events, meta.runtime.round)
            row = await replay_synthesis(engine)
        else:
            meta, events, final = round_state(item)
            engine = build_engine(settings, model, meta, events, final.runtime.round)
            row = await replay_round(engine, item.round)
            row.update(score_plants(item, row.get("flaws", [])))
        await engine.registry.aclose()
    except Exception as exc:  # a bench keeps going: the error is the result
        row = {"error": f"{type(exc).__name__}: {exc}"}
    return row


# --- Scoring -------------------------------------------------------------------------


def _squash(text: str) -> str:
    """Lowercase and drop spaces, so "62 $" and "$62" compare on their digits."""
    return "".join(text.lower().split())


def score_plants(item: Item, flaws: list[dict[str, Any]]) -> dict[str, Any]:
    """Return, per plant, whether a flaw caught it and whether one blamed someone else."""
    if not item.plants:
        return {}
    results = []
    for plant in item.plants:
        marker = _squash(plant.marker)
        quoting = [f for f in flaws if marker in _squash(f["quote"] + " " + f["note"])]
        results.append(
            {
                "speaker": plant.speaker,
                "marker": plant.marker,
                "caught": any(f["member"] == plant.speaker for f in quoting),
                "misattributed": any(f["member"] != plant.speaker for f in quoting),
                "speaker_flagged": any(f["member"] == plant.speaker for f in flaws),
            }
        )
    return {"plants": results}


def claude_row(item: Item) -> dict[str, Any]:
    """Return what Claude's Chair did at this item in the saved session (the reference)."""
    if item.kind == "synthesis":
        path = item.session / SYNTHESIS_MD
        return {"text": path.read_text(encoding="utf-8").strip() if path.is_file() else ""}
    store = SessionStore.open(item.session, {})
    final, events = store.load_meta(), store.events()
    updates = [e for e in events if isinstance(e, StateUpdated)]
    update = updates[item.round - 1]
    flaws = [
        {"member": f.member, "kind": f.kind.value, "quote": f.excerpt, "note": f.note}
        for f in final.runtime.flaws
        if f.by == "moderator" and f.round == item.round
    ]
    # Older sessions predate flaw reports (M11): no reference there, not "none found".
    has_flaws = int(final.prompts_version or 0) >= 33
    return {
        "consensus": update.consensus.value if update.consensus else None,
        "flaws": flaws if has_flaws and not item.plants else None,
    }


# --- Running -------------------------------------------------------------------------


def done_keys(out: Path) -> set[tuple[str, int | None, str]]:
    """Return the (model, num_ctx, item id) rows already in the output file."""
    if not out.is_file():
        return set()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    return {(r["model"], r.get("num_ctx"), r["item"]) for r in rows}


def append(out: Path, row: dict[str, Any]) -> None:
    """Append one result line (written at once, so a stop keeps it)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace, items: list[Item]) -> None:
    """Replay every item on every model, model by model, skipping rows already done."""
    base = load_settings(args.root)
    done = done_keys(args.out)
    for item in items:  # Claude's reference rows, once
        if (CLAUDE, None, item.id) not in done:
            append(
                args.out, {"model": CLAUDE, "num_ctx": None, "item": item.id, **claude_row(item)}
            )
    for model in args.models:
        settings = bench_settings(base, model, args.num_ctx)
        todo = [i for i in items if (model, args.num_ctx, i.id) not in done]
        print(f"== {model} (num_ctx {args.num_ctx or 'config'}): {len(todo)} item(s) to run")
        for n, item in enumerate(todo, 1):
            row = await run_item(settings, model, item)
            append(args.out, {"model": model, "num_ctx": args.num_ctx, "item": item.id, **row})
            print(f"   [{n}/{len(todo)}] {item.id}: {brief(row)}  {datetime.now():%H:%M}")


def brief(row: dict[str, Any]) -> str:
    """Return a one-line view of a result row."""
    if "error" in row:
        return f"ERROR {row['error']}"
    parts = [
        f"{row['seconds']:.0f} s",
        f"~{row.get('prompt_tokens', 0)} tok in",
        f"{row.get('output_tokens', 0)} out",
    ]
    if row.get("truncated"):
        parts.append("CUT OFF")
    if row.get("dropped_messages"):
        parts.append(f"{row['dropped_messages']} msg dropped")
    if "json" in row:
        parts += [f"json {row['json']}", f"{len(row.get('flaws', []))} flaw(s)"]
    for p in row.get("plants", []):
        parts.append(f"plant {p['speaker']}: {'caught' if p['caught'] else 'missed'}")
    return ", ".join(parts)


def dry_run(args: argparse.Namespace, items: list[Item]) -> None:
    """Print each item's prompt size and trimming at the given num_ctx; no calls."""
    model = args.models[0]
    settings = bench_settings(load_settings(args.root), model, args.num_ctx)
    for item in items:
        if item.kind == "synthesis":
            meta, events = synthesis_state(item)
            synth_engine = build_engine(settings, model, meta, events, meta.runtime.round)
            prompt = synth_engine._synthesis_prompt()
        else:
            meta, events, final = round_state(item)
            engine = build_engine(settings, model, meta, events, final.runtime.round)
            stats = dynamics.member_stats(engine.messages, list(engine.members), item.round)
            last = item.round >= engine.council.protocol.max_rounds
            prompt = engine._moderate_request(stats, last)
        size = estimate_tokens(prompt.system) + estimate_tokens(prompt.user)
        print(f"{item.id}: ~{size} tok, {prompt.dropped_messages} msg dropped")


# --- Reading results ---------------------------------------------------------------


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a results file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summary(path: Path) -> None:
    """Print one line of scores per model and num_ctx."""
    rows = load_rows(path)
    ref = {r["item"]: r for r in rows if r["model"] == CLAUDE}
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["model"] != CLAUDE:
            groups[(r["model"], r.get("num_ctx"))].append(r)
    for (model, num_ctx), group in groups.items():
        print(f"{model} @ {num_ctx or 'config'}: {score_line(group, ref)}")


def score_line(group: list[dict[str, Any]], ref: dict[str, dict[str, Any]]) -> str:
    """Return a model's scores: JSON, plants, flags, consensus agreement, times."""
    rounds = [r for r in group if not r["item"].startswith("synthesis") and "error" not in r]
    errors = sum("error" in r for r in group)
    json_ok = [r["json"] for r in rounds]
    plants = [p for r in rounds for p in r.get("plants", [])]
    real = [r for r in rounds if r["item"].startswith("round")]
    flagged = [(len(r.get("flaws", [])), ref[r["item"]]["flaws"]) for r in real]
    with_ref = [(n, len(c)) for n, c in flagged if c is not None]
    agree = [
        r.get("consensus") == ref[r["item"]]["consensus"]
        for r in real
        if ref[r["item"]]["consensus"]
    ]
    synth = [r["seconds"] for r in group if r["item"].startswith("synthesis") and "error" not in r]
    secs = [r["seconds"] for r in rounds]
    return (
        f"JSON first {json_ok.count('first')}/{len(json_ok)}, repaired {json_ok.count('repaired')},"
        f" failed {json_ok.count('failed')}; errors {errors}; "
        f"plants caught {sum(p['caught'] for p in plants)}/{len(plants)},"
        f" misattributed {sum(p['misattributed'] for p in plants)}; "
        f"flaws/round {sum(n for n, _ in with_ref)} vs Claude {sum(c for _, c in with_ref)}"
        f" ({len(with_ref)} rounds); consensus = Claude {sum(agree)}/{len(agree)}; "
        f"round s median {statistics.median(secs) if secs else 0:.0f}, "
        f"synthesis s median {statistics.median(synth) if synth else 0:.0f}"
    )


def dump(path: Path) -> None:
    """Print every item's answers side by side as Markdown, Claude's first, to read by hand."""
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in load_rows(path):
        by_item[r["item"]].append(r)
    for item, group in by_item.items():
        print(f"\n# {item}\n")
        for r in sorted(group, key=lambda r: r["model"] != CLAUDE):
            label = "Claude (saved session)" if r["model"] == CLAUDE else r["model"]
            ctx = f" @ {r.get('num_ctx') or 'config'}" if r["model"] != CLAUDE else ""
            print(f"## {label}{ctx}\n")
            print(dump_row(r))


def dump_row(r: dict[str, Any]) -> str:
    """Render one answer for reading."""
    if "error" in r:
        return f"ERROR: {r['error']}\n"
    if "text" in r:
        return f"{r['text']}\n"
    lines = [f"consensus: {r.get('consensus')}", *(f"problem: {p}" for p in r.get("problems", []))]
    lines += [
        f'- flaw {f["member"]} [{f["kind"]}] "{f["quote"]}" {f["note"]}'
        for f in r.get("flaws") or []
    ]
    lines += [f"- next {t['speaker']}: {t['prompt']}" for t in r.get("next_turns", [])]
    lines += [f"- plant {p['speaker']} {p['marker']}: {p}" for p in r.get("plants", [])]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse the command line and run, dry-run, summarize or dump."""
    parser = argparse.ArgumentParser(description="Bench local models as Chair.")
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/chair_manifest.toml"))
    parser.add_argument("--models", nargs="+", default=["qwen3.6:27b"])
    parser.add_argument("--num-ctx", type=int, default=None, help="Override num_ctx")
    parser.add_argument("--out", type=Path, default=Path("data/bench/runs/chair.jsonl"))
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    parser.add_argument("--dry-run", action="store_true", help="Prompt sizes only, no calls")
    parser.add_argument("--summary", type=Path, help="Score a results file")
    parser.add_argument("--dump", type=Path, help="Print a results file as Markdown")
    args = parser.parse_args()
    if args.summary:
        summary(args.summary)
    elif args.dump:
        dump(args.dump)
    elif args.dry_run:
        dry_run(args, load_manifest(args.manifest))
    else:
        asyncio.run(run(args, load_manifest(args.manifest)))


if __name__ == "__main__":
    main()
