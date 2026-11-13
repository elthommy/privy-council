"""Count the flaws the engine's checks find in member turns, per model (M11; no Claude).

Rebuilds each turn of a manifest the way `replay_turn.py` does (the exact prompt
the member got), sends it to each local model, runs `engine/checks.py` on the
answers and prints the counts per model and kind. `--original` also checks what
the members actually said in the saved session. Contradictions and logic errors
aren't counted: read the answers, saved as JSON lines under data/bench/runs/.

    uv run python -u scripts/flaw_bench.py [--manifest data/bench/manifest.toml] \\
        [--original] [--models qwen3.6:27b=false gemma4:26b=false ...] [--runs 1]
    uv run python scripts/flaw_bench.py --recheck data/bench/runs/RUN.jsonl
    uv run python -u scripts/flaw_bench.py --prompts-from RUN.jsonl --models ...  # same prompts

A run appends to `--out` and skips the answers already there: after a Ctrl-C, the
same command resumes. `--fresh` overwrites instead.

The manifest holds real sessions, so it stays in data/bench/ (gitignored):

    [[turn]]
    session = "data/sessions/2026-10-19_purchases_..."
    turn = "value:2"  # speaker:round
    note = "32 GB max; ~5 GB more"
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from replay_turn import ReplayTurn, _run, build_turn, model_think

from privy_council.config import Settings, load_settings
from privy_council.engine import checks, context, dynamics
from privy_council.models.flaw import Flaw, FlawKind

ORIGINAL = "original"
# Counted beside the flaws: the update marker (true or false is read by hand).
MARKER = "marker"
# The kinds the engine finds (the moderator's need a debate), then the markers.
COLUMNS = [
    *(
        k.value
        for k in (
            FlawKind.UNSOURCED,
            FlawKind.MISLABELLED,
            FlawKind.WRONG_UNIT,
            FlawKind.SELF_PREFIX,
            FlawKind.SELF_MENTION,
            FlawKind.UNKNOWN_MENTION,
            FlawKind.NICK_LITERAL,
            FlawKind.MISSING_DEVIL_LINE,
        )
    ),
    MARKER,
]


def _check(turn: ReplayTurn, text: str) -> list[Flaw]:
    """Run the engine's checks on one answer to a turn."""
    return checks.check_turn(
        text,
        member=turn.spec.partition(":")[0],
        nick=turn.nick,
        sources=context.sources_text(turn.system, turn.user),
        given=context.given_text(turn.user),
        known=turn.known,
        devil_line=turn.devil_line,
        round_=int(turn.spec.partition(":")[2]),
    )


def _counts(flaws: list[Flaw], text: str) -> Counter[str]:
    """Count one answer's flaws by kind, plus its update markers."""
    counts = Counter(f.kind.value for f in flaws)
    counts[MARKER] += int(dynamics.marks_update(text))
    return counts


def _table(totals: dict[str, Counter[str]], answers: Counter[str]) -> str:
    """Render the counts per model (rows) and kind (columns)."""
    short = {c: re.sub(r"_?(mention|literal|line|prefix)$", "", c)[:10] for c in COLUMNS}
    head = f"{'model':<34}{'answers':>8}" + "".join(f"{short[c]:>12}" for c in COLUMNS)
    rows = [head]
    for model, counts in totals.items():
        cells = "".join(f"{counts[c]:>12}" for c in COLUMNS)
        rows.append(f"{model:<34}{answers[model]:>8}{cells}")
    return "\n".join(rows)


def _report(model: str, turn: ReplayTurn, flaws: list[Flaw], verbose: bool) -> None:
    """Print one answer's flaws (all of them with --verbose, else a count)."""
    print(f"[{model}] {turn.spec}: {len(flaws)} flaw(s)", flush=True)
    if verbose:
        for f in flaws:
            print(f"    {f.kind.value}: {f.note + ' in ' if f.note else ''}{f.excerpt}")


def _load_manifest(path: Path) -> list[dict[str, str]]:
    """Read the manifest's turns."""
    with path.open("rb") as fh:
        turns: list[dict[str, str]] = tomllib.load(fh)["turn"]
    return turns


def _bench(args: argparse.Namespace) -> None:
    """Replay the manifest on the models and count the flaws.

    Appends to `--out` and skips the answers already in it, so a stopped run resumes
    with the same command (`--fresh` starts over).
    """
    settings = load_settings(args.root)
    entries, turns = _turns(args, settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done, has_turns = _done(args.out, args.fresh)
    totals, answers = _saved_totals(args.out)
    sessions = {id(t): e["session"] for e, t in zip(entries, turns, strict=True)}
    with args.out.open("a", encoding="utf-8") as out:
        _write_turns(out, entries, turns, has_turns)

        def keep(model: str, turn: ReplayTurn, text: str, extra: dict[str, Any]) -> None:
            """Check, count, print and save one answer."""
            flaws = _check(turn, text)
            totals.setdefault(model, Counter()).update(_counts(flaws, text))
            answers[model] += 1
            _report(model, turn, flaws, args.verbose)
            record = {"type": "answer", "model": model, "spec": turn.spec, "text": text}
            record["session"] = sessions[id(turn)]
            record |= {"flaws": [f.model_dump(mode="json") for f in flaws], **extra}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

        def todo(model: str, turn: ReplayTurn, runs: int) -> int:
            """Return how many answers this model still owes this turn."""
            return max(0, runs - done[sessions[id(turn)], turn.spec, model])

        if args.original:
            for turn in turns:
                if todo(ORIGINAL, turn, 1):
                    keep(ORIGINAL, turn, turn.original, {})
        # Model-major order: each model loads once for all its turns.
        for raw in args.models:
            model, think = model_think(settings, raw)
            for turn in turns:
                request = (turn.system, turn.user, turn.temperature, turn.max_output)
                for _ in range(todo(model, turn, args.runs)):
                    r = _run(model, think, request, args.num_ctx)
                    if "error" in r:
                        print(f"[{model}] {turn.spec}: ERROR {r['error']}", flush=True)
                        continue
                    keep(model, turn, r["message"]["content"].strip(), {"wall_s": r["wall_s"]})
    print(f"\n{_table(totals, answers)}\n\nanswers saved to {args.out}")


def _turns(
    args: argparse.Namespace, settings: Settings
) -> tuple[list[dict[str, str]], list[ReplayTurn]]:
    """Return the turns to replay: a saved run's prompts or the manifest's, rebuilt."""
    if args.prompts_from:
        saved = _saved_turns(args.prompts_from)
        entries = [{"session": session, "turn": spec} for session, spec in saved]
        return entries, list(saved.values())
    entries = _load_manifest(args.manifest)
    return entries, [build_turn(settings, Path(e["session"]), e["turn"]) for e in entries]


def _write_turns(
    out: TextIO,
    entries: list[dict[str, str]],
    turns: list[ReplayTurn],
    has_turns: set[tuple[str, str]],
) -> None:
    """Save the turns' prompts the run file doesn't have yet."""
    for entry, turn in zip(entries, turns, strict=True):
        if (entry["session"], turn.spec) not in has_turns:
            record = {"type": "turn", "session": entry["session"], **asdict(turn)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _done(path: Path, fresh: bool) -> tuple[Counter[tuple[str, str, str]], set[tuple[str, str]]]:
    """Return a run file's answers per (session, spec, model) and its saved turns.

    With `fresh`, delete the file first, so nothing counts as done.
    """
    if fresh:
        path.unlink(missing_ok=True)
    done: Counter[tuple[str, str, str]] = Counter()
    has_turns: set[tuple[str, str]] = set()
    if not path.exists():
        return done, has_turns
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["type"] == "turn":
            has_turns.add((rec["session"], rec["spec"]))
        elif rec["type"] == "answer":
            done[rec["session"], rec["spec"], rec["model"]] += 1
    if done:
        print(f"{sum(done.values())} answer(s) already done in {path}", flush=True)
    return done, has_turns


def _saved_totals(path: Path) -> tuple[dict[str, Counter[str]], Counter[str]]:
    """Count a run file's saved answers per model, from the flaws saved with them."""
    totals: dict[str, Counter[str]] = {}
    answers: Counter[str] = Counter()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in lines:
        rec = json.loads(line)
        if rec["type"] == "answer":
            counts = Counter(f["kind"] for f in rec["flaws"])
            counts[MARKER] += int(dynamics.marks_update(rec["text"]))
            totals.setdefault(rec["model"], Counter()).update(counts)
            answers[rec["model"]] += 1
    return totals, answers


def _saved_turns(path: Path) -> dict[tuple[str, str], ReplayTurn]:
    """Return a saved run's turns (prompts as they were then), keyed by (session, spec)."""
    turns: dict[tuple[str, str], ReplayTurn] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["type"] == "turn":
            fields = {k: v for k, v in rec.items() if k not in ("type", "session")}
            turns[rec["session"], rec["spec"]] = ReplayTurn(**fields)
    return turns


def _recheck(path: Path, verbose: bool) -> None:
    """Re-run the (possibly changed) checks on a saved run's answers."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    turns = _saved_turns(path)
    totals: dict[str, Counter[str]] = {}
    answers: Counter[str] = Counter()
    for rec in lines:
        if rec["type"] != "answer":
            continue
        turn = turns[rec["session"], rec["spec"]]
        flaws = _check(turn, rec["text"])
        totals.setdefault(rec["model"], Counter()).update(_counts(flaws, rec["text"]))
        answers[rec["model"]] += 1
        _report(rec["model"], turn, flaws, verbose)
    print(f"\n{_table(totals, answers)}")


def main() -> None:
    """Parse the arguments and run the bench or a recheck."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Count flaws in replayed member turns.")
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/manifest.toml"))
    parser.add_argument(
        "--models", nargs="*", default=[], help="model[=think] (default: config's think)"
    )
    parser.add_argument("--original", action="store_true", help="Check the saved answers too")
    parser.add_argument("--runs", type=int, default=1, help="Answers per model and turn")
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    parser.add_argument("--out", type=Path, default=Path(f"data/bench/runs/{stamp}.jsonl"))
    parser.add_argument(
        "--prompts-from", type=Path, default=None, help="Replay a saved run's prompts"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Overwrite --out instead of resuming it"
    )
    parser.add_argument("--recheck", type=Path, default=None, help="A saved run to re-check")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every flaw")
    args = parser.parse_args()
    if args.recheck:
        _recheck(args.recheck, args.verbose)
    else:
        _bench(args)


if __name__ == "__main__":
    main()
