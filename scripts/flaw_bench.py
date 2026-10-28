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
from typing import Any

from replay_turn import ReplayTurn, _run, _think, build_turn

from privy_council.config import load_settings
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
    """Replay the manifest on the models and count the flaws."""
    if args.prompts_from:
        saved = _saved_turns(args.prompts_from)
        entries = [{"session": session, "turn": spec} for session, spec in saved]
        turns = list(saved.values())
    else:
        settings = load_settings(args.root)
        entries = _load_manifest(args.manifest)
        turns = [build_turn(settings, Path(e["session"]), e["turn"]) for e in entries]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sessions = {id(t): e["session"] for e, t in zip(entries, turns, strict=True)}
    totals: dict[str, Counter[str]] = {}
    answers: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as out:
        for entry, turn in zip(entries, turns, strict=True):
            record = {"type": "turn", "session": entry["session"], **asdict(turn)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        if args.original:
            for turn in turns:
                keep(ORIGINAL, turn, turn.original, {})
        # Model-major order: each model loads once for all its turns.
        for raw in args.models:
            model, _, think = raw.partition("=")
            for turn in turns:
                request = (turn.system, turn.user, turn.temperature, turn.max_output)
                for _ in range(args.runs):
                    r = _run(model, _think(think or None), request, args.num_ctx)
                    if "error" in r:
                        print(f"[{model}] {turn.spec}: ERROR {r['error']}", flush=True)
                        continue
                    keep(model, turn, r["message"]["content"].strip(), {"wall_s": r["wall_s"]})
    print(f"\n{_table(totals, answers)}\n\nanswers saved to {args.out}")


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
    parser.add_argument("--models", nargs="*", default=[], help="model[=think]")
    parser.add_argument("--original", action="store_true", help="Check the saved answers too")
    parser.add_argument("--runs", type=int, default=1, help="Answers per model and turn")
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    parser.add_argument("--out", type=Path, default=Path(f"data/bench/runs/{stamp}.jsonl"))
    parser.add_argument(
        "--prompts-from", type=Path, default=None, help="Replay a saved run's prompts"
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
