"""Score local Chairs on saved rounds against the reference labels.

`chair_bench.py` replays a round's update with Chair reseated on a local model and
saves the flaws it reports. This script picks the rounds and scores those flaws
against the labels of `label_rounds.py`:

- caught: a flaw on the right member whose quote matches a known error, that is a
  flag the labeller called real (raised by the saved Chair) or an error no flag
  caught. Real engine flags don't count: the replayed Chair is shown them already.
- false alarm: a flaw matching a flag the labeller called false;
- unlabelled: any other flaw (a real error the labeller missed, or noise: read them).

The saved Chair's own flags give the reference row (Claude's Chair, or Super's in
local-only sessions). Real sessions go in, so the manifest and results stay in
data/bench/ (gitignored). No Claude call; the GPU work is chair_bench.py's.

    uv run python scripts/chair_label_bench.py manifest --rounds 16 \\
        --out data/bench/chair_manifest_labels.toml
    uv run python -u scripts/chair_bench.py --manifest data/bench/chair_manifest_labels.toml \\
        --models nemotron-3.5-bolt:latest --num-ctx 24576 --out data/bench/runs/chair-labels.jsonl
    uv run python scripts/chair_label_bench.py score data/bench/runs/chair-labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = Path("data/bench/labels/full.jsonl")
SESSIONS = Path("data/sessions")
OVERLAP = 0.5  # share of the shorter quote's words the other must hold
PER_SESSION = 2  # rounds per session at most, for variety


@dataclass(frozen=True)
class Known:
    """One labelled passage: a known error (real) or a false alarm (false)."""

    member: str
    quote: str
    real: bool
    source: str  # "moderator", "engine" or "missed"


def words(text: str) -> list[str]:
    """Return the lower-cased words and numbers of a text."""
    return re.findall(r"\w+(?:[.,]\d+)*", text.lower())


def same_passage(a: str, b: str) -> bool:
    """Return True when two quotes point at the same passage."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    short, other = (wa, set(wb)) if len(wa) <= len(wb) else (wb, set(wa))
    return sum(w in other for w in short) / len(short) >= OVERLAP


def known_passages(row: dict[str, Any]) -> list[Known]:
    """Return a labelled round's known errors and false alarms."""
    out = []
    for f in row["flags"]:
        if f["verdict"] == "unsure":
            continue
        flag = row["flags_in"][f["id"]]
        out.append(Known(flag["member"], flag["excerpt"], f["verdict"] == "real", flag["by"]))
    for m in row["missed"]:
        if m.get("quote_found"):
            out.append(Known(m["speaker"], m["quote"], True, "missed"))
    return out


def targets(known: list[Known]) -> list[Known]:
    """Return the errors a replayed Chair can be credited for (not engine-flagged)."""
    return [k for k in known if k.real and k.source != "engine"]


def score_flaws(flaws: list[dict[str, Any]], known: list[Known]) -> dict[str, Any]:
    """Return what a Chair's flaws caught, falsely flagged, or left unlabelled."""
    caught: set[int] = set()
    false_alarms, unlabelled = 0, []
    goals = targets(known)
    for f in flaws:
        match = [
            i
            for i, k in enumerate(goals)
            if k.member == f["member"] and same_passage(f["quote"], k.quote)
        ]
        if match:
            caught.update(match)
        elif any(
            not k.real and k.member == f["member"] and same_passage(f["quote"], k.quote)
            for k in known
        ):
            false_alarms += 1
        elif not any(
            k.real and k.member == f["member"] and same_passage(f["quote"], k.quote) for k in known
        ):
            unlabelled.append(f)
    return {
        "targets": len(goals),
        "caught": len(caught),
        "false_alarms": false_alarms,
        "unlabelled": unlabelled,
    }


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Return the label rows by key (session:round)."""
    return {r["key"]: r for r in map(json.loads, path.read_text().splitlines())}


def pick_rounds(labels: dict[str, dict[str, Any]], n: int) -> list[str]:
    """Pick n rounds with the most targets, councils in turn, PER_SESSION per session."""
    by_council: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for key, row in labels.items():
        by_council[key.split("_")[1]].append((len(targets(known_passages(row))), key))
    for rounds in by_council.values():
        rounds.sort(reverse=True)
    chosen: list[str] = []
    per_session: dict[str, int] = defaultdict(int)
    while len(chosen) < n and any(by_council.values()):
        for rounds in sorted(by_council.values(), key=lambda r: -r[0][0] if r else 0):
            while rounds and len(chosen) < n:
                _, key = rounds.pop(0)
                session = key.rsplit(":", 1)[0]
                if per_session[session] < PER_SESSION:
                    per_session[session] += 1
                    chosen.append(key)
                    break
    return chosen


def write_manifest(keys: list[str], sessions: Path, out: Path) -> None:
    """Write a chair_bench.py manifest of round items."""
    lines = ["# Chair vs the reference labels (scripts/chair_label_bench.py; gitignored).", ""]
    for key in keys:
        session, rnd = key.rsplit(":", 1)
        lines += ["[[round]]", f'session = "{sessions / session}"', f"round = {rnd}", ""]
    out.write_text("\n".join(lines))


def item_key(item: str) -> str:
    """Return the label key of a chair_bench item id (round:SESSION:ROUND)."""
    return item.split(":", 1)[1]


def report(results: Path, labels: dict[str, dict[str, Any]]) -> str:
    """Return a per-model table: JSON, caught, false alarms, unlabelled, seconds."""
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        label = labels.get(item_key(r["item"]))
        if label is None or not r["item"].startswith("round:"):
            continue
        t = totals[r["model"]]
        t["rounds"] += 1
        if r.get("flaws") is None:
            t["no_flaws"] += 1
            continue
        s = score_flaws(r["flaws"], known_passages(label))
        for k in ("targets", "caught", "false_alarms"):
            t[k] += s[k]
        t["unlabelled"] += len(s["unlabelled"])
        t["json_first"] += r.get("json") == "first"
        t["seconds"] += r.get("seconds") or 0
    cols = ("rounds", "json 1st", "caught", "false", "unlab.", "s/round")
    head = f"{'model':34} " + " ".join(f"{c:>8}" for c in cols)
    lines = [head]
    for model, t in totals.items():
        scored = t["rounds"] - t["no_flaws"]
        rate = f"{int(t['caught'])}/{int(t['targets'])}"
        sec = t["seconds"] / scored if scored else 0
        lines.append(
            f"{model:34} {int(t['rounds']):>8} {int(t['json_first']):>8} {rate:>8} "
            f"{int(t['false_alarms']):>8} {int(t['unlabelled']):>8} {sec:>8.0f}"
        )
    return "\n".join(lines)


def unlabelled(results: Path, labels: dict[str, dict[str, Any]], model: str) -> str:
    """Return a model's unlabelled flaws as Markdown, to read by hand."""
    out = []
    for r in map(json.loads, results.read_text().splitlines()):
        label = labels.get(item_key(r["item"]))
        if r["model"] != model or label is None or not r.get("flaws"):
            continue
        for f in score_flaws(r["flaws"], known_passages(label))["unlabelled"]:
            where = f"{item_key(r['item'])} · {f['member']} · {f['kind']}"
            out.append(f'- {where}: "{f["quote"]}" — {f["note"]}')
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Score local Chairs against the reference labels.")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="pick rounds, write a chair_bench manifest")
    m.add_argument("--rounds", type=int, default=16)
    m.add_argument("--out", type=Path, default=Path("data/bench/chair_manifest_labels.toml"))
    s = sub.add_parser("score", help="score a chair_bench results file")
    s.add_argument("results", type=Path)
    s.add_argument("--unlabelled", metavar="MODEL", help="list a model's unlabelled flaws")
    for q in (m, s):
        q.add_argument("--labels", type=Path, default=LABELS)
    m.add_argument("--sessions", type=Path, default=SESSIONS)
    return p.parse_args()


def main() -> None:
    """Write a manifest or score results."""
    args = parse_args()
    labels = load_labels(args.labels)
    if args.cmd == "manifest":
        keys = pick_rounds(labels, args.rounds)
        write_manifest(keys, args.sessions, args.out)
        n = sum(len(targets(known_passages(labels[k]))) for k in keys)
        print(f"{len(keys)} rounds, {n} known errors to catch -> {args.out}")
    elif args.unlabelled:
        print(unlabelled(args.results, labels, args.unlabelled))
    else:
        print(report(args.results, labels))


if __name__ == "__main__":
    main()
