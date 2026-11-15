"""M19 bench: let each model fix its own flagged drafts, as `repair_turns` would.

Takes the answers of saved flaw_bench runs, re-checks them with today's checks, and
for every draft with a repairable flag (models.flaw.REPAIR_KINDS) sends the engine's
fix call to the model that wrote it: the same system and user prompt, the draft as
its answer, then `prompts.repair_task`. The rewrite is checked the same way; the
engine would post it only with fewer repairable flags. Read each pair by hand: fixed,
unchanged, or worse (a new invented claim, lost content).

    uv run python -u scripts/repair_bench.py data/bench/runs/m18-vouch.jsonl \\
        --out data/bench/runs/m19-repair.jsonl

`--seed N` adds, per model, up to N of its own clean drafts with one fault planted:
a tag moved to a fact that doesn't hold the figure, or (forecast turns, N/2) the
estimate line removed. Natural flagged drafts are rare (20 in all saved runs on
2026-11-15), so the seeds give each model a sample; the clean draft is kept as
`original`, which a good fix should match.

Appends to --out and skips the drafts already done, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from flaw_bench import _check, _estimating, _saved_turns
from replay_turn import OLLAMA, ReplayTurn, model_think

from privy_council.config import load_settings
from privy_council.engine import checks, context, facts, prompts
from privy_council.models.flaw import REPAIR_KINDS, Flaw, FlawKind
from privy_council.util.language import Language, detect_language


def _repairable(turn: ReplayTurn, text: str) -> list[Flaw]:
    """Return a text's flaws that a fix call would be made for."""
    return [f for f in _check(turn, text) if f.kind in REPAIR_KINDS]


def _fix(
    model: str, think: bool | str | None, turn: ReplayTurn, draft: str, task: str
) -> dict[str, Any]:
    """Send the fix call; return Ollama's response plus wall time."""
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "10m",
        "messages": [
            {"role": "system", "content": turn.system},
            {"role": "user", "content": turn.user},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": task},
        ],
        "options": {
            "temperature": turn.temperature,
            "num_ctx": 12288,
            "num_predict": turn.max_output,
        },
    }
    if think is not None:
        body["think"] = think
    start = time.monotonic()
    response: dict[str, Any] = httpx.post(OLLAMA, json=body, timeout=1800).json()
    response["wall_s"] = time.monotonic() - start
    return response


def _key(model: str, session: str, spec: str, draft: str) -> str:
    """Return a draft's identity in the output file."""
    digest = hashlib.sha1(draft.encode()).hexdigest()[:12]
    return f"{model}|{session}|{spec}|{digest}"


def _done(path: Path) -> set[str]:
    """Return the keys already in the output file."""
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


def _language(turn: ReplayTurn) -> Language:
    """Return the debate language the turn's prompt is written in."""
    return detect_language(turn.user)


def _summary(path: Path) -> None:
    """Print per model: drafts, rewrites the engine would post, flags before/after, time."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    wall: dict[str, float] = defaultdict(float)
    for r in rows:
        c = by_model[r["model"]]
        c["drafts"] += 1
        c["posted"] += r["posted"] == "fixed"
        c["clean"] += not r["fixed_flaws"]
        c["flags_before"] += len(r["flaws"])
        c["flags_after"] += len(r["fixed_flaws"])
        wall[r["model"]] += r.get("wall_s", 0.0)
    print(f"\n{'model':28} drafts  posted  clean  flags before->after  s/fix")
    for model, c in by_model.items():
        mean = wall[model] / c["drafts"] if c["drafts"] else 0
        print(
            f"{model:28} {c['drafts']:6}  {c['posted']:6}  {c['clean']:5}  "
            f"{c['flags_before']:6} -> {c['flags_after']:<10} {mean:5.0f}"
        )


Case = dict[str, Any]


def _moved_tag(turn: ReplayTurn, text: str) -> str | None:
    """Return the text with its first right citation moved to a fact lacking the figure."""
    shown = facts.in_prompt(facts.facts_section(context.given_text(turn.user)))
    for start, end, ids in checks.vouched(text, facts.tagged_sentences(text)):
        figure = text[start:end]
        if not all(i in shown and checks._holds(shown[i], figure) for i in ids):
            continue
        other = next(
            (i for i in shown if i not in ids and not checks._holds(shown[i], figure)), None
        )
        tag = facts.TAG.search(text, end)
        if other is None or tag is None:
            continue
        seeded = f"{text[: tag.start()]}[{other}]{text[tag.end() :]}"
        if any(f.kind is FlawKind.MISCITED for f in _repairable(turn, seeded)):
            return seeded
    return None


def _no_estimate(turn: ReplayTurn, text: str) -> str | None:
    """Return the text without its estimate line, or None if it has none to remove."""
    if not _estimating(turn):
        return None
    labels = [v["estimate"] for v in prompts.LABELS.values()]
    lines = [line for line in text.splitlines() if not any(lab in line for lab in labels)]
    seeded = "\n".join(lines).strip()
    flagged = any(f.kind is FlawKind.MISSING_ESTIMATE for f in _repairable(turn, seeded))
    return seeded if seeded != text.strip() and flagged else None


def _cases(runs: list[Path], wanted: dict[str, str], seed: int) -> list[Case]:
    """Return the flagged drafts of the runs, plus `seed` seeded ones per model."""
    natural: list[Case] = []
    clean: dict[str, list[Case]] = defaultdict(list)
    for run in runs:
        turns = _saved_turns(run)
        for line in run.read_text().splitlines():
            row = json.loads(line)
            if row["type"] != "answer" or (wanted and row["model"] not in wanted):
                continue
            case = {"model": row["model"], "session": row["session"], "draft": row["text"]}
            case["turn"] = turns[row["session"], row["spec"]]
            flaws = _repairable(case["turn"], row["text"])
            if flaws:
                natural.append({**case, "flaws": flaws, "seeded": "", "original": ""})
            else:
                clean[row["model"]].append(case)
    seeded: list[Case] = []
    for cases in clean.values():
        for kind, make, cap in (("tag", _moved_tag, seed), ("estimate", _no_estimate, seed // 2)):
            made = 0
            for case in cases:
                text = make(case["turn"], case["draft"]) if made < cap else None
                if text is not None:
                    flaws = _repairable(case["turn"], text)
                    seeded.append(
                        {
                            **case,
                            "draft": text,
                            "flaws": flaws,
                            "seeded": kind,
                            "original": case["draft"],
                        }
                    )
                    made += 1
    return natural + seeded


def _run_case(case: Case, wanted: dict[str, str], settings: Any) -> Case | None:
    """Send one fix call and return the record to save (None on an Ollama error)."""
    turn: ReplayTurn = case["turn"]
    _, think = model_think(settings, wanted.get(case["model"], f"{case['model']}=false"))
    task = prompts.repair_task(case["flaws"], _language(turn))
    r = _fix(case["model"], think, turn, case["draft"], task)
    if "error" in r:
        print(f"[{case['model']}] {turn.spec}: ERROR {r['error']}", flush=True)
        return None
    fixed = r["message"]["content"].strip()
    fixed_flaws = _repairable(turn, fixed) if fixed else case["flaws"]
    posted = "fixed" if fixed and len(fixed_flaws) < len(case["flaws"]) else "draft"
    return {
        "key": _key(case["model"], case["session"], turn.spec, case["draft"]),
        "model": case["model"],
        "session": case["session"],
        "spec": turn.spec,
        "seeded": case["seeded"],
        "original": case["original"],
        "draft": case["draft"],
        "flaws": [f.model_dump(mode="json") for f in case["flaws"]],
        "fixed": fixed,
        "fixed_flaws": [f.model_dump(mode="json") for f in fixed_flaws],
        "posted": posted,
        "wall_s": r["wall_s"],
    }


def main() -> None:
    """Fix every flagged (or seeded) draft of the given runs on its own model."""
    parser = argparse.ArgumentParser(description="M19 repair bench.")
    parser.add_argument("runs", nargs="*", type=Path, help="flaw_bench run files")
    parser.add_argument("--out", type=Path, default=Path("data/bench/runs/m19-repair.jsonl"))
    parser.add_argument("--models", nargs="*", default=[], help="only these models (with =think)")
    parser.add_argument("--seed", type=int, default=0, help="seeded drafts per model and kind")
    parser.add_argument("--summary", action="store_true", help="only print the summary")
    args = parser.parse_args()
    if args.summary:
        _summary(args.out)
        return
    settings = load_settings(None)
    wanted = {m.partition("=")[0]: m for m in args.models}
    done = _done(args.out)
    todo = []
    for case in _cases(args.runs, wanted, args.seed):
        key = _key(case["model"], case["session"], case["turn"].spec, case["draft"])
        if key not in done:
            done.add(key)
            todo.append(case)
    todo.sort(key=lambda c: c["model"])  # model-major: each model loads once
    print(f"{len(todo)} drafts to fix", flush=True)
    with args.out.open("a", encoding="utf-8") as out:
        for case in todo:
            record = _run_case(case, wanted, settings)
            if record is None:
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            counts = f"{len(record['flaws'])} -> {len(record['fixed_flaws'])}"
            how = f"{record['posted']}, {record['wall_s']:.0f}s, {record['seeded'] or 'natural'}"
            print(f"[{case['model']}] {record['spec']}: {counts} ({how})", flush=True)
    _summary(args.out)


if __name__ == "__main__":
    main()
