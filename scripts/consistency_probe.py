"""M22 step 0: do invented figures reappear less often across drafts than sourced ones?

Offline, no model call. Reads `flaw_bench.py` run files that hold two or more
answers by one model to the same turn, and for each figure of each answer counts
whether the model's other answers to that turn state it too.

    uv run python scripts/consistency_probe.py data/bench/runs/m18-vouch.jsonl [...]

A figure is "sourced" when each of its values is a number of the prompt (facts,
dossier, debate state, earlier messages), "unsourced" otherwise. Percentages are
counted apart: in forecast councils most unsourced ones are the member's own
probability, which is no invention. "kept by 2+" is the share a merge rule
"keep a figure at least two drafts state" would keep, on turns with 3 answers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from privy_council.engine import checks

Key = tuple[str, tuple[float, ...]]
KINDS = ("sourced", "unsourced", "sourced %", "unsourced %")


def _keys(text: str, known: set[float]) -> dict[Key, str]:
    """Return an answer's figures (family, values), each with its kind."""
    out: dict[Key, str] = {}
    for fig in checks.figures(text):
        kind = "sourced" if all(v in known for v in fig.values) else "unsourced"
        out[(fig.family, fig.values)] = kind + (" %" if fig.family == "%" else "")
    return out


def _load(paths: list[Path]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, ...], list[str]]]:
    """Return the prompts by (session, spec) and the answers by (session, spec, model)."""
    prompts: dict[tuple[str, str], str] = {}
    answers: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            turn = (row["session"], row["spec"])
            if row["type"] == "turn":
                prompts[turn] = f"{row['system']}\n{row['user']}"
            elif row["type"] == "answer" and row.get("text"):
                answers[(*turn, row["model"])].append(row["text"])
    return prompts, answers


def main() -> None:
    """Print, per model and kind of figure, how often it reappears in another draft."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    prompts, answers = _load(args.runs)
    # model -> kind -> [pairs, pairs where the other draft has it, figures, kept by 2+]
    tally: dict[str, dict[str, list[int]]] = defaultdict(lambda: {k: [0, 0, 0, 0] for k in KINDS})
    turns: dict[str, int] = defaultdict(int)
    for (session, spec, model), texts in answers.items():
        if len(texts) < 2 or (session, spec) not in prompts:
            continue
        turns[model] += 1
        known = set(checks.numbers(prompts[(session, spec)]))
        drafts = [_keys(t, known) for t in texts]
        for i, draft in enumerate(drafts):
            others = [d for j, d in enumerate(drafts) if j != i]
            for key, kind in draft.items():
                row = tally[model][kind]
                row[0] += len(others)
                row[1] += sum(key in d for d in others)
                if len(drafts) >= 3:
                    row[2] += 1
                    row[3] += any(key in d for d in others)
    for model, kinds in tally.items():
        print(f"\n{model} ({turns[model]} turns with 2+ answers)")
        for kind, (pairs, again, figs, kept) in kinds.items():
            share = f"{again / pairs:.0%}" if pairs else "-"
            keep = f", kept by 2+ of 3: {kept}/{figs}" if figs else ""
            print(f"  {kind:12} reappears {again}/{pairs} ({share}){keep}")


if __name__ == "__main__":
    main()
