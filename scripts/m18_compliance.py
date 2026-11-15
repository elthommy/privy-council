"""Score fact-tag compliance in flaw_bench runs (M18 gate: 70 %).

Strict proxy: a sentence counts when it has a money or % figure whose value is a
money or % figure of the turn's facts section; it complies when it cites a tag.
The loose proxy (any number shared with the facts) counts "3 years" or "20 %".
Both overcount: a member's own probability often equals some fact's percentage, so
read the untagged sentences by hand before calling the gate.

Invented specifics (gate: no rise against the pre-M18 baseline): UNSOURCED figures
once the tags are stripped, so runs with and without tags are checked the same way.

    uv run python scripts/m18_compliance.py data/bench/runs/m18-vouch.jsonl [...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from privy_council.engine import checks, context, facts, ledger

_FAMILIES = ("money", "%")


def _values(text: str, strict: bool) -> set[float]:
    """Return the figure values of a text (money and % only when strict)."""
    if not strict:
        return set(checks.numbers(text))
    return {v for f in checks.figures(text) if f.family in _FAMILIES for v in f.values}


def _score(answer: str, prompt: str, strict: bool) -> tuple[int, int]:
    """Return (tagged, total) over the answer's sentences that use a fact figure."""
    known = _values(facts.facts_section(prompt), strict)
    tagged = total = 0
    for start, end in facts.sentences(answer):
        sentence = answer[start:end]
        if known & _values(sentence, strict):
            total += 1
            tagged += bool(facts.tags(sentence))
    return tagged, total


def _invented(answer: str, turn: dict[str, Any]) -> int:
    """Return the answer's UNSOURCED figures, tags stripped, estimate lines blanked."""
    text = ledger.blank_lines(facts.TAG.sub("", answer))
    flaws = checks.figure_flaws(
        text,
        context.sources_text(turn["system"], turn["user"]),
        context.given_text(turn["user"]),
        turn["spec"].partition(":")[0],
    )
    return sum(f.kind == checks.FlawKind.UNSOURCED for f in flaws)


def _load(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Return a run's turns by (session, spec) and its answers."""
    prompts: dict[tuple[str, str], dict[str, Any]] = {}
    answers: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["type"] == "turn":
            prompts[(row["session"], row["spec"])] = row
        elif row["type"] == "answer":
            answers.append(row)
    return prompts, answers


def main() -> None:
    """Print compliance and invented figures per model for each run given."""
    for arg in sys.argv[1:]:
        prompts, answers = _load(Path(arg))
        print(arg)
        for strict in (True, False):
            score: dict[str, Counter[str]] = {}
            for a in answers:
                c = score.setdefault(a["model"], Counter())
                turn = prompts[(a["session"], a["spec"])]
                tagged, total = _score(a["text"], turn["user"], strict)
                if strict:
                    c["invented"] += _invented(a["text"], turn)
                c["tagged"] += tagged
                c["total"] += total
                c["answers"] += 1
            name = "strict" if strict else "loose"
            for model, c in score.items():
                share = c["tagged"] / c["total"] if c["total"] else 0.0
                count = f"{c['tagged']:3}/{c['total']:<3}"
                extra = f"({c['answers']} answers)"
                extra += f"  invented {c['invented']}" if strict else ""
                print(f"  {name:6} {model:28} {count} {share:4.0%}  {extra}")


if __name__ == "__main__":
    main()
