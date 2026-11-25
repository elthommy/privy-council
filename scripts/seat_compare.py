"""Run frozen member prompts on several models, to separate model size from prompt faults.

Prompts come from JSON files ({"macro:3": {system, user, temperature, max_output}}), one
per arm, as a replay built them; each answer is appended to a JSONL file at once, so an
interrupted run loses one call. No repair pass: the raw answers are compared.

    uv run python -u scripts/seat_compare.py --prompts old.json new.json \\
        --models nemotron-3.5-bolt:latest sonnet --runs 4 --out runs.jsonl --root SCRATCH

A model with no ":" runs on the claude CLI (quota); the others on Ollama. Use a --root
whose data dir is a scratch dir: calls go into its usage.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from privy_council.backends.base import CallTag, ChatMessage, CompletionRequest
from privy_council.backends.registry import BackendRegistry
from privy_council.config import load_settings
from privy_council.engine.budget import Ledger
from privy_council.models.persona import BackendKind


def _done(out: Path) -> set[tuple[str, str, str, int]]:
    """Return the (arm, model, turn, run) already in the output file."""
    if not out.exists():
        return set()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    return {(r["arm"], r["model"], r["turn"], r["run"]) for r in rows}


async def _answer(registry: BackendRegistry, model: str, prompt: dict[str, Any]) -> str:
    """Send one prompt to a model; return the text."""
    kind = BackendKind.OLLAMA if ":" in model else BackendKind.CLAUDE_CLI
    request = CompletionRequest(
        system=prompt["system"],
        messages=[ChatMessage(role="user", content=prompt["user"])],
        model=model,
        temperature=prompt["temperature"],
        max_output_tokens=prompt["max_output"],
        tag=CallTag(session="seat-compare", purpose="turn"),
    )
    return (await registry.backend(kind).complete(request)).text


async def main() -> None:
    """Run every prompt on every model, model by model."""
    parser = argparse.ArgumentParser(description="Compare models on frozen member prompts.")
    parser.add_argument("--prompts", nargs="+", type=Path, required=True, help="one file per arm")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="scratch config root")
    args = parser.parse_args()
    settings = load_settings(args.root)
    registry = BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir))
    done = _done(args.out)
    for model in args.models:
        for path in args.prompts:
            for turn, prompt in json.loads(path.read_text()).items():
                for run in range(args.runs):
                    if (path.stem, model, turn, run) in done:
                        continue
                    text = await _answer(registry, model, prompt)
                    row = {"arm": path.stem, "model": model, "turn": turn, "run": run}
                    with args.out.open("a") as f:
                        f.write(json.dumps({**row, "text": text}, ensure_ascii=False) + "\n")
                    print(path.stem, model, turn, run, len(text))
    await registry.aclose()


if __name__ == "__main__":
    asyncio.run(main())
