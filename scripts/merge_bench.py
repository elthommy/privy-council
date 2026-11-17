"""Run the private-notes merge on local models (no Claude calls): does it fit the cap?

Copies a notes file into a scratch memory store and runs `MemoryWriter.consolidate`
for one persona and council, the way the engine does at a session's close (first
call, one retry, then the keep-the-newest fallback). Prints, per run, each reply's
size and note count and the outcome. The real memory files are never written.

    uv run python scripts/merge_bench.py data/memory/personas/macro/notes.md \\
        --persona macro --council economics [--models nemotron-3.5-bolt:latest] [--runs 2]

Without --models the persona's own model is used.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from privy_council.backends.base import ChatMessage, CompletionRequest, CompletionResult
from privy_council.backends.ollama import OllamaBackend
from privy_council.config import load_settings
from privy_council.engine import context
from privy_council.loaders.personas import load_personas
from privy_council.memory.distill import MemoryWriter
from privy_council.memory.store import MemoryStore, bullet_lines, size_bytes
from privy_council.models.persona import Persona


async def _run_once(
    notes: Path, persona: Persona, council: str, model: str, backend: OllamaBackend
) -> None:
    """Merge a scratch copy of the notes once and print what each call returned."""
    settings = load_settings()
    root = Path(tempfile.mkdtemp(prefix="merge-bench-"))
    try:
        store = MemoryStore(root)
        target = store.private_path(persona.slug)
        target.parent.mkdir(parents=True)
        shutil.copy(notes, target)

        async def complete(
            persona: Persona,
            prompt: context.Prompt,
            purpose: str,
            *,
            json_schema: dict[str, Any] | None = None,
            max_output: int | None = None,
        ) -> CompletionResult:
            """Call the local model as the engine would, and print the reply's size."""
            request = CompletionRequest(
                system=prompt.system,
                messages=[ChatMessage(role="user", content=prompt.user)],
                model=model,
                temperature=persona.meta.temperature,
                max_output_tokens=max_output or 1500,
            )
            start = time.monotonic()
            result = await backend.complete(request)
            lines = bullet_lines(result.text)
            longest = max((len(line.split()) for line in lines), default=0)
            print(
                f"  call: {time.monotonic() - start:.0f}s, {size_bytes(result.text)} bytes, "
                f"{len(lines)} notes, longest {longest} words, truncated={result.truncated}"
            )
            return result

        writer = MemoryWriter(store, settings.memory, complete, lambda _: 8000)
        before = writer.council_notes(persona, council)
        outcome = await writer.consolidate(persona, council)
        after = writer.council_notes(persona, council)
        print(
            f"  outcome: {outcome}; {size_bytes(before)} -> {size_bytes(after)} bytes, "
            f"{len(bullet_lines(before))} -> {len(bullet_lines(after))} notes"
        )
    finally:
        shutil.rmtree(root)


async def main() -> None:
    """Parse the arguments and run the merges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", type=Path)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--council", required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    settings = load_settings()
    personas, _ = load_personas(settings.personas_dir)
    persona = personas[args.persona]
    backend = OllamaBackend(settings.ollama)
    try:
        for model in args.models or [persona.meta.model]:
            for n in range(1, args.runs + 1):
                print(f"--- {args.persona} on {model}, run {n}")
                await _run_once(args.notes, persona, args.council, model, backend)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
