"""Spike: enforce a hard output cap on the `claude` CLI (no max-tokens flag exists).

Live script — consumes Claude Pro quota (one call per run):

    python3 scripts/spike_claude_stream.py --mode cut --cap-chars 400   # stream, kill at cap
    python3 scripts/spike_claude_stream.py --mode schema                 # stream + --json-schema
    python3 scripts/spike_claude_stream.py --mode envcap   # CLAUDE_CODE_MAX_OUTPUT_TOKENS=100

Findings are recorded in docs/spike-notes.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from spike_claude import build_argv, child_env

SYSTEM_PROMPT = "You are a verbose essayist."
LONG_PROMPT = "Write a 600-word essay about rivers."
STREAM_FLAGS = ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]


def stream_argv(binary: str, model: str, sp_file: Path, schema: bool) -> list[str]:
    """Reuse the verified M0 argv, swapping json output for stream-json."""
    argv = build_argv(binary, model, sp_file, schema)
    i = argv.index("--output-format")
    del argv[i : i + 2]
    return argv + STREAM_FLAGS


def summarize_event(event: dict[str, Any], seen: set[str]) -> None:
    """Print the first occurrence of each interesting event type."""
    inner = event.get("event") or {}
    key = f"{event.get('type')}/{inner.get('type') or event.get('subtype')}"
    delta = (inner.get("delta") or {}).get("type")
    key += f"/{delta}" if delta else ""
    if key not in seen:
        seen.add(key)
        print(f"  first {key} at t={time.monotonic():.1f}")


async def run_stream(argv: list[str], cwd: Path, cap_chars: int | None) -> None:
    """Stream events; kill the process once text output exceeds cap_chars."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=child_env(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(LONG_PROMPT.encode())
    proc.stdin.close()
    text, seen, start, killed = "", set[str](), time.monotonic(), False
    async for raw in proc.stdout:
        event = json.loads(raw)
        summarize_event(event, seen)
        inner = event.get("event") or {}
        delta = inner.get("delta") or {}
        if delta.get("type") == "text_delta":
            text += delta["text"]
        elif delta.get("type") == "input_json_delta":
            text += delta.get("partial_json", "")
        if event.get("type") == "result":
            print("  result:", {k: event.get(k) for k in ("is_error", "stop_reason", "num_turns")})
            print("  structured_output:", event.get("structured_output"))
        if cap_chars and len(text) > cap_chars and not killed:
            proc.kill()
            killed = True
            break
    rc = await proc.wait()
    print(f"exit={rc} killed={killed} wall={time.monotonic() - start:.1f}s chars={len(text)}")
    print(f"text tail: ...{text[-200:]!r}")


async def run_envcap(binary: str, model: str, sp_file: Path, cwd: Path, cap: int) -> None:
    """Plain JSON call with CLAUDE_CODE_MAX_OUTPUT_TOKENS set; print the envelope."""
    argv = build_argv(binary, model, sp_file, schema=False)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=child_env({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(cap)}),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(LONG_PROMPT.encode())
    print(f"exit={proc.returncode} stderr={err.decode()[:300]!r}")
    env = json.loads(out)
    keys = ("is_error", "subtype", "stop_reason", "terminal_reason", "api_error_status")
    print({k: env.get(k) for k in keys})
    print("usage.output_tokens:", env.get("usage", {}).get("output_tokens"))
    print(f"result ({len(env.get('result', ''))} chars): {env.get('result', '')[:300]!r}")


async def main() -> None:
    """Dispatch on --mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cut", "schema", "envcap"], default="cut")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--binary", default="claude")
    parser.add_argument("--cap-chars", type=int, default=400)
    parser.add_argument("--cap-tokens", type=int, default=100)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="council-claude-") as tmp:
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()
        sp_file = Path(tmp) / "system.md"
        sp_file.write_text(SYSTEM_PROMPT)
        if args.mode == "envcap":
            await run_envcap(args.binary, args.model, sp_file, cwd, args.cap_tokens)
            return
        schema = args.mode == "schema"
        argv = stream_argv(args.binary, args.model, sp_file, schema)
        await run_stream(argv, cwd, None if schema else args.cap_chars)


if __name__ == "__main__":
    asyncio.run(main())
