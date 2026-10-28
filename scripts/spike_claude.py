"""M0 spike: get one reply from the `claude` CLI in headless mode.

Live script — consumes Claude Pro quota (one call per mode). Stdlib only, so it
runs before the project is scaffolded:

    python3 scripts/spike_claude.py                 # persona-style reply
    python3 scripts/spike_claude.py --mode schema   # structured JSON output
    python3 scripts/spike_claude.py --mode noauth   # logged-out error shape (no quota)

Findings are recorded in docs/spike-notes.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

# Never let the CLI bill API credits: drop every API-key-like variable.
_STRIPPED_ENV_PREFIXES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are Rigor, a terse former credit risk analyst on an advisory council. "
    "Answer in at most three short sentences."
)
USER_PROMPT = (
    "Introduce yourself by name. Then list the tools you can call (say 'none' if "
    "none), and say whether you can see any CLAUDE.md or project instructions."
)
SCHEMA_PROMPT = "Give your position on: 'Keep 6 months of expenses in cash?'"
SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["position", "confidence"],
}


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ without API-key variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_STRIPPED_ENV_PREFIXES)}
    env.update(extra or {})
    return env


def build_argv(binary: str, model: str, system_prompt_file: Path, schema: bool) -> list[str]:
    """Build the headless argv. No prompt content goes here — only a file path."""
    argv = [
        binary,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--system-prompt-file", str(system_prompt_file),
        "--tools", "",
        "--no-session-persistence",
        "--safe-mode",
        "--strict-mcp-config",
        "--setting-sources", "",
    ]  # fmt: skip
    if schema:
        argv += ["--json-schema", json.dumps(SCHEMA)]
    return argv


async def run_claude(
    argv: list[str], prompt: str, cwd: Path, env: dict[str, str], timeout_s: float
) -> tuple[int, str, str, float]:
    """Run the CLI with the prompt on stdin; return (rc, stdout, stderr, seconds)."""
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out.decode(), err.decode(), time.monotonic() - start


async def main() -> None:
    """Run one spike call in an empty temp dir and print the parsed result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["persona", "schema", "noauth"], default="persona")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--binary", default="claude")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="council-claude-") as tmp:
        workdir = Path(tmp) / "cwd"
        workdir.mkdir()  # stays empty: no CLAUDE.md or files leak into context
        sp_file = Path(tmp) / "system.md"
        sp_file.write_text(SYSTEM_PROMPT)
        sp_file.chmod(0o600)
        extra = {"CLAUDE_CONFIG_DIR": str(Path(tmp) / "noauth")} if args.mode == "noauth" else {}
        argv = build_argv(args.binary, args.model, sp_file, schema=args.mode == "schema")
        prompt = SCHEMA_PROMPT if args.mode == "schema" else USER_PROMPT
        rc, out, err, secs = await run_claude(argv, prompt, workdir, child_env(extra), args.timeout)

    print(f"exit={rc} wall={secs:.1f}s")
    if err.strip():
        print(f"stderr: {err.strip()[:1000]}")
    try:
        print(json.dumps(json.loads(out), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(f"non-JSON stdout: {out[:1000]!r}")


if __name__ == "__main__":
    asyncio.run(main())
