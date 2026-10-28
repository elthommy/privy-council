"""Drive a live `--no-tui` session from a command file, for local and live checks.

The REPL reads one line at a time from stdin; this script feeds it the lines
appended to ROOT/cmds, so a check can be steered between rounds ("/as Ana ...",
"/step", "/quit") by another shell, with no terminal attached. Output goes to
ROOT/out.log. Runs on a scratch root, so the real data/ stays untouched.

    uv run python scripts/live_session.py setup ROOT --council finance --max-rounds 3
    uv run python scripts/live_session.py start ROOT -- new finance "question" --local-only
    uv run python scripts/live_session.py send ROOT "/as Ana I can put in 9,000 EUR" /step
    uv run python scripts/live_session.py send ROOT /step --wait   # until back at the prompt
    uv run python scripts/live_session.py wait ROOT                 # e.g. the opening after start
    uv run python scripts/live_session.py start ROOT -- resume SESSION_ID

`start` blocks until the session exits (run it in the background). It empties
ROOT/cmds first, so `send` lines meant for it go after it starts. Soft stop: send
`/quit` at a round boundary; `/step` already pauses after each round.

The REPL prints no "next round" line after the opening: it just waits at its `> `
prompt for a `/step`. `wait` (or `send --wait`) returns once the output has grown
and ends at that prompt, or the session has ended, so a driver never sits idle.
SearXNG isn't managed from a scratch root (no scripts/ there): start it by hand.
config.local.toml is copied without its [chat] tables (the Discord bot token):
the REPL never needs them.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
COPIED = ("config.toml", "personas", "councils")
LOCAL_CONFIG = "config.local.toml"
_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?")
POLL_S = 0.5
PROMPT = "> "
ENDED = ("Session files:", "Resume with:")
IDLE_S = 2.0  # the prompt must stay last this long: output can arrive in bursts


def setup(root: Path, council: str, max_rounds: int | None, memory: str) -> None:
    """Copy the config, personas and councils into ROOT and override one council."""
    root.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        src, dst = REPO / name, root / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, dst)
    if (REPO / LOCAL_CONFIG).exists():
        text = without_chat((REPO / LOCAL_CONFIG).read_text(encoding="utf-8"))
        dst = root / LOCAL_CONFIG
        dst.touch(mode=0o600)
        dst.write_text(text, encoding="utf-8")
    path = root / "councils" / f"{council}.yaml"
    preset = yaml.safe_load(path.read_text(encoding="utf-8"))
    if max_rounds is not None:
        preset.setdefault("protocol", {})["max_rounds"] = max_rounds
    preset["memory"] = memory
    path.write_text(yaml.safe_dump(preset, sort_keys=False), encoding="utf-8")
    print(f"{root}: {council} with max_rounds={max_rounds or 'unchanged'}, memory={memory}")


def without_chat(text: str) -> str:
    """Return a config.local.toml without its [chat] and [chat.*] tables.

    Line-based, so the owner's comments and layout stay; the result is parsed to make
    sure nothing under `chat` is left (a `chat = {...}` inline table, say).
    """
    kept: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if header := _HEADER.match(line):
            name = header[1].strip().strip('"')
            skipping = name == "chat" or name.startswith("chat.")
        if not skipping:
            kept.append(line)
    out = "".join(kept)
    if "chat" in tomllib.loads(out):
        raise SystemExit(f"{LOCAL_CONFIG}: a chat setting outside a [chat] table; not copied")
    return out


def send(root: Path, lines: list[str]) -> None:
    """Append lines to ROOT/cmds, for the running session to read."""
    with (root / "cmds").open("a", encoding="utf-8") as fh:
        fh.writelines(f"{line}\n" for line in lines)


def _log_size(root: Path) -> int:
    """Return the size of ROOT/out.log (0 before the session writes it)."""
    log = root / "out.log"
    return log.stat().st_size if log.exists() else 0


def _prompts(text: str) -> int:
    """Count the prompts in a stretch of output: one per command line the REPL read."""
    return len(re.findall(rf"^{re.escape(PROMPT)}", text, re.M))


def wait(root: Path, after: int = 0, timeout: float = 3600.0, lines: int = 1) -> int:
    """Block until out.log grew past `after` and ends at the prompt (0), the session
    ended (2), or `timeout` seconds passed (1). Prints the last lines.

    `lines`: the command lines just sent. The REPL prints a prompt after each, so the
    first one's prompt must not end the wait (a synthesis after "answer" + "/close").
    """
    log = root / "out.log"
    deadline = time.monotonic() + timeout
    idle_since: float | None = None
    while time.monotonic() < deadline:
        data = log.read_bytes() if log.exists() else b""
        # `after` counts bytes: slice before decoding, or "★" and "—" shift the start.
        text, new = (d.decode("utf-8", errors="replace") for d in (data, data[after:]))
        grown = len(data) > after
        if grown and any(marker in new for marker in ENDED):
            print("\n".join(text.rstrip().splitlines()[-3:]))
            return 2
        if grown and text.endswith(PROMPT) and _prompts(new) >= lines:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= IDLE_S:
                print("\n".join(text.rstrip().splitlines()[-3:]))
                return 0
        else:
            idle_since = None
        time.sleep(POLL_S)
    return 1


def _feed(proc: subprocess.Popen[str], cmds: Path) -> None:
    """Pass every line appended to `cmds` to the process's stdin until it exits."""
    assert proc.stdin is not None
    offset = 0
    while proc.poll() is None:
        text = cmds.read_text(encoding="utf-8")
        complete = text[offset:].rpartition("\n")[0]
        if complete:
            offset += len(complete) + 1
            try:
                proc.stdin.write(complete + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                break
        time.sleep(POLL_S)


def start(root: Path, council_args: list[str]) -> int:
    """Run `council --root ROOT <args> --no-tui`, fed from ROOT/cmds; return its code."""
    cmds = root / "cmds"
    cmds.write_text("", encoding="utf-8")
    command = ["uv", "run", "--project", str(REPO), "council", "--root", str(root)]
    command += [*council_args, "--no-tui"]
    with (root / "out.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        _feed(proc, cmds)
        return proc.wait()


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_setup = sub.add_parser("setup", help="create a scratch root")
    p_setup.add_argument("root", type=Path)
    p_setup.add_argument("--council", required=True)
    p_setup.add_argument("--max-rounds", type=int)
    p_setup.add_argument("--memory", default="off", choices=["on", "read_only", "off"])
    p_start = sub.add_parser("start", help="run a session fed from ROOT/cmds")
    p_start.add_argument("root", type=Path)
    p_start.add_argument("council_args", nargs="+", help="after --: new ... or resume ...")
    p_send = sub.add_parser("send", help="append lines for the running session")
    p_send.add_argument("root", type=Path)
    p_send.add_argument("lines", nargs="+")
    p_send.add_argument("--wait", action="store_true", help="then wait for the prompt")
    p_send.add_argument("--timeout", type=float, default=3600.0)
    p_wait = sub.add_parser("wait", help="wait until the session is back at its prompt")
    p_wait.add_argument("root", type=Path)
    p_wait.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.root, args.council, args.max_rounds, args.memory)
    elif args.cmd == "start":
        sys.exit(start(args.root, args.council_args))
    elif args.cmd == "wait":
        sys.exit(wait(args.root, timeout=args.timeout))
    else:
        size = _log_size(args.root)  # before sending: the reply is what comes after
        send(args.root, args.lines)
        if args.wait:
            sys.exit(wait(args.root, size, args.timeout, len(args.lines)))


if __name__ == "__main__":
    main()
