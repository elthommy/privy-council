"""Check that local models load on the first try, on real turn prompts (no Claude calls).

Each run loads `--primer` first (another seat's model), so the candidate has to take
its place as it would in a debate, then sends it one saved turn prompt with the
engine's options. Ollama hides a failed load behind its own retry, so the starts of
llama-server and the out-of-memory lines are read from its journal. The lowest free
GPU memory is sampled while the model loads and answers.

A run passes with one start, no out-of-memory line and a non-empty answer that ended
by itself. Add `+nodraft` to a model to send `draft_num_predict = 0` (no draft model).
The prompts come from a bench run file and the answers go to `--out`: both hold real
sessions, so keep them in data/bench/ (gitignored). Runs already in `--out` are skipped.

    uv run python -u scripts/load_gate.py --prompts-from data/bench/runs/m18-vouch.jsonl \\
        --out data/bench/runs/load-gate.jsonl --runs 5 \\
        --models granite4.2:30b nemotron-3.5-bolt:latest+nodraft
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from privy_council.config import OllamaModelConfig, Settings, load_settings

OLLAMA = "http://localhost:11434"
TIMEOUT_S = 900.0
NS = 1e9
NO_DRAFT = "+nodraft"
START = 'msg="starting llama-server"'
OUT_OF_MEMORY = re.compile(r"cudaMalloc failed|CUDA error: out of memory")


@dataclass
class Run:
    """One load and one answer of a candidate."""

    version: str
    model: str
    index: int
    starts: int
    oom_lines: int
    min_free_mib: int
    load_s: float
    total_s: float
    tok_s: float
    tokens: int
    done_reason: str
    error: str
    spec: str
    text: str

    @property
    def passed(self) -> bool:
        """Say whether the model loaded at once and gave a whole answer."""
        loaded = self.starts == 1 and self.oom_lines == 0 and not self.error
        return loaded and self.done_reason == "stop" and bool(self.text.strip())


def journal_counts(log: str) -> tuple[int, int]:
    """Return the llama-server starts and the out-of-memory lines in an Ollama log."""
    lines = log.splitlines()
    return sum(START in ln for ln in lines), sum(bool(OUT_OF_MEMORY.search(ln)) for ln in lines)


def read_log(since: datetime, log: Path | None, offset: int) -> str:
    """Return Ollama's log since a time (journal) or an offset (log file)."""
    if log is not None:
        return log.read_bytes()[offset:].decode("utf-8", errors="replace")
    stamp = since.strftime("%Y-%m-%d %H:%M:%S")
    cmd = ["journalctl", "-u", "ollama", "--since", stamp, "--no-pager", "-o", "cat"]
    # A model's token pieces can put invalid UTF-8 in the log.
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return raw.decode("utf-8", errors="replace")


def free_mib() -> int:
    """Return the GPU's free memory in MiB."""
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    return int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()[0])


class FreeMemory:
    """Sample the lowest free GPU memory in a background thread, once a model is loaded."""

    def __init__(self, base_url: str, model: str) -> None:
        """Start with no sample."""
        self.lowest = -1
        self._ps = f"{base_url}/api/ps"
        self._model = model
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        """Keep the lowest reading until stopped.

        Only while the candidate is the one listed: the primer still holds the GPU at first.
        """
        while not self._stop.wait(0.5):
            names = [m["name"] for m in httpx.get(self._ps, timeout=10).json()["models"]]
            if names == [self._model]:
                free = free_mib()
                self.lowest = free if self.lowest < 0 else min(self.lowest, free)

    def __enter__(self) -> FreeMemory:
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop sampling."""
        self._stop.set()
        self._thread.join()


def payload(
    name: str, config: OllamaModelConfig, turn: dict[str, Any], think: bool | str | None = None
) -> dict[str, Any]:
    """Build the /api/chat body for a candidate, with the engine's options."""
    options: dict[str, Any] = {
        "temperature": turn["temperature"],
        "num_ctx": 12288,
        "num_predict": turn["max_output"],
    }
    if config.num_batch is not None:
        options["num_batch"] = config.num_batch
    if name.endswith(NO_DRAFT):
        options["draft_num_predict"] = 0
    body = {
        "model": name.removesuffix(NO_DRAFT),
        "stream": False,
        "keep_alive": "5m",
        "options": options,
        "messages": [
            {"role": "system", "content": turn["system"]},
            {"role": "user", "content": turn["user"]},
        ],
    }
    if think is not None:
        body["think"] = think
    return body


def can_think(base_url: str, model: str) -> bool:
    """Say whether Ollama lets a model think (sending `think` to others is an error)."""
    shown = httpx.post(f"{base_url}/api/show", json={"model": model}, timeout=30).json()
    return "thinking" in shown.get("capabilities", [])


def think_option(client: httpx.Client, name: str, config: OllamaModelConfig) -> bool | str | None:
    """Return the config's `think`, else off for a model that can think."""
    if config.think is not None:
        return config.think
    return False if can_think(str(client.base_url).rstrip("/"), name) else None


def unload_all(client: httpx.Client) -> None:
    """Unload every model and wait until none is left."""
    while loaded := client.get("/api/ps").json()["models"]:
        for m in loaded:
            client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
        time.sleep(1)


def prime(client: httpx.Client, settings: Settings, primer: str) -> None:
    """Load another seat's model with the council's options."""
    unload_all(client)
    turn = {"temperature": 0.7, "max_output": 5, "system": "Be brief.", "user": "Say OK."}
    config = settings.ollama.models.get(primer, OllamaModelConfig())
    body = payload(primer, config, turn, think_option(client, primer, config))
    client.post("/api/chat", json=body).raise_for_status()


def run_once(
    client: httpx.Client,
    settings: Settings,
    name: str,
    primer: str,
    index: int,
    turn: dict[str, Any],
    log: Path | None,
) -> Run:
    """Load a candidate over the primer and answer one turn."""
    prime(client, settings, primer)
    config = settings.ollama.models.get(name.removesuffix(NO_DRAFT), OllamaModelConfig())
    model = name.removesuffix(NO_DRAFT)
    body = payload(name, config, turn, think_option(client, model, config))
    # The journal is read at 1 s resolution: keep the primer's lines out.
    time.sleep(2)
    since, start = datetime.now(), time.monotonic()
    offset = log.stat().st_size if log is not None else 0
    with FreeMemory(str(client.base_url).rstrip("/"), model) as memory:
        response = client.post("/api/chat", json=body)
    total = time.monotonic() - start
    time.sleep(1)
    starts, oom = journal_counts(read_log(since, log, offset))
    data = response.json()
    tokens = data.get("eval_count", 0)
    return Run(
        version=client.get("/api/version").json()["version"],
        model=name,
        index=index,
        starts=starts,
        oom_lines=oom,
        min_free_mib=memory.lowest,
        load_s=round(data.get("load_duration", 0) / NS, 1),
        total_s=round(total, 1),
        tok_s=round(tokens / (data.get("eval_duration", 0) / NS or 1), 1),
        tokens=tokens,
        done_reason=data.get("done_reason", ""),
        error=data.get("error", ""),
        spec=turn["spec"],
        text=data.get("message", {}).get("content", ""),
    )


def summary(runs: list[Run]) -> str:
    """Return one line per model: passes, starts, headroom and times."""
    lines = []
    for key in dict.fromkeys((r.version, r.model) for r in runs):
        rows = [r for r in runs if (r.version, r.model) == key]
        lines.append(
            f"{key[0]} {key[1]:<36} pass {sum(r.passed for r in rows)}/{len(rows)}"
            f" starts {[r.starts for r in rows]} oom {sum(r.oom_lines for r in rows)}"
            f" min free {min(r.min_free_mib for r in rows)} MiB"
            f" load {statistics.median(r.load_s for r in rows):.0f} s"
            f" turn {statistics.median(r.total_s for r in rows):.0f} s"
            f" {statistics.median(r.tok_s for r in rows):.1f} tok/s"
        )
    return "\n".join(lines)


def load_runs(path: Path) -> list[Run]:
    """Return the runs already saved."""
    if not path.exists():
        return []
    return [Run(**json.loads(line)) for line in path.read_text().splitlines()]


def parse_args() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prompts-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--primer", default="gemma4:26b")
    parser.add_argument("--url", default=OLLAMA)
    parser.add_argument("--log", type=Path, help="server log file (default: systemd journal)")
    return parser.parse_args()


def main() -> None:
    """Run the gate and print the summary."""
    args = parse_args()
    saved = [json.loads(line) for line in args.prompts_from.read_text().splitlines()]
    turns = [t for t in saved if t["type"] == "turn"]
    settings = load_settings(None)
    runs = load_runs(args.out)
    with httpx.Client(base_url=args.url, timeout=TIMEOUT_S) as client:
        version = client.get("/api/version").json()["version"]
        done = {(r.version, r.model, r.index) for r in runs}
        for name in args.models:
            for i in range(args.runs):
                if (version, name, i) in done:
                    continue
                turn = turns[i % len(turns)]
                run = run_once(client, settings, name, args.primer, i, turn, args.log)
                runs.append(run)
                with args.out.open("a") as out:
                    out.write(json.dumps(asdict(run)) + "\n")
                print(
                    f"{name} {i}: {'ok' if run.passed else 'FAIL'} starts {run.starts}"
                    f" free {run.min_free_mib} load {run.load_s} total {run.total_s}"
                    f" {run.tok_s} tok/s {run.done_reason} {run.error[:150]}",
                    flush=True,
                )
        unload_all(client)
    print(summary(runs))


if __name__ == "__main__":
    main()
