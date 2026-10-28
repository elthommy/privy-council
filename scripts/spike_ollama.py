"""M0 spike: get replies from Ollama's /api/chat and measure model swap time.

Live script (local GPU, no Claude quota). Stdlib only, so it runs before the
project is scaffolded:

    python3 scripts/spike_ollama.py                          # qwen3:8b <-> gemma3:12b-it-qat
    python3 scripts/spike_ollama.py --models qwen3:8b qwen3:4b

For each model in the sequence A, A, B, A it prints load/prompt/eval timings
taken from Ollama's own response fields. Findings go to docs/spike-notes.md.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are Rigor, a terse former credit risk analyst on an advisory council. "
    "Answer in at most two short sentences."
)
USER_PROMPT = "Should someone keep six months of expenses in cash? Give a position."
NS_PER_S = 1e9


def http_json(url: str, payload: dict[str, object] | None, timeout: float) -> dict[str, object]:
    """GET (payload None) or POST JSON to url and return the decoded JSON body."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: dict[str, object] = json.loads(resp.read())
    return body


def check_health(base_url: str, models: list[str]) -> bool:
    """Print server version and whether each model is pulled; return True if all is well."""
    try:
        version = http_json(f"{base_url}/api/version", None, 5)["version"]
        tags = http_json(f"{base_url}/api/tags", None, 5)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Ollama unreachable at {base_url}: {exc}")
        print("Start it with: sudo systemctl start ollama")
        return False
    pulled = {m["name"] for m in tags["models"]}  # type: ignore[union-attr,index]
    print(f"ollama server {version}; pulled: {', '.join(sorted(pulled))}")
    missing = [m for m in models if m not in pulled]
    for m in missing:
        print(f"missing model {m}: run `ollama pull {m}`")
    return not missing


def chat(base_url: str, model: str, num_ctx: int, timeout: float) -> dict[str, object]:
    """Send one non-streaming /api/chat request and return the response JSON."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "stream": False,
        "think": False,  # qwen3 otherwise spends hundreds of tokens thinking
        "keep_alive": "10m",
        "options": {"temperature": 0.6, "num_ctx": num_ctx, "num_predict": 200},
    }
    return http_json(f"{base_url}/api/chat", payload, timeout)


def report(label: str, model: str, resp: dict[str, object], wall: float) -> None:
    """Print one timing line plus the reply text."""

    def sec(key: str) -> float:
        """Convert an Ollama nanosecond duration field to seconds."""
        return float(resp.get(key, 0)) / NS_PER_S  # type: ignore[arg-type]

    eval_s = sec("eval_duration") or 1e-9
    tok_s = float(resp.get("eval_count", 0)) / eval_s  # type: ignore[arg-type]
    print(
        f"[{label}] {model}: wall={wall:.1f}s load={sec('load_duration'):.1f}s "
        f"prompt={resp.get('prompt_eval_count')}tok/{sec('prompt_eval_duration'):.2f}s "
        f"eval={resp.get('eval_count')}tok @ {tok_s:.0f} tok/s"
    )
    message = resp.get("message", {})
    print(f"    > {message.get('content', '').strip()[:300]}")  # type: ignore[union-attr]


def print_loaded(base_url: str) -> None:
    """Print the models currently resident and their VRAM footprint."""
    ps = http_json(f"{base_url}/api/ps", None, 5)
    for m in ps.get("models", []):  # type: ignore[union-attr]
        size, vram = m.get("size", 0) / 1e9, m.get("size_vram", 0) / 1e9
        print(f"    resident: {m['name']} size={size:.1f} GB vram={vram:.1f} GB")


def main() -> None:
    """Run the A, A, B, A sequence and report timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--models", nargs=2, default=["qwen3:8b", "gemma3:12b-it-qat"])
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    a, b = args.models
    if not check_health(args.base_url, [a, b]):
        raise SystemExit(1)
    for label, model in [("cold", a), ("warm", a), ("swap", b), ("swap back", a)]:
        start = time.monotonic()
        resp = chat(args.base_url, model, args.num_ctx, args.timeout)
        report(label, model, resp, time.monotonic() - start)
        print_loaded(args.base_url)


if __name__ == "__main__":
    main()
