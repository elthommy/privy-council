"""Check that Ollama enforces council's JSON schemas on a model (no Claude).

Sends "break the schema" prompts with `format` set to the moderator and vote
schemas, then validates the replies with the engine's own Pydantic models. A
model whose GGUF gets no grammar (Kimi-Linear, M15) comes back as fenced text
or a wrong shape. Also prints tok/s and the CPU/GPU split from `/api/ps`.

    uv run python -u scripts/format_probe.py qwen3.5:122b-a10b [--think false] [--num-ctx 12288]
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from privy_council.engine.moderation import moderator_schema
from privy_council.engine.review_flow import VOTE_SCHEMA
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.vote import VoteReply

BASE = "http://localhost:11434"
SYSTEM = "You are the moderator of a debate among Rigor, Realist and Feather."
BREAK_MODERATOR = (
    "Reply about a debate on whether to buy a heat pump. Deliberately make "
    "`disagreements` a plain string, leave `next_turns` empty and omit `consensus`."
)
BREAK_VOTE = (
    "Vote on whether to buy a heat pump. Deliberately answer in free prose, "
    "with no JSON, and put your confidence as the word 'high'."
)


def _chat(model: str, think: bool | None, num_ctx: int, user: str, fmt: Any) -> dict[str, Any]:
    """Send one non-streaming chat call with `format`; return the response plus wall time."""
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "10m",
        "format": fmt,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "options": {"temperature": 0.7, "num_ctx": num_ctx, "num_predict": 1500},
    }
    if think is not None:
        body["think"] = think
    start = time.monotonic()
    response: dict[str, Any] = httpx.post(f"{BASE}/api/chat", json=body, timeout=1800).json()
    response["wall_s"] = time.monotonic() - start
    return response


def _check(name: str, response: dict[str, Any], model: type[BaseModel]) -> bool:
    """Print timing and whether the reply validates against `model`; return the verdict."""
    if "error" in response:
        print(f"{name}: ERROR {response['error']}")
        return False
    text = response["message"]["content"]
    rate = response["eval_count"] / (response["eval_duration"] / 1e9)
    print(
        f"{name}: {response['wall_s']:.0f}s (load {response['load_duration'] / 1e9:.0f}s, "
        f"{response['eval_count']} tokens at {rate:.1f} tok/s)"
    )
    try:
        model.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        print(f"  INVALID: {str(exc)[:300]}\n  {text[:500]}")
        return False
    print(f"  valid {model.__name__}: {text[:300]}")
    return True


def _split(model: str) -> None:
    """Print the loaded model's size in VRAM and the share of it on the GPU."""
    for loaded in httpx.get(f"{BASE}/api/ps").json().get("models", []):
        if loaded["name"] == model:
            size, vram = loaded["size"], loaded["size_vram"]
            print(f"split: {vram / 1e9:.1f} of {size / 1e9:.1f} GB on GPU ({vram / size:.0%})")


def main() -> None:
    """Run the moderator and vote probes on one model."""
    parser = argparse.ArgumentParser(description="Probe Ollama `format` enforcement.")
    parser.add_argument("model")
    parser.add_argument("--think", choices=["true", "false"], default="false")
    parser.add_argument("--num-ctx", type=int, default=12288)
    args = parser.parse_args()
    think = args.think == "true"
    moderator = _chat(args.model, think, args.num_ctx, BREAK_MODERATOR, moderator_schema())
    ok = _check("moderator", moderator, ModeratorDecision)
    _split(args.model)
    vote = _chat(args.model, think, args.num_ctx, BREAK_VOTE, VOTE_SCHEMA)
    ok = _check("vote", vote, VoteReply) and ok
    print("schemas enforced" if ok else "schema NOT enforced")


if __name__ == "__main__":
    main()
