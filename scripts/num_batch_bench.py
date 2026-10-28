"""Time prefill at several Ollama `num_batch` sizes on one Chair prompt (no Claude calls).

Spike 6 hypothesis (docs/spike-notes.md): when most of a model sits in RAM,
llama.cpp copies those layers to the GPU once per batch of `num_batch` prompt
tokens, so a bigger batch means fewer copies over PCIe and a faster prefill.

The prompt is a real Chair round update, rebuilt from a saved session as
`chair_bench.py` does, and sent straight to /api/chat with the engine's options
plus `num_batch`. Each size starts from an unloaded model (the load is timed
apart), and a fresh nonce at the top of the system prompt defeats Ollama's
prompt cache, so every run pays the full prefill. Output is capped
(`--num-predict`): only the speeds matter here, not the decision.

    uv run python -u scripts/num_batch_bench.py --model nemotron-3.5-super:latest \\
        --num-ctx 24576 --num-batch 512 2048 --repeat 2
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from chair_bench import bench_settings, build_engine, load_manifest, round_state

from privy_council.config import load_settings
from privy_council.engine import dynamics

OLLAMA = "http://localhost:11434"
TIMEOUT_S = 1800.0
NS = 1e9


def chair_messages(
    manifest: Path, index: int, model: str, num_ctx: int
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Return (item id, chat messages, JSON schema) of the manifest's round item `index`."""
    item = [i for i in load_manifest(manifest) if i.kind != "synthesis"][index]
    settings = bench_settings(load_settings(None), model, num_ctx)
    meta, events, final = round_state(item)
    engine = build_engine(settings, model, meta, events, final.runtime.round)
    stats = dynamics.member_stats(engine.messages, list(engine.members), item.round)
    prompt = engine._moderate_request(stats, item.round >= engine.council.protocol.max_rounds)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    return item.id, messages, engine._moderator_schema()


def unload(client: httpx.Client, model: str) -> None:
    """Unload the model, so the next call loads it with the new options."""
    client.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0})
    while any(m["name"] == model for m in client.get(f"{OLLAMA}/api/ps").json()["models"]):
        time.sleep(1)


def load(client: httpx.Client, model: str, options: dict[str, Any]) -> dict[str, Any]:
    """Load the model with `options` (empty prompt); return load time and memory split."""
    start = time.monotonic()
    client.post(
        f"{OLLAMA}/api/generate", json={"model": model, "options": options}
    ).raise_for_status()
    seconds = time.monotonic() - start
    ps = next(m for m in client.get(f"{OLLAMA}/api/ps").json()["models"] if m["name"] == model)
    return {
        "load_s": round(seconds, 1),
        "size_gib": ps["size"] / 2**30,
        "vram_gib": ps["size_vram"] / 2**30,
    }


def timed_call(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    options: dict[str, Any],
    think: bool | str | None,
) -> dict[str, Any]:
    """Send the Chair prompt with a cache-defeating nonce; return Ollama's timings."""
    nonce = {"role": "system", "content": f"Run {uuid.uuid4()}\n\n{messages[0]['content']}"}
    payload = {
        "model": model,
        "messages": [nonce, *messages[1:]],
        "stream": False,
        "think": think,
        "format": schema,
        "options": options,
    }
    start = time.monotonic()
    r = client.post(f"{OLLAMA}/api/chat", json=payload).raise_for_status().json()
    return {
        "wall_s": round(time.monotonic() - start, 1),
        "prompt_tokens": r.get("prompt_eval_count", 0),
        "prefill_s": round(r.get("prompt_eval_duration", 0) / NS, 1),
        "out_tokens": r["eval_count"],
        "gen_s": round(r["eval_duration"] / NS, 1),
    }


def show(row: dict[str, Any]) -> str:
    """Return one readable line for a run."""
    prefill = row["prompt_tokens"] / row["prefill_s"] if row["prefill_s"] else 0
    gen = row["out_tokens"] / row["gen_s"] if row["gen_s"] else 0
    return (
        f"num_batch {row['num_batch']:5} run {row['run']}: prefill {row['prompt_tokens']} tok in "
        f"{row['prefill_s']} s ({prefill:.0f} tok/s) | gen {gen:.1f} tok/s | "
        f"wall {row['wall_s']} s | "
        f"load {row['load_s']} s, VRAM {row['vram_gib']:.1f} of {row['size_gib']:.1f} GiB"
    )


def main() -> None:
    """Run each num_batch size `--repeat` times and append the timings to --out."""
    parser = argparse.ArgumentParser(description="Time prefill at several num_batch sizes.")
    parser.add_argument("--model", default="nemotron-3.5-super:latest")
    parser.add_argument("--manifest", type=Path, default=Path("data/bench/chair_manifest.toml"))
    parser.add_argument("--item", type=int, default=0, help="Index among round/plant items")
    parser.add_argument("--num-ctx", type=int, default=24576)
    parser.add_argument("--num-batch", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/bench/runs/num-batch.jsonl"))
    args = parser.parse_args()
    item_id, messages, schema = chair_messages(args.manifest, args.item, args.model, args.num_ctx)
    think = load_settings(None).ollama.models[args.model].think  # as the engine sends it
    print(f"== {args.model} @ num_ctx {args.num_ctx}, {item_id}", flush=True)
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for num_batch in args.num_batch:
            options = {
                "num_ctx": args.num_ctx,
                "num_batch": num_batch,
                "temperature": 0.2,
                "num_predict": args.num_predict,
            }
            unload(client, args.model)
            loaded = load(client, args.model, options)
            for run in range(1, args.repeat + 1):
                row = {
                    "model": args.model,
                    "item": item_id,
                    "num_batch": num_batch,
                    "run": run,
                    **loaded,
                }
                row.update(timed_call(client, args.model, messages, schema, options, think))
                print(show(row), flush=True)
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        unload(client, args.model)


if __name__ == "__main__":
    main()
