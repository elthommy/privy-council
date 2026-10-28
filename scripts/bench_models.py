"""Benchmark local Ollama models on realistic persona turns (no Claude calls).

Builds the same blind-round prompt the engine sends (persona + engine rules +
optional web research from a saved session), runs it cold then warm on each
model, and prints Ollama's own timings plus the answers side by side.

    uv run python scripts/bench_models.py qwen3.6:latest gemma4:26b=false gpt-oss:20b=low \\
        --personas rigor,contrarian --session data/sessions/<id>

A `=value` suffix sets Ollama's `think` option: true/false, or low/medium/high
for models that only support levels (gpt-oss). Models must already be pulled.

Context size (`num_ctx`): `--num-ctx 8192,12288,16384`
runs every model at each size. `--round 2` builds a round-2 prompt from the
session's last messages and debate state instead of the blind one; `--dossier`
adds attached files. Each size reloads the model, so its first run is cold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from privy_council.config import DossierConfig
from privy_council.engine import context, prompts
from privy_council.engine.dossier import read_attachable
from privy_council.engine.resume import messages_of
from privy_council.loaders.personas import load_personas
from privy_council.models.persona import Persona
from privy_council.models.session import DossierFile, SessionMeta
from privy_council.storage.sessions import SessionStore
from privy_council.util.tokens import estimate_tokens

OLLAMA = "http://localhost:11434/api/chat"
OLLAMA_PS = "http://localhost:11434/api/ps"
NUM_CTX = 8192
MAX_OUTPUT = 1500
LAST_K = 6  # [engine] context_last_k_messages
DEFAULT_TOPIC = "Is it worth holding 20% of a long-term portfolio in gold?"


def think_value(raw: str | None) -> bool | str | None:
    """Parse the `=value` suffix into Ollama's think option."""
    if raw is None:
        return None
    return {"true": True, "false": False}.get(raw.lower(), raw.lower())


def dossier_files(paths: list[Path]) -> str | None:
    """Render files for the prompt, verbatim (as the engine does below the summary threshold)."""
    limit = DossierConfig().max_file_bytes
    files = [
        DossierFile(
            name=p.name, text=(t := read_attachable(p, limit)).strip(), tokens=estimate_tokens(t)
        )
        for p in paths
    ]
    return context.dossier_text(files)


def build_prompt(
    persona: Persona,
    moderator: Persona,
    session: Path | None,
    *,
    num_ctx: int = NUM_CTX,
    round_no: int = 1,
    dossier: str | None = None,
) -> context.Prompt:
    """Build a member prompt for `persona`: blind (round 1) or later from a session."""
    topic, facts, state, history = DEFAULT_TOPIC, None, None, []
    if session is not None:
        meta = SessionMeta.model_validate_json((session / "session.json").read_text())
        topic, facts = meta.topic, context.facts_text(meta.runtime.research)
        if round_no > 1:
            nicks = {p.slug: p.name for p in (persona, moderator)}
            nick = lambda slug: nicks.get(slug, slug.capitalize())  # noqa: E731
            messages = messages_of(SessionStore.open(session).events())
            history = context.render_history(context.recent_messages(messages, LAST_K), nick)
            state = context.state_text(meta.runtime.debate_state, nick)
    task = prompts.BLIND_TASK if round_no == 1 else prompts.OPEN_ROUND_TASK.format(round=round_no)
    return context.build_prompt(
        system=f"{persona.body}\n\n{prompts.member_protocol(persona, moderator)}",
        topic=topic,
        task=task,
        budget_tokens=num_ctx - MAX_OUTPUT,
        facts=facts,
        dossier=dossier,
        dossier_share=DossierConfig().context_share,
        state_text=state,
        history=history,
    )


def run_once(
    model: str, think: bool | str | None, prompt: context.Prompt, num_ctx: int = NUM_CTX
) -> dict[str, Any]:
    """Run one non-streaming chat call and return Ollama's response plus wall time."""
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "5m",
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "options": {"temperature": 0.7, "num_ctx": num_ctx, "num_predict": MAX_OUTPUT},
    }
    if think is not None:
        body["think"] = think
    start = time.monotonic()
    response: dict[str, Any] = httpx.post(OLLAMA, json=body, timeout=900).json()
    response["wall_s"] = time.monotonic() - start
    return response


def placement(model: str) -> str:
    """Return the loaded model's resident size and the share of it in VRAM (`/api/ps`)."""
    for loaded in httpx.get(OLLAMA_PS, timeout=10).json().get("models", []):
        if loaded["name"] == model:
            size, vram = loaded["size"], loaded.get("size_vram", 0)
            gpu = 100 * vram / size if size else 0
            return (
                f"{size / 2**30:.1f} GiB resident, {vram / 2**30:.1f} GiB in VRAM ({gpu:.0f}% GPU)"
            )
    return "?"


def describe(model: str, run: str, r: dict[str, Any]) -> str:
    """Format one timing line."""
    ns = 1e9
    gen = r["eval_count"] / (r["eval_duration"] / ns)
    thinking = len(r["message"].get("thinking") or "")
    return (
        f"{model:24} {run:4}: {r['wall_s']:5.1f}s total | load {r['load_duration'] / ns:5.1f}s | "
        f"prefill {r.get('prompt_eval_count', 0):5} tok in "
        f"{r.get('prompt_eval_duration', 0) / ns:4.1f}s | "
        f"{r['eval_count']:4} tok @ {gen:5.1f} tok/s | thinking {thinking} chars"
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="model[=think], e.g. gpt-oss:20b=low")
    parser.add_argument("--personas", default="rigor", help="comma-separated persona slugs")
    parser.add_argument("--session", type=Path, help="session dir whose research to include")
    parser.add_argument("--num-ctx", default=str(NUM_CTX), help="comma-separated sizes")
    parser.add_argument("--round", type=int, default=1, help="2+: a later round (needs --session)")
    parser.add_argument("--dossier", type=Path, action="append", default=[], help="file to attach")
    parser.add_argument("--out", type=Path, help="write answers as JSON here")
    return parser.parse_args()


def main() -> None:
    """Benchmark each model at each context size on each persona, cold then warm."""
    args = parse_args()
    personas, _ = load_personas(Path("personas"))
    dossier = dossier_files(args.dossier)
    answers: dict[str, dict[str, str]] = {}
    for num_ctx in (int(n) for n in args.num_ctx.split(",")):
        for spec in args.models:
            model, _, raw = spec.partition("=")
            think = think_value(raw or None)
            for i, slug in enumerate(args.personas.split(",")):
                prompt = build_prompt(
                    personas[slug],
                    personas["chair"],
                    args.session,
                    num_ctx=num_ctx,
                    round_no=args.round,
                    dossier=dossier,
                )
                trimmed = (
                    f"{prompt.dropped_messages} msg dropped" if prompt.dropped_messages else ""
                )
                runs = ["cold", "warm"] if i == 0 else ["warm"]
                for run in runs:
                    result = run_once(model, think, prompt, num_ctx)
                    label = f"{num_ctx // 1024}k/{slug[:4]}/{run}"
                    print(describe(model, label, result), trimmed, flush=True)
                answers.setdefault(f"{spec}@{num_ctx}", {})[slug] = result["message"]["content"]
            print(f"{'':24}       {num_ctx // 1024}k placement: {placement(model)}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(answers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
