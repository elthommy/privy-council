"""Ollama backend over its HTTP API (`POST /api/chat`): streamed for chat turns, plain for JSON."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from typing import Any

import httpx

from privy_council.backends.base import (
    BackendError,
    BackendHealth,
    BackendOutputError,
    BackendTimeoutError,
    BackendUnavailableError,
    CompletionRequest,
    CompletionResult,
    ModelLoadError,
    TextSink,
    Usage,
)
from privy_council.config import OllamaConfig, OllamaModelConfig

START_HINT = "start it with `sudo systemctl start ollama`"
HEALTH_TIMEOUT_S = 5.0


def pull_hint(model: str) -> str:
    """Return the command that installs a missing model."""
    return f"run `ollama pull {model}`"


# Ollama's 500 bodies worth one retry: a model that didn't finish loading within its
# load timeout, and a CUDA OOM (Ollama expires loaded models before the next request;
# seen with gemma4, whose MTP draft model llama.cpp leaves out of its VRAM fit).
LOAD_TIMEOUT = "timed out waiting for llama-server to start"
OUT_OF_MEMORY = "out of memory"
THINK_END = "</think>"


def strip_reasoning(text: str) -> str:
    """Return the answer after a reasoning block the model wrote into its reply.

    With `think = false`, nemotron-3.5-super now and then still reasons in plain
    text, closes with </think> and then answers: a draft, the tag, the final text
    (seen 3 times: Rigor's turns and a synthesis posted twice). An empty answer
    after the tag keeps the text before it, without the tags.
    """
    if THINK_END not in text:
        return text
    answer = text.rsplit(THINK_END, 1)[1].strip()
    return answer or text.replace(THINK_END, "").replace("<think>", "").strip()


class OllamaBackend:
    """Calls a local Ollama server through one shared AsyncClient.

    All calls go through one semaphore (concurrency 1 by default): with 12 GB
    VRAM only one model fits at a time, and parallel calls would force swaps.
    Create a single instance per app (the registry does).
    """

    name = "ollama"

    def __init__(self, config: OllamaConfig, client: httpx.AsyncClient | None = None) -> None:
        """Use `client` if given (tests), else create one for `config.base_url`."""
        self.config = config
        self.client = client or httpx.AsyncClient(base_url=config.base_url)
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def model_config(self, model: str) -> OllamaModelConfig:
        """Return per-model options from config, or defaults."""
        return self.config.models.get(model, OllamaModelConfig())

    def build_payload(self, req: CompletionRequest) -> dict[str, Any]:
        """Build the /api/chat request body."""
        opts = self.model_config(req.model)
        messages = [{"role": "system", "content": req.system}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": req.temperature,
                "num_ctx": opts.num_ctx,
                "num_predict": req.max_output_tokens,
            },
        }
        if opts.num_batch is not None:
            payload["options"]["num_batch"] = opts.num_batch
        if opts.think is not None:
            payload["think"] = opts.think
        if req.json_schema is not None:
            payload["format"] = req.json_schema
        return payload

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Run one chat completion under the global semaphore (streamed with `on_text`)."""
        payload = self.build_payload(req)
        timeout = req.timeout_s or self.config.timeout_s
        if on_text is not None and req.json_schema is None:
            payload["stream"] = True
        async with self._semaphore:
            start = time.monotonic()
            try:
                data = await self._chat(payload, req.model, timeout, on_text)
            except ModelLoadError:
                # A big model swap can outlast Ollama's load wait (the next try finds it
                # loading or loaded), and an OOM frees VRAM for the next request.
                # Retried here so it never costs a caller's own retry.
                data = await self._chat(payload, req.model, timeout, on_text)
            latency = time.monotonic() - start
        return self._to_result(data, req, latency)

    async def _chat(
        self, payload: dict[str, Any], model: str, timeout: float, on_text: TextSink | None
    ) -> dict[str, Any]:
        """Run one /api/chat request, streamed when the payload asks for it."""
        if payload["stream"] and on_text is not None:
            return await self._stream_chat(payload, model, timeout, on_text)
        return await self._post_chat(payload, model, timeout)

    async def _stream_chat(
        self, payload: dict[str, Any], model: str, timeout: float, on_text: TextSink
    ) -> dict[str, Any]:
        """Stream /api/chat; return the final chunk, with the whole reply as its message."""
        parts: list[str] = []
        final: dict[str, Any] | None = None
        try:
            # httpx's timeout is per read; the whole call gets the same budget as non-streamed.
            async with (
                asyncio.timeout(timeout),
                self.client.stream("POST", "/api/chat", json=payload, timeout=timeout) as resp,
            ):
                if resp.status_code >= 400:
                    await resp.aread()
                    self._check_status(resp, model)
                async for line in resp.aiter_lines():
                    chunk = _stream_chunk(line)
                    if chunk is None:
                        continue
                    piece = str((chunk.get("message") or {}).get("content") or "")
                    if piece:
                        parts.append(piece)
                        on_text(piece)
                    if chunk.get("done"):
                        final = chunk
                        break
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise BackendTimeoutError(f"Ollama ({model}) timed out after {timeout:.0f}s") from exc
        except httpx.TransportError as exc:
            raise BackendUnavailableError(
                f"Ollama unreachable at {self.config.base_url}; {START_HINT}"
            ) from exc
        if final is None:
            raise BackendOutputError("Ollama stream ended before its final chunk")
        return {**final, "message": {"role": "assistant", "content": "".join(parts)}}

    def _check_status(self, resp: httpx.Response, model: str) -> None:
        """Map an HTTP error status to a typed error."""
        if resp.status_code == 404:
            raise BackendUnavailableError(f"Ollama model {model!r} not found; {pull_hint(model)}")
        if resp.status_code >= 400:
            retryable = LOAD_TIMEOUT in resp.text or OUT_OF_MEMORY in resp.text
            error = ModelLoadError if retryable else BackendError
            raise error(f"Ollama error {resp.status_code}: {resp.text[:300]}")

    async def _post_chat(
        self, payload: dict[str, Any], model: str, timeout: float
    ) -> dict[str, Any]:
        """POST /api/chat and return the JSON body, mapping failures to typed errors."""
        try:
            resp = await self.client.post("/api/chat", json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"Ollama ({model}) timed out after {timeout:.0f}s") from exc
        except httpx.TransportError as exc:
            raise BackendUnavailableError(
                f"Ollama unreachable at {self.config.base_url}; {START_HINT}"
            ) from exc
        self._check_status(resp, model)
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise BackendOutputError("Ollama returned non-JSON output") from exc
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise BackendOutputError("Ollama response has no 'message'")
        return data

    def _to_result(
        self, data: dict[str, Any], req: CompletionRequest, latency: float
    ) -> CompletionResult:
        """Build a CompletionResult from an /api/chat response."""
        text = strip_reasoning(str(data["message"].get("content") or ""))
        structured = None
        if req.json_schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            structured = parsed if isinstance(parsed, dict) else None
        return CompletionResult(
            text=text,
            structured=structured,
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            ),
            latency_s=latency,
            truncated=data.get("done_reason") == "length",
            backend=self.name,
            model=req.model,
            raw={k: v for k, v in data.items() if k != "message"},
        )

    async def health(self, required_models: Iterable[str] = ()) -> BackendHealth:
        """Check the server is up, models are pulled, and nothing spills to CPU."""
        health = BackendHealth(backend=self.name, ok=True)
        try:
            version = await self._get("/api/version")
            tags = await self._get("/api/tags")
            ps = await self._get("/api/ps")
        except BackendError as exc:
            health.problems.append(str(exc))
            health.ok = False
            return health
        health.version = str(version.get("version"))
        pulled = {m.get("name") for m in tags.get("models", [])}
        for model in sorted(set(required_models)):
            if model in pulled:
                health.details.append(f"{model} pulled")
            else:
                health.problems.append(f"model {model} is not pulled; {pull_hint(model)}")
        health.warnings += offload_warnings(ps.get("models", []))
        health.ok = not health.problems
        return health

    async def _get(self, path: str) -> dict[str, Any]:
        """GET a JSON endpoint, mapping connection failures to BackendUnavailableError."""
        try:
            resp = await self.client.get(path, timeout=HEALTH_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TransportError as exc:
            raise BackendUnavailableError(
                f"Ollama unreachable at {self.config.base_url}; {START_HINT}"
            ) from exc
        except (httpx.HTTPStatusError, json.JSONDecodeError) as exc:
            raise BackendError(f"Ollama {path} failed: {exc}") from exc
        return data if isinstance(data, dict) else {}


def offload_warnings(running: list[dict[str, Any]]) -> list[str]:
    """Warn about resident models that don't fit fully in VRAM (slow CPU offload)."""
    warnings = []
    for m in running:
        size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
        if size and vram < size:
            warnings.append(
                f"{m.get('name')} is partly on CPU ({vram / 1e9:.1f}/{size / 1e9:.1f} GB in VRAM). "
                "Fine for mixture-of-experts models (qwen3.6, gemma4: ~35-45 tok/s measured); "
                "a dense one is much slower (qwen3.6:27b: ~5 tok/s, so 30-100 s turns)"
            )
    return warnings


def _stream_chunk(line: str) -> dict[str, Any] | None:
    """Parse one NDJSON line of a streamed reply; None for blank lines."""
    if not line.strip():
        return None
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as exc:
        raise BackendOutputError("Ollama streamed non-JSON output") from exc
    if not isinstance(chunk, dict):
        raise BackendOutputError("Ollama streamed an unexpected chunk")
    if "error" in chunk:
        raise BackendError(f"Ollama error: {str(chunk['error'])[:300]}")
    return chunk
