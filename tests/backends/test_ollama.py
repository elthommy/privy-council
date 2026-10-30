from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from privy_council.backends.base import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatMessage,
    CompletionRequest,
    ModelLoadError,
)
from privy_council.backends.ollama import OllamaBackend, strip_reasoning
from privy_council.config import OllamaConfig, OllamaModelConfig

BASE = "http://localhost:11434"
CHAT_URL = f"{BASE}/api/chat"


def chat_response(content: str = "No. Six months is excessive.") -> dict[str, Any]:
    """Build an /api/chat response like the one observed in M0."""
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 59,
        "eval_count": 24,
        "load_duration": 6_000_000_000,
    }


def request(**overrides: Any) -> CompletionRequest:
    """Build a qwen3:8b request."""
    fields: dict[str, Any] = {
        "system": "You are Rigor.",
        "messages": [ChatMessage(role="user", content="Cash?")],
        "model": "qwen3:8b",
        "temperature": 0.6,
        "max_output_tokens": 200,
    }
    fields.update(overrides)
    return CompletionRequest(**fields)


@pytest.fixture
async def backend() -> AsyncIterator[OllamaBackend]:
    """An OllamaBackend with qwen3:8b configured like config.toml."""
    config = OllamaConfig(models={"qwen3:8b": OllamaModelConfig(num_ctx=4096, think=False)})
    b = OllamaBackend(config)
    yield b
    await b.aclose()


def sent_json(httpx_mock: HTTPXMock) -> dict[str, Any]:
    """Return the JSON body of the single request sent."""
    req = httpx_mock.get_request()
    assert req is not None
    body: dict[str, Any] = json.loads(req.content)
    return body


async def test_chat_payload_and_result(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=CHAT_URL, method="POST", json=chat_response())
    result = await backend.complete(request())
    body = sent_json(httpx_mock)
    assert body["messages"][0] == {"role": "system", "content": "You are Rigor."}
    assert body["messages"][1] == {"role": "user", "content": "Cash?"}
    assert body["stream"] is False
    assert body["think"] is False
    assert body["keep_alive"] == "10m"
    assert body["options"] == {"temperature": 0.6, "num_ctx": 4096, "num_predict": 200}
    assert "format" not in body
    assert result.text == "No. Six months is excessive."
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (59, 24)
    assert result.raw["load_duration"] == 6_000_000_000
    assert result.truncated is False


async def test_a_reasoning_draft_before_think_end_is_dropped(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    reply = "## Consensus\nDraft.</think>## Consensus\nFinal."
    httpx_mock.add_response(url=CHAT_URL, method="POST", json=chat_response(reply))
    assert (await backend.complete(request())).text == "## Consensus\nFinal."


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("Plain answer.", "Plain answer."),
        ("<think>why</think>\n\nAnswer.", "Answer."),
        ("Only a draft.</think>", "Only a draft."),
        ("a</think>b</think> c", "c"),
    ],
)
def test_strip_reasoning(text: str, answer: str) -> None:
    assert strip_reasoning(text) == answer


async def test_hard_cap_marks_truncated(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=CHAT_URL, json={**chat_response("and another thing"), "done_reason": "length"}
    )
    result = await backend.complete(request(max_output_tokens=1500))
    assert sent_json(httpx_mock)["options"]["num_predict"] == 1500
    assert result.truncated is True


async def test_unconfigured_model_uses_defaults(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=CHAT_URL, json=chat_response())
    await backend.complete(request(model="llama3.1:8b"))
    body = sent_json(httpx_mock)
    assert "think" not in body
    assert body["options"]["num_ctx"] == 8192


async def test_num_batch_is_sent_only_when_configured(httpx_mock: HTTPXMock) -> None:
    models = {"big:120b": OllamaModelConfig(num_batch=2048), "qwen3:8b": OllamaModelConfig()}
    b = OllamaBackend(OllamaConfig(models=models))
    httpx_mock.add_response(url=CHAT_URL, json=chat_response())
    httpx_mock.add_response(url=CHAT_URL, json=chat_response())
    await b.complete(request(model="big:120b"))
    await b.complete(request())
    await b.aclose()
    first, second = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert first["options"]["num_batch"] == 2048
    assert "num_batch" not in second["options"]


async def test_json_schema_sets_format(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=CHAT_URL, json=chat_response('{"stop": true}'))
    schema = {"type": "object"}
    result = await backend.complete(request(json_schema=schema))
    assert sent_json(httpx_mock)["format"] == schema
    assert result.structured == {"stop": True}


async def test_bad_json_with_schema_gives_no_structured(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=CHAT_URL, json=chat_response("not json"))
    result = await backend.complete(request(json_schema={}))
    assert result.structured is None


async def test_server_down(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    with pytest.raises(BackendUnavailableError, match="systemctl start ollama"):
        await backend.complete(request())


async def test_timeout(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendTimeoutError):
        await backend.complete(request())


async def test_missing_model(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=CHAT_URL, status_code=404, json={"error": "model not found"})
    with pytest.raises(BackendUnavailableError, match="ollama pull qwen3:8b"):
        await backend.complete(request())


async def test_server_error(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=CHAT_URL, status_code=500, text="oops")
    with pytest.raises(BackendError, match="500"):
        await backend.complete(request())


LOAD_FAILED = '{"error":"timed out waiting for llama-server to start - "}'
OOM = (
    '{"error":"an error was encountered while running the model: '
    'CUDA error\\nCUDA error: out of memory"}'
)


@pytest.mark.parametrize("body", [LOAD_FAILED, OOM])
@pytest.mark.parametrize("streamed", [False, True])
async def test_model_load_timeout_or_oom_is_retried_once(
    backend: OllamaBackend, httpx_mock: HTTPXMock, streamed: bool, body: str
) -> None:
    httpx_mock.add_response(url=CHAT_URL, status_code=500, text=body)
    final = {**chat_response("Loaded."), "done_reason": "stop"}
    httpx_mock.add_response(
        url=CHAT_URL,
        content=ndjson(final) if streamed else None,
        json=None if streamed else chat_response("Loaded."),
    )
    result = await backend.complete(request(), (lambda _: None) if streamed else None)
    assert result.text == "Loaded."
    assert len(httpx_mock.get_requests()) == 2


async def test_model_load_timeout_twice_fails(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=CHAT_URL, status_code=500, text=LOAD_FAILED, is_reusable=True)
    with pytest.raises(ModelLoadError, match="llama-server to start"):
        await backend.complete(request())
    assert len(httpx_mock.get_requests()) == 2


def _mock_health(httpx_mock: HTTPXMock, pulled: list[str], running: list[dict[str, Any]]) -> None:
    """Mock the three health endpoints."""
    httpx_mock.add_response(url=f"{BASE}/api/version", json={"version": "0.40.4"})
    httpx_mock.add_response(url=f"{BASE}/api/tags", json={"models": [{"name": m} for m in pulled]})
    httpx_mock.add_response(url=f"{BASE}/api/ps", json={"models": running})


async def test_health_ok(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    _mock_health(httpx_mock, ["qwen3:8b"], [])
    health = await backend.health(["qwen3:8b"])
    assert health.ok
    assert health.version == "0.40.4"


async def test_health_missing_model_and_offload(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    running = [{"name": "gemma3:12b-it-qat", "size": 9_800_000_000, "size_vram": 7_900_000_000}]
    _mock_health(httpx_mock, ["qwen3:8b"], running)
    health = await backend.health(["qwen3:8b", "mistral:7b"])
    assert not health.ok
    assert health.problems == ["model mistral:7b is not pulled; run `ollama pull mistral:7b`"]
    assert "gemma3:12b-it-qat is partly on CPU" in health.warnings[0]


async def test_health_server_down(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    health = await backend.health(["qwen3:8b"])
    assert not health.ok
    assert "systemctl start ollama" in health.problems[0]


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_semaphore_limits_concurrency(
    concurrency: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = OllamaBackend(OllamaConfig(concurrency=concurrency))
    active = peak = 0

    async def fake_post(payload: dict[str, Any], model: str, timeout: float) -> dict[str, Any]:
        """Track how many calls are in flight at once."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return chat_response()

    monkeypatch.setattr(backend, "_post_chat", fake_post)
    await asyncio.gather(*(backend.complete(request()) for _ in range(4)))
    await backend.aclose()
    assert peak == concurrency


def ndjson(*chunks: dict[str, Any]) -> bytes:
    """Encode a streamed /api/chat reply."""
    return b"".join(json.dumps(c).encode() + b"\n" for c in chunks)


def piece(text: str) -> dict[str, Any]:
    """One streamed content chunk."""
    return {"model": "qwen3:8b", "message": {"role": "assistant", "content": text}, "done": False}


async def test_streamed_reply_feeds_the_sink(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    final = {**chat_response(""), "done_reason": "stop"}
    httpx_mock.add_response(
        url=CHAT_URL, content=ndjson(piece("No. "), piece("Six months."), final)
    )
    seen: list[str] = []
    result = await backend.complete(request(), seen.append)
    assert sent_json(httpx_mock)["stream"] is True
    assert seen == ["No. ", "Six months."]
    assert result.text == "No. Six months."
    assert result.usage is not None and result.usage.output_tokens == 24
    assert result.truncated is False


async def test_streamed_reply_cut_at_the_cap(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    final = {**chat_response(""), "done_reason": "length"}
    httpx_mock.add_response(url=CHAT_URL, content=ndjson(piece("and another"), final))
    result = await backend.complete(request(), lambda _: None)
    assert result.truncated is True and result.text == "and another"


async def test_streamed_errors(backend: OllamaBackend, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=CHAT_URL, content=ndjson(piece("a"), {"error": "model crashed"}))
    with pytest.raises(BackendError, match="Ollama error: model crashed"):
        await backend.complete(request(), lambda _: None)
    httpx_mock.add_response(url=CHAT_URL, content=ndjson(piece("a")))
    with pytest.raises(BackendError, match="ended before its final chunk"):
        await backend.complete(request(), lambda _: None)
    httpx_mock.add_response(url=CHAT_URL, status_code=404, text="model not found")
    with pytest.raises(BackendUnavailableError, match="ollama pull qwen3:8b"):
        await backend.complete(request(), lambda _: None)


async def test_json_requests_are_never_streamed(
    backend: OllamaBackend, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=CHAT_URL, json=chat_response('{"a": 1}'))
    seen: list[str] = []
    result = await backend.complete(request(json_schema={"type": "object"}), seen.append)
    assert sent_json(httpx_mock)["stream"] is False
    assert seen == [] and result.structured == {"a": 1}
