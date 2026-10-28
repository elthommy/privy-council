from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import (
    BackendAuthError,
    BackendError,
    BackendOutputError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
    CallTag,
    ChatMessage,
    CompletionRequest,
)
from privy_council.backends.claude_cli import (
    STREAM_LINE_LIMIT,
    ClaudeCliBackend,
    StreamCollector,
    build_argv,
    classify_error,
)
from privy_council.config import ClaudeConfig
from privy_council.engine.budget import Ledger

SECRET_SYSTEM = "SECRET-SYSTEM-persona prompt"
SECRET_PROMPT = "SECRET-PROMPT my salary is 1234"
RESETS_5H = 1792105200


def envelope(result: str = "Hello from Rigor", **extra: Any) -> dict[str, Any]:
    """Build a final `result` event like the ones observed in the spikes."""
    data: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "num_turns": 1,
        "terminal_reason": "completed",
        "api_error_status": None,
        "total_cost_usd": 0.0002,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 30,
            "cache_creation_input_tokens": 600,
            "cache_read_input_tokens": 0,
        },
        "modelUsage": {"claude-haiku-5-5": {"inputTokens": 2}},
    }
    data.update(extra)
    return data


def error_envelope(result: str, status: int | None = None) -> dict[str, Any]:
    """Build an is_error result event (note: subtype still says success)."""
    return envelope(result, is_error=True, terminal_reason="api_error", api_error_status=status)


def delta(kind: str, **fields: Any) -> dict[str, Any]:
    """Build a content_block_delta stream event."""
    return {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": kind, **fields}},
    }


def rate_limit(status: str = "allowed", five: float = 0.2) -> dict[str, Any]:
    """Build a rate_limit_event like the one observed in the stream spike."""
    windows = {
        "five_hour": {"utilization": five, "resetsAt": RESETS_5H},
        "seven_day": {"utilization": 0.08, "resetsAt": 1792353600},
    }
    info = {"status": status, "resetsAt": RESETS_5H, "unifiedWindows": windows}
    return {"type": "rate_limit_event", "rate_limit_info": info}


MESSAGE_START = {
    "type": "stream_event",
    "event": {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 2, "cache_creation_input_tokens": 610}},
    },
}


def text_stream(*chunks: str, final: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build a full successful stream: init, deltas, rate limit and result."""
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init"}, MESSAGE_START]
    events += [delta("text_delta", text=c) for c in chunks]
    events.append(rate_limit())
    events.append(final or envelope("".join(chunks)))
    return events


class FakeStdin:
    """Captures what the backend writes to stdin."""

    def __init__(self) -> None:
        """Start empty."""
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        """Record written bytes."""
        self.data += data

    async def drain(self) -> None:
        """No-op."""

    def close(self) -> None:
        """Record the close."""
        self.closed = True


class FakeStderr:
    """Returns canned stderr once the process has exited."""

    def __init__(self, proc: FakeProc, data: bytes) -> None:
        """Bind to the owning process."""
        self.proc, self.data = proc, data

    async def read(self) -> bytes:
        """Wait for exit (like a real pipe reaching EOF), then return the data."""
        while self.proc.returncode is None:
            await asyncio.sleep(0.001)
        return self.data


class FakeProc:
    """Stand-in for asyncio.subprocess.Process with a line-streaming stdout."""

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        *,
        stdout: bytes | None = None,
        stderr: bytes = b"",
        rc: int = 0,
        hang: bool = False,
    ) -> None:
        """Lines come from `events` (JSON-encoded) or raw `stdout`."""
        raw = (
            stdout
            if stdout is not None
            else b"".join(json.dumps(e).encode() + b"\n" for e in events or [])
        )
        self.lines = raw.splitlines(keepends=True)
        self.rc, self.hang = rc, hang
        self.returncode: int | None = None
        self.stdin = FakeStdin()
        self.stderr = FakeStderr(self, stderr)
        self.stdout = self._stdout()
        self.lines_read = 0
        self.killed = self.waited = False

    async def _stdout(self) -> AsyncIterator[bytes]:
        """Yield lines, then hang forever if asked to."""
        for line in self.lines:
            self.lines_read += 1
            yield line
        if self.hang:
            await asyncio.sleep(3600)

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        """Non-streaming path (health commands)."""
        self.stdin.write(data)
        self.returncode = self.rc
        return b"".join(self.lines), self.stderr.data

    def kill(self) -> None:
        """Record the kill."""
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        """Exit normally unless killed."""
        self.waited = True
        if self.returncode is None:
            self.returncode = self.rc
        return self.returncode


class Spawner:
    """Replaces create_subprocess_exec, capturing each call."""

    def __init__(self, *procs: FakeProc | Exception) -> None:
        """Queue the processes (or exceptions) to hand out."""
        self.procs = list(procs)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *argv: str, **kwargs: Any) -> FakeProc:
        """Capture argv/env/cwd and the temp files as they exist at spawn time."""
        cwd = Path(kwargs["cwd"])
        call: dict[str, Any] = {"argv": list(argv), "env": kwargs["env"], "kwargs": kwargs}
        call["cwd_files"] = list(cwd.iterdir())
        if "--system-prompt-file" in argv:
            sp = Path(argv[argv.index("--system-prompt-file") + 1])
            call["system_file"] = sp.read_text()
            call["system_mode"] = sp.stat().st_mode & 0o777
        self.calls.append(call)
        proc = self.procs.pop(0)
        if isinstance(proc, Exception):
            raise proc
        call["proc"] = proc
        return proc


@pytest.fixture
def spawn(monkeypatch: pytest.MonkeyPatch) -> Spawner:
    """Install a Spawner; tests queue processes on it."""
    spawner = Spawner()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawner)
    return spawner


def request(**overrides: Any) -> CompletionRequest:
    """Build a request carrying secret-looking prompt text."""
    fields: dict[str, Any] = {
        "system": SECRET_SYSTEM,
        "messages": [ChatMessage(role="user", content=SECRET_PROMPT)],
        "model": "haiku",
        "tag": CallTag(session="s1", persona="rigor", purpose="turn"),
    }
    fields.update(overrides)
    return CompletionRequest(**fields)


async def test_success_parses_and_isolates(
    spawn: Spawner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")
    spawn.procs.append(FakeProc(text_stream("Hello ", "from Rigor")))
    ledger = Ledger(tmp_path / "usage.jsonl")

    result = await ClaudeCliBackend(ClaudeConfig(), ledger).complete(request())

    assert result.text == "Hello from Rigor"
    assert result.truncated is False
    assert result.model == "claude-haiku-5-5"
    assert result.usage is not None and result.usage.cache_write_tokens == 600
    assert result.raw["quota"]["five_hour"] == 0.2
    call = spawn.calls[0]
    argv = call["argv"]
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "subscription-token"
    assert not any("SECRET" in a for a in argv)
    assert call["proc"].stdin.data == SECRET_PROMPT.encode()
    assert call["proc"].stdin.closed
    assert call["system_file"] == SECRET_SYSTEM
    assert call["system_mode"] == 0o600
    assert call["cwd_files"] == []
    assert call["kwargs"]["limit"] == STREAM_LINE_LIMIT
    assert "--bare" not in argv
    for flag in ["-p", "--verbose", "--include-partial-messages", "--safe-mode"]:
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--model") + 1] == "haiku"

    [entry] = ledger.entries()
    assert (entry.ok, entry.persona, entry.session, entry.num_turns) == (True, "rigor", "s1", 1)
    assert (entry.input_tokens, entry.output_tokens, entry.truncated) == (602, 30, False)
    assert entry.quota is not None and entry.quota.five_hour == 0.2
    assert entry.quota.five_hour_resets_at == datetime.fromtimestamp(RESETS_5H, UTC)


async def test_hard_cap_kills_and_keeps_partial_text(spawn: Spawner, tmp_path: Path) -> None:
    proc = FakeProc(text_stream(*["0123456789"] * 50))  # 500 chars, then a result
    spawn.procs.append(proc)
    ledger = Ledger(tmp_path / "usage.jsonl")

    result = await ClaudeCliBackend(ClaudeConfig(), ledger).complete(request(max_output_tokens=10))

    assert result.truncated is True
    assert result.text == "0123456789" * 5  # cap is 40 chars: cut on the 5th chunk
    assert proc.killed and proc.waited
    assert proc.lines_read < len(proc.lines)  # stopped reading early
    assert result.usage is not None and result.usage.cache_write_tokens == 610
    [entry] = ledger.entries()
    assert (entry.ok, entry.truncated, entry.output_tokens) == (True, True, 13)


async def test_json_schema_streams_structured_output(spawn: Spawner) -> None:
    obj = {"position": "yes", "confidence": 80}
    events = [
        delta("input_json_delta", partial_json='{"position": '),
        delta("input_json_delta", partial_json='"yes", "confidence": 80}'),
        envelope(json.dumps(obj), structured_output=obj, num_turns=2),
    ]
    spawn.procs.append(FakeProc(events))
    schema = {"type": "object"}
    result = await ClaudeCliBackend(ClaudeConfig()).complete(request(json_schema=schema))
    assert result.structured == obj
    argv = spawn.calls[0]["argv"]
    assert json.loads(argv[argv.index("--json-schema") + 1]) == schema


async def test_truncated_json_has_no_structured(spawn: Spawner) -> None:
    events = [delta("input_json_delta", partial_json='{"debate_state": "' + "x" * 100)]
    spawn.procs.append(FakeProc([*events, envelope("{}")]))
    req = request(json_schema={}, max_output_tokens=10)
    result = await ClaudeCliBackend(ClaudeConfig()).complete(req)
    assert result.truncated and result.structured is None
    assert result.text.startswith('{"debate_state"')


async def test_structured_falls_back_to_parsing_result(spawn: Spawner) -> None:
    spawn.procs.append(FakeProc([envelope('{"a": 1}')]))
    result = await ClaudeCliBackend(ClaudeConfig()).complete(request(json_schema={}))
    assert result.structured == {"a": 1}


async def test_multi_message_is_flattened_on_stdin(spawn: Spawner) -> None:
    spawn.procs.append(FakeProc(text_stream("ok")))
    msgs = [
        ChatMessage(role="user", content="Q1"),
        ChatMessage(role="assistant", content="A1"),
        ChatMessage(role="user", content="Q2"),
    ]
    await ClaudeCliBackend(ClaudeConfig()).complete(request(messages=msgs))
    stdin = spawn.calls[0]["proc"].stdin.data.decode()
    assert stdin == "[User]\nQ1\n\n[Assistant]\nA1\n\n[User]\nQ2"


async def test_not_logged_in_is_auth_error_and_recorded(spawn: Spawner, tmp_path: Path) -> None:
    events = [
        {"type": "system", "subtype": "init"},
        error_envelope("Not logged in · Please run /login"),
    ]
    spawn.procs.append(FakeProc(events, rc=1))
    ledger = Ledger(tmp_path / "usage.jsonl")
    with pytest.raises(BackendAuthError, match="claude auth login"):
        await ClaudeCliBackend(ClaudeConfig(), ledger).complete(request())
    [entry] = ledger.entries()
    assert (entry.ok, entry.error) == (False, "BackendAuthError")


async def test_rate_limit_from_rejected_quota_event(spawn: Spawner, tmp_path: Path) -> None:
    events = [rate_limit("rejected", five=1.0), error_envelope("Something went wrong")]
    spawn.procs.append(FakeProc(events, rc=1))
    ledger = Ledger(tmp_path / "usage.jsonl")
    with pytest.raises(BackendRateLimitError) as excinfo:
        await ClaudeCliBackend(ClaudeConfig(), ledger).complete(request())
    assert excinfo.value.retry_at == datetime.fromtimestamp(RESETS_5H, UTC)
    [entry] = ledger.entries()
    assert entry.quota is not None and entry.quota.status == "rejected"


async def test_rate_limit_with_reset_epoch_in_text(spawn: Spawner) -> None:
    events = [error_envelope("Claude AI usage limit reached|1760000000", 429)]
    spawn.procs.append(FakeProc(events, rc=1))
    with pytest.raises(BackendRateLimitError) as excinfo:
        await ClaudeCliBackend(ClaudeConfig()).complete(request())
    assert excinfo.value.retry_at == datetime.fromtimestamp(1760000000, UTC)


async def test_rate_limit_detected_from_text_only(spawn: Spawner) -> None:
    spawn.procs.append(FakeProc([error_envelope("5-hour limit reached ∙ resets 3pm")], rc=1))
    with pytest.raises(BackendRateLimitError) as excinfo:
        await ClaudeCliBackend(ClaudeConfig()).complete(request())
    assert excinfo.value.retry_at is None


async def test_timeout_kills_and_reaps(spawn: Spawner) -> None:
    proc = FakeProc([delta("text_delta", text="thinking...")], hang=True)
    spawn.procs.append(proc)
    with pytest.raises(BackendTimeoutError):
        await ClaudeCliBackend(ClaudeConfig()).complete(request(timeout_s=0.05))
    assert proc.killed and proc.waited


async def test_garbage_output(spawn: Spawner) -> None:
    spawn.procs.append(FakeProc(stdout=b"segfault\x00garbage\n[1,2]\n", stderr=b"boom", rc=139))
    with pytest.raises(BackendOutputError, match="without a result"):
        await ClaudeCliBackend(ClaudeConfig()).complete(request())


async def test_unknown_flag_points_to_doctor(spawn: Spawner) -> None:
    spawn.procs.append(FakeProc(stdout=b"", stderr=b"error: unknown option '--safe-mode'", rc=1))
    with pytest.raises(BackendOutputError, match="council doctor"):
        await ClaudeCliBackend(ClaudeConfig()).complete(request())


async def test_missing_binary(spawn: Spawner) -> None:
    spawn.procs.append(FileNotFoundError("claude"))
    with pytest.raises(BackendUnavailableError, match=r"\[claude\] binary"):
        await ClaudeCliBackend(ClaudeConfig()).complete(request())


def test_unrecognised_error_is_generic() -> None:
    err = classify_error({"is_error": True, "result": "something odd", "api_error_status": 500})
    assert type(err) is BackendError


def test_collector_ignores_noise_and_stops_after_result() -> None:
    collector = StreamCollector(cap_chars=5)
    assert collector.feed(b"not json\n") is False
    assert collector.feed(b"[1, 2]\n") is False
    assert collector.feed(json.dumps(envelope("done")).encode()) is False
    # Text after the result never triggers a cut: the answer is already complete.
    assert collector.feed(json.dumps(delta("text_delta", text="x" * 50)).encode()) is False
    assert collector.result is not None


def _health_spawner(auth: dict[str, Any], probe: FakeProc) -> Spawner:
    """Queue --version, auth status and flag-probe processes, in that order."""
    return Spawner(
        FakeProc(stdout=b"2.1.301 (Claude Code)\n"),
        FakeProc(stdout=json.dumps(auth).encode()),
        probe,
    )


LOGGED_IN = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"}


def probe_ok() -> FakeProc:
    """A logged-out flag probe: accepted flags, then 'Not logged in'."""
    return FakeProc(
        [
            {"type": "system", "subtype": "init"},
            error_envelope("Not logged in · Please run /login"),
        ],
        rc=1,
    )


async def test_health_ok_and_probe_is_logged_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    spawner = _health_spawner(LOGGED_IN, probe_ok())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawner)
    health = await ClaudeCliBackend(ClaudeConfig()).health()
    assert health.ok, health
    assert health.version == "2.1.301 (Claude Code)"
    assert "logged in (claude.ai, pro)" in health.details
    probe_env = spawner.calls[2]["env"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in probe_env
    assert "empty-config" in probe_env["CLAUDE_CONFIG_DIR"]


async def test_health_logged_out(monkeypatch: pytest.MonkeyPatch) -> None:
    spawner = _health_spawner({"loggedIn": False}, probe_ok())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawner)
    health = await ClaudeCliBackend(ClaudeConfig()).health()
    assert not health.ok
    assert any("claude auth login" in p for p in health.problems)


async def test_health_flag_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = FakeProc(stdout=b"", stderr=b"error: unknown option '--safe-mode'", rc=1)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _health_spawner(LOGGED_IN, probe))
    health = await ClaudeCliBackend(ClaudeConfig()).health()
    assert not health.ok
    assert any("flag probe failed" in p for p in health.problems)


async def test_health_warns_on_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _health_spawner(LOGGED_IN, probe_ok()))
    health = await ClaudeCliBackend(ClaudeConfig()).health()
    assert health.ok
    assert any("ANTHROPIC_API_KEY" in w for w in health.warnings)


async def test_health_missing_binary(spawn: Spawner) -> None:
    spawn.procs.append(FileNotFoundError("claude"))
    health = await ClaudeCliBackend(ClaudeConfig()).health()
    assert not health.ok


def block(kind: str, **fields: Any) -> dict[str, Any]:
    """Build a content_block_start / _stop stream event."""
    inner: dict[str, Any] = {"type": kind}
    if kind == "content_block_start":
        inner["content_block"] = fields
    return {"type": "stream_event", "event": inner}


def feed_all(collector: StreamCollector, events: list[dict[str, Any]]) -> None:
    """Feed events as stream-json lines."""
    for event in events:
        collector.feed(json.dumps(event).encode())


def test_argv_tools_off_by_default_and_preapproved_when_on(tmp_path: Path) -> None:
    plain = build_argv("claude", "sonnet", tmp_path / "s.md", None)
    assert plain[plain.index("--tools") + 1] == ""
    assert "--allowedTools" not in plain
    web = build_argv("claude", "sonnet", tmp_path / "s.md", None, ["WebSearch", "WebFetch"])
    assert web[web.index("--tools") + 1] == "WebSearch,WebFetch"
    assert web[web.index("--allowedTools") + 1] == "WebSearch,WebFetch"
    assert web.count("--tools") == 1


def test_collector_separates_tool_inputs_from_structured_output() -> None:
    collector = StreamCollector(cap_chars=None)
    feed_all(
        collector,
        [
            block("content_block_start", type="tool_use", name="WebSearch"),
            delta("input_json_delta", partial_json='{"query": "ecb '),
            delta("input_json_delta", partial_json='rate"}'),
            block("content_block_stop"),
            block("content_block_start", type="text"),
            delta("text_delta", text="- fact"),
            block("content_block_stop"),
            block("content_block_start", type="tool_use", name="StructuredOutput"),
            delta("input_json_delta", partial_json='{"stop": true}'),
            block("content_block_stop"),
        ],
    )
    assert collector.tool_calls == [{"name": "WebSearch", "input": {"query": "ecb rate"}}]
    assert collector.json_text == '{"stop": true}'
    assert collector.text == "- fact"


LONG_QUERY = "gold price and central bank buying " * 4  # longer than the 40-char cap


async def test_web_call_passes_tools_and_reports_queries(spawn: Spawner) -> None:
    events = [
        block("content_block_start", type="tool_use", name="WebSearch"),
        delta("input_json_delta", partial_json=json.dumps({"query": LONG_QUERY})),
        block("content_block_stop"),
        delta("text_delta", text="- gold is up"),
        envelope("- gold is up"),
    ]
    spawn.procs.append(FakeProc(events))
    req = request(tools=["WebSearch", "WebFetch"], max_output_tokens=10)
    result = await ClaudeCliBackend(ClaudeConfig()).complete(req)
    assert not result.truncated  # tool input JSON doesn't count toward the cap
    assert result.text == "- gold is up"
    assert result.raw["tool_calls"] == [{"name": "WebSearch", "input": {"query": LONG_QUERY}}]
    argv = spawn.calls[0]["argv"]
    assert argv[argv.index("--allowedTools") + 1] == "WebSearch,WebFetch"


def test_collector_streams_text_but_not_json() -> None:
    seen: list[str] = []
    collector = StreamCollector(cap_chars=None, on_text=seen.append)

    def delta(kind: str, **fields: str) -> dict[str, Any]:
        """A stream_event carrying one content delta."""
        return {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": kind, **fields}},
        }

    feed_all(
        collector,
        [
            delta("text_delta", text="Hello "),
            delta("thinking_delta", thinking="hmm"),
            delta("input_json_delta", partial_json='{"a"'),
            delta("text_delta", text="world"),
        ],
    )
    assert seen == ["Hello ", "world"]
    assert collector.text == "Hello world"
