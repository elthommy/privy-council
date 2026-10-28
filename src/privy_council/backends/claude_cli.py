"""Claude backend through the `claude` CLI in headless mode (Pro subscription only).

ALL `claude` flag handling lives in this module. Flags were verified on Claude
Code 2.1.301; see docs/spike-notes.md before changing them. Never use `--bare`:
it forces API-key auth.

Calls use `--output-format stream-json`: the CLI has no max-tokens flag, so we
count output as it streams and kill the process past the hard cap, keeping the
partial text. The stream also carries a `rate_limit_event` with real Pro quota
utilization. Its final `result` event has the same shape as `--output-format json`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from privy_council.backends.base import (
    BackendAuthError,
    BackendError,
    BackendHealth,
    BackendOutputError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnavailableError,
    CompletionRequest,
    CompletionResult,
    TextSink,
    Usage,
)
from privy_council.config import ClaudeConfig
from privy_council.engine.budget import Ledger, QuotaSnapshot, UsageEntry
from privy_council.engine.prompts import flatten_messages
from privy_council.util.tokens import estimate_tokens, tokens_to_chars

log = logging.getLogger(__name__)

# Removed from the child env so the CLI can only use the subscription login.
API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")
# Subscription token (`claude setup-token`): kept for real calls, dropped for the probe.
OAUTH_TOKEN_ENV_VAR = "CLAUDE_CODE_OAUTH_TOKEN"

HEADLESS_FLAGS = (
    "-p",
    "--output-format", "stream-json",
    "--verbose",                   # required by stream-json in print mode
    "--include-partial-messages",  # token deltas, so the hard cap can cut early
    "--no-session-persistence",
    "--safe-mode",                 # no CLAUDE.md, skills, plugins, hooks, MCP
    "--strict-mcp-config",
    "--setting-sources", "",
)  # fmt: skip

# Tool through which --json-schema output streams (input_json_delta chunks).
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

AUTH_HINT = "run `claude auth login` (subscription login, not an API key)"
_AUTH_MARKERS = ("not logged in", "/login", "invalid api key", "oauth token")
_LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached|limit will reset", re.IGNORECASE)
_RESET_EPOCH_RE = re.compile(r"\|(\d{10})\b")
PROBE_TIMEOUT_S = 20.0
# Stream events can be large (full messages, signatures); asyncio's 64 KiB default is too small.
STREAM_LINE_LIMIT = 16 * 1024 * 1024
EXIT_GRACE_S = 5.0


@dataclass(frozen=True)
class ProcResult:
    """Exit code and decoded output of a finished subprocess."""

    returncode: int
    stdout: str
    stderr: str


@dataclass
class StreamCollector:
    """Accumulates stream-json events; reports when visible output passes the cap."""

    cap_chars: int | None
    on_text: TextSink | None = None  # streamed copy of the visible text
    text: str = ""
    json_text: str = ""  # structured output arrives as input_json_delta chunks
    result: dict[str, Any] | None = None
    rate_limit: dict[str, Any] | None = None
    start_usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    # Inputs of tools the model called (e.g. WebSearch queries), in order.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Tool of the current content block: None = unknown (treat JSON as structured
    # output, the shape seen without tools), "" = not a tool block.
    _block_tool: str | None = None
    _tool_input: str = ""

    def feed(self, line: bytes) -> bool:
        """Consume one line; return True once the cap is exceeded (caller kills)."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        kind = event.get("type")
        if kind == "stream_event":
            self._on_stream_event(event.get("event") or {})
        elif kind == "rate_limit_event":
            self.rate_limit = event.get("rate_limit_info") or {}
        elif kind == "result":
            self.result = event
        over = self.cap_chars is not None and len(self.text) + len(self.json_text) > self.cap_chars
        if over and self.result is None:
            self.truncated = True
        return self.truncated

    def _on_stream_event(self, inner: dict[str, Any]) -> None:
        """Collect text/JSON deltas, tool inputs and the input usage from message_start."""
        kind = inner.get("type")
        delta = inner.get("delta") or {}
        if delta.get("type") == "text_delta":
            chunk = str(delta.get("text") or "")
            self.text += chunk
            if chunk and self.on_text is not None:
                self.on_text(chunk)
        elif delta.get("type") == "input_json_delta":
            chunk = str(delta.get("partial_json") or "")
            if self._block_tool in (None, STRUCTURED_OUTPUT_TOOL):
                self.json_text += chunk
            else:
                self._tool_input += chunk
        elif kind == "content_block_start":
            block = inner.get("content_block") or {}
            self._block_tool = (
                str(block.get("name") or "") if block.get("type") == "tool_use" else ""
            )
            self._tool_input = ""
        elif kind == "content_block_stop":
            self._finish_tool_block()
        elif kind == "message_start":
            self.start_usage = (inner.get("message") or {}).get("usage") or {}

    def _finish_tool_block(self) -> None:
        """Record the input of a finished (non-structured-output) tool call."""
        if self._block_tool and self._block_tool != STRUCTURED_OUTPUT_TOOL:
            try:
                tool_input = json.loads(self._tool_input or "{}")
            except json.JSONDecodeError:
                tool_input = {"raw": self._tool_input}
            self.tool_calls.append({"name": self._block_tool, "input": tool_input})
        self._block_tool, self._tool_input = None, ""

    @property
    def visible_text(self) -> str:
        """Return the streamed output: structured JSON if any, else text."""
        return self.json_text or self.text


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ minus API-key variables, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if k not in API_KEY_ENV_VARS}
    env.update(extra or {})
    return env


def api_keys_in_env() -> list[str]:
    """Return the API-key variables set in the parent environment."""
    return [k for k in API_KEY_ENV_VARS if os.environ.get(k)]


def build_argv(
    binary: str,
    model: str,
    system_prompt_file: Path,
    json_schema: dict[str, Any] | None,
    tools: Sequence[str] = (),
) -> list[str]:
    """Build the headless argv. Prompt text never goes here, only a file path.

    No tools by default (one turn, nothing to loop on). Requested tools are also
    pre-approved, since print mode can't answer a permission prompt.
    """
    argv = [binary, *HEADLESS_FLAGS, "--tools", ",".join(tools)]
    if tools:
        argv += ["--allowedTools", ",".join(tools)]
    argv += ["--model", model]
    argv += ["--system-prompt-file", str(system_prompt_file)]
    if json_schema is not None:
        argv += ["--json-schema", json.dumps(json_schema)]
    return argv


async def _spawn(argv: Sequence[str], cwd: Path, env: dict[str, str]) -> asyncio.subprocess.Process:
    """Start argv without a shell, with all three pipes."""
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_LINE_LIMIT,
    )


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Kill the process if it is still running, and wait for it."""
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


async def run_process(
    argv: Sequence[str], stdin: str, cwd: Path, env: dict[str, str], timeout_s: float
) -> ProcResult:
    """Run a short command to completion; kill and reap it on timeout or cancel."""
    proc = await _spawn(argv, cwd, env)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin.encode()), timeout_s)
    except TimeoutError as exc:
        raise BackendTimeoutError(f"claude CLI timed out after {timeout_s:.0f}s") from exc
    finally:
        await _reap(proc)
    return ProcResult(
        proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")
    )


async def _send_stdin(proc: asyncio.subprocess.Process, data: str) -> None:
    """Write the prompt and close stdin; a CLI that exits early is not an error here."""
    assert proc.stdin is not None
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        proc.stdin.write(data.encode())
        await proc.stdin.drain()
        proc.stdin.close()


async def run_stream(
    argv: Sequence[str],
    stdin: str,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    cap_chars: int | None,
    on_text: TextSink | None = None,
) -> tuple[StreamCollector, ProcResult]:
    """Run a streaming call, killing it at the output cap, timeout or cancellation."""
    proc = await _spawn(argv, cwd, env)
    assert proc.stdout is not None and proc.stderr is not None
    stderr_task = asyncio.create_task(proc.stderr.read())
    collector = StreamCollector(cap_chars, on_text)
    try:
        async with asyncio.timeout(timeout_s):
            await _send_stdin(proc, stdin)
            async for line in proc.stdout:
                if collector.feed(line):
                    break
        if not collector.truncated:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), EXIT_GRACE_S)
    except TimeoutError as exc:
        raise BackendTimeoutError(f"claude CLI timed out after {timeout_s:.0f}s") from exc
    finally:
        await _reap(proc)
        stderr = (await stderr_task).decode(errors="replace")
    return collector, ProcResult(proc.returncode or 0, "", stderr)


def parse_retry_at(text: str) -> datetime | None:
    """Extract a reset time from a limit message, if it carries an epoch timestamp."""
    match = _RESET_EPOCH_RE.search(text)
    return datetime.fromtimestamp(int(match.group(1)), UTC) if match else None


def classify_error(envelope: dict[str, Any], quota: QuotaSnapshot | None = None) -> BackendError:
    """Map an `is_error` result envelope (plus quota info, if any) to a typed error."""
    text = str(envelope.get("result") or envelope.get("terminal_reason") or "unknown error")
    status = envelope.get("api_error_status")
    lowered = text.lower()
    if status in (401, 403) or any(m in lowered for m in _AUTH_MARKERS):
        return BackendAuthError(f"Claude CLI is not authenticated: {text} → {AUTH_HINT}")
    limited = quota is not None and quota.status not in (None, "allowed")
    if limited or status == 429 or _LIMIT_RE.search(text):
        retry_at = parse_retry_at(text) or (quota.resets_at if quota else None)
        return BackendRateLimitError(f"Claude usage limit reached: {text}", retry_at)
    if status == 529 or "overloaded" in lowered:
        return BackendUnavailableError(f"Claude is overloaded, try again shortly: {text}")
    log.warning(
        "unrecognised claude error: status=%s terminal_reason=%s keys=%s",
        status,
        envelope.get("terminal_reason"),
        sorted(envelope),
    )
    return BackendError(f"claude CLI error: {text}")


def check_envelope(
    envelope: dict[str, Any] | None, proc: ProcResult, quota: QuotaSnapshot | None = None
) -> dict[str, Any]:
    """Validate the final result envelope, raising typed errors on failure."""
    if envelope is None:
        if "unknown option" in proc.stderr:
            raise BackendOutputError(
                f"claude CLI rejected a flag ({proc.stderr.strip()}); the CLI may have "
                "changed, run `council doctor`"
            )
        tail = proc.stderr.strip()[-300:]
        raise BackendOutputError(
            f"claude CLI ended without a result (exit {proc.returncode}): {tail}"
        )
    if envelope.get("is_error"):
        # Note: `subtype` says "success" even for errors; trust `is_error` only.
        raise classify_error(envelope, quota)
    if not isinstance(envelope.get("result"), str):
        raise BackendOutputError("claude CLI result has no 'result' text")
    return envelope


def _usage(usage: Any) -> Usage | None:
    """Convert an API usage object to Usage, if present."""
    if not isinstance(usage, dict):
        return None
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )


def _structured(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Return the structured output object, falling back to parsing `result` as JSON."""
    value = envelope.get("structured_output")
    if value is None:
        try:
            value = json.loads(envelope["result"])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def to_result(
    envelope: dict[str, Any], req: CompletionRequest, latency_s: float
) -> CompletionResult:
    """Build a CompletionResult from a successful envelope."""
    model_usage = envelope.get("modelUsage")
    resolved = next(iter(model_usage), req.model) if isinstance(model_usage, dict) else req.model
    return CompletionResult(
        text=envelope["result"],
        structured=_structured(envelope) if req.json_schema is not None else None,
        usage=_usage(envelope.get("usage")),
        latency_s=latency_s,
        backend=ClaudeCliBackend.name,
        model=resolved,
        raw={k: v for k, v in envelope.items() if k not in ("result", "structured_output")},
    )


def truncated_result(
    collector: StreamCollector, req: CompletionRequest, latency_s: float
) -> CompletionResult:
    """Build a result from the partial text of a call cut at the hard cap."""
    usage = _usage(collector.start_usage) or Usage()
    text = collector.visible_text
    return CompletionResult(
        text=text,
        structured=None,  # partial JSON is never valid; the caller repairs or falls back
        usage=usage.model_copy(update={"output_tokens": estimate_tokens(text)}),
        latency_s=latency_s,
        truncated=True,
        backend=ClaudeCliBackend.name,
        model=req.model,
        raw={"truncated_at_chars": len(text)},
    )


class ClaudeCliBackend:
    """Runs one `claude -p` subprocess per completion, in an empty temp dir.

    The CLI has no temperature flag, so `temperature` is ignored.
    `max_output_tokens` is enforced by cutting the stream (chars/4 estimate).
    """

    name = "claude_cli"

    def __init__(self, config: ClaudeConfig, ledger: Ledger | None = None) -> None:
        """Configure the binary, timeout and optional usage ledger."""
        self.config = config
        self.ledger = ledger

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Run one completion and record it in the ledger, success or failure."""
        start = time.monotonic()
        quota: QuotaSnapshot | None = None
        envelope: dict[str, Any] = {}
        try:
            collector, proc = await self._invoke(req, on_text)
            if collector.rate_limit is not None:
                quota = QuotaSnapshot.from_event(collector.rate_limit)
            if collector.truncated:
                result = truncated_result(collector, req, time.monotonic() - start)
            else:
                envelope = check_envelope(collector.result, proc, quota)
                result = to_result(envelope, req, time.monotonic() - start)
        except BackendError as exc:
            self._record(req, time.monotonic() - start, error=exc, quota=quota)
            raise
        if quota is not None:
            result.raw["quota"] = quota.model_dump(mode="json")
        if collector.tool_calls:
            result.raw["tool_calls"] = collector.tool_calls
        self._record(req, result.latency_s, envelope=envelope, result=result, quota=quota)
        return result

    async def _invoke(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> tuple[StreamCollector, ProcResult]:
        """Write the system prompt file and stream the CLI's output."""
        with tempfile.TemporaryDirectory(prefix="council-claude-") as tmp:
            workdir = Path(tmp) / "cwd"
            workdir.mkdir()  # stays empty: no CLAUDE.md or files leak into context
            system_file = Path(tmp) / "system.md"
            system_file.touch(mode=0o600)
            system_file.write_text(req.system, encoding="utf-8")
            argv = build_argv(
                self.config.binary, req.model, system_file, req.json_schema, req.tools
            )
            timeout = req.timeout_s or self.config.timeout_s
            cap = tokens_to_chars(req.max_output_tokens)
            try:
                return await run_stream(
                    argv,
                    flatten_messages(req.messages),
                    workdir,
                    child_env(),
                    timeout,
                    cap,
                    on_text,
                )
            except FileNotFoundError as exc:
                raise BackendUnavailableError(
                    f"claude CLI not found ({self.config.binary!r}); install Claude Code "
                    "or set [claude] binary"
                ) from exc

    def _record(
        self,
        req: CompletionRequest,
        latency_s: float,
        *,
        envelope: dict[str, Any] | None = None,
        result: CompletionResult | None = None,
        error: BackendError | None = None,
        quota: QuotaSnapshot | None = None,
    ) -> None:
        """Append a metadata-only entry to the ledger, if one is configured."""
        if self.ledger is None:
            return
        usage = result.usage if result and result.usage else Usage()
        env = envelope or {}
        self.ledger.record(
            UsageEntry(
                session=req.tag.session,
                persona=req.tag.persona,
                purpose=req.tag.purpose,
                model=result.model if result else req.model,
                ok=error is None,
                error=type(error).__name__ if error else None,
                input_tokens=usage.input_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                num_turns=env.get("num_turns"),
                latency_s=round(latency_s, 3),
                notional_usd=env.get("total_cost_usd"),
                truncated=bool(result and result.truncated),
                quota=quota,
            )
        )

    async def health(self) -> BackendHealth:
        """Check binary, version, flags and login without spending any quota."""
        health = BackendHealth(backend=self.name, ok=True)
        for key in api_keys_in_env():
            health.warnings.append(
                f"{key} is set in your shell; council strips it from the claude subprocess, "
                "but unset it to be safe"
            )
        try:
            version = await self._run_simple(["--version"])
        except (FileNotFoundError, BackendError) as exc:
            health.problems.append(f"claude CLI not usable ({self.config.binary!r}): {exc}")
            health.ok = False
            return health
        health.version = version.stdout.strip()
        await self._check_auth(health)
        await self._check_flags(health)
        health.ok = not health.problems
        return health

    async def _run_simple(self, args: list[str]) -> ProcResult:
        """Run a short, quota-free CLI command in a throwaway dir."""
        with tempfile.TemporaryDirectory(prefix="council-claude-") as tmp:
            return await run_process(
                [self.config.binary, *args], "", Path(tmp), child_env(), PROBE_TIMEOUT_S
            )

    async def _check_auth(self, health: BackendHealth) -> None:
        """Check `claude auth status` reports a subscription login."""
        proc = await self._run_simple(["auth", "status"])
        try:
            status = json.loads(proc.stdout)
        except json.JSONDecodeError:
            health.warnings.append("couldn't parse `claude auth status` output")
            return
        if not status.get("loggedIn"):
            health.problems.append(f"claude CLI is not logged in → {AUTH_HINT}")
            return
        method, plan = status.get("authMethod"), status.get("subscriptionType")
        health.details.append(f"logged in ({method}, {plan})")
        if method != "claude.ai":
            health.warnings.append(
                f"auth method is {method!r}, expected 'claude.ai' (subscription)"
            )

    async def _check_flags(self, health: BackendHealth) -> None:
        """Probe our exact flag set while logged out, so it can't spend quota."""
        with tempfile.TemporaryDirectory(prefix="council-probe-") as tmp:
            system_file = Path(tmp) / "system.md"
            system_file.write_text("probe", encoding="utf-8")
            argv = build_argv(self.config.binary, self.config.default_model, system_file, {})
            env = child_env({"CLAUDE_CONFIG_DIR": str(Path(tmp) / "empty-config")})
            env.pop(OAUTH_TOKEN_ENV_VAR, None)  # an env token would log the probe in
            collector, proc = await run_stream(argv, "probe", Path(tmp), env, PROBE_TIMEOUT_S, None)
        try:
            check_envelope(collector.result, proc)
        except BackendAuthError:
            health.details.append("headless flags accepted")
        except BackendError as exc:
            health.problems.append(f"flag probe failed: {exc} (see docs/spike-notes.md)")
        else:
            health.warnings.append("flag probe unexpectedly succeeded while logged out")
