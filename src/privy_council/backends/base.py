"""Backend interface: request/result models, health report and typed errors."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """One conversation message sent to a model (the system prompt is separate)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class CallTag(BaseModel):
    """Who is calling and why; recorded in the usage ledger, never sent to a model."""

    model_config = ConfigDict(frozen=True)

    session: str | None = None
    persona: str | None = None
    purpose: str = "turn"


class CompletionRequest(BaseModel):
    """A single-turn completion request, backend-agnostic."""

    model_config = ConfigDict(frozen=True)

    system: str
    messages: list[ChatMessage] = Field(min_length=1)
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # ignored by claude_cli
    # Safety net against runaway output, not a style limit: output past it is cut.
    max_output_tokens: int = Field(default=1500, gt=0)
    # When set, the backend asks for JSON matching this schema (JSON mode).
    json_schema: dict[str, Any] | None = None
    # CLI tools to enable, e.g. web search (claude_cli only; others ignore it).
    tools: list[str] = Field(default_factory=list)
    timeout_s: float | None = Field(default=None, gt=0)
    tag: CallTag = Field(default_factory=CallTag)


class Usage(BaseModel):
    """Token usage reported by the backend."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class CompletionResult(BaseModel):
    """What a backend returns for one completion."""

    text: str
    structured: dict[str, Any] | None = None
    usage: Usage | None = None
    latency_s: float
    truncated: bool = False  # True when the hard output cap cut the reply
    backend: str
    model: str
    raw: dict[str, Any] = Field(default_factory=dict)


class BackendHealth(BaseModel):
    """Health-check outcome; problems are blocking, warnings are not."""

    backend: str
    ok: bool
    version: str | None = None
    details: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class BackendError(Exception):
    """Base class for backend failures; the message is shown to the user."""


class BackendAuthError(BackendError):
    """The backend is not authenticated (e.g. `claude` not logged in)."""


class BackendRateLimitError(BackendError):
    """Usage or rate limit reached; `retry_at` is set when the backend says when."""

    def __init__(self, message: str, retry_at: datetime | None = None) -> None:
        """Store the user-facing message and the optional reset time."""
        super().__init__(message)
        self.retry_at = retry_at


class BackendTimeoutError(BackendError):
    """The call exceeded its timeout and was cancelled."""


class BackendOutputError(BackendError):
    """The backend returned output we couldn't parse."""


class BackendUnavailableError(BackendError):
    """The backend can't serve the call (server down, model not pulled)."""


class ModelLoadError(BackendError):
    """The model didn't finish loading in time (Ollama retries it once by itself)."""


class LocalOnlyError(BackendError):
    """A call would reach Claude in a local-only session (M15); it never runs."""


# Receives the reply text as it is generated, chunk by chunk (the TUI typing effect).
TextSink = Callable[[str], None]


class Backend(Protocol):
    """Common async interface every backend implements."""

    name: str

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Run one completion; with `on_text`, stream the reply text into it as well."""
        ...

    async def health(self) -> BackendHealth:
        """Check that the backend is usable, with actionable hints."""
        ...
