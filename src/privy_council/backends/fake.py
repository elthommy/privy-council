"""Deterministic scripted backend for tests and demos. Never calls a model."""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from privy_council.backends.base import (
    BackendHealth,
    BackendOutputError,
    CompletionRequest,
    CompletionResult,
    TextSink,
    Usage,
)

# A scripted reply: text, a structured object (sent back as JSON text too), a
# ready-made CompletionResult (e.g. truncated=True), an exception to raise, or a
# function of the request returning text/object.
Reply = (
    str
    | dict[str, Any]
    | CompletionResult
    | Exception
    | Callable[[CompletionRequest], "str | dict[str, Any]"]
)


class FakeBackend:
    """Returns scripted replies in order and records every request it receives.

    When the script runs out it uses `responder` if given, else `default`
    (None means raise BackendOutputError, to catch unexpected extra calls).
    """

    def __init__(
        self,
        script: Iterable[Reply] = (),
        *,
        responder: Callable[[CompletionRequest], Reply] | None = None,
        default: str | None = "ok",
        name: str = "fake",
    ) -> None:
        """Set up the reply queue."""
        self.name = name
        self.script: deque[Reply] = deque(script)
        self.responder = responder
        self.default = default
        self.requests: list[CompletionRequest] = []

    def _next_reply(self, req: CompletionRequest) -> Reply:
        """Pick the next reply from the script, responder or default."""
        if self.script:
            return self.script.popleft()
        if self.responder is not None:
            return self.responder(req)
        if self.default is None:
            raise BackendOutputError(f"{self.name}: script exhausted")
        return self.default

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Return (or raise) the next scripted reply; stream it word by word to `on_text`."""
        self.requests.append(req)
        reply = self._next_reply(req)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, CompletionResult):
            return reply
        if callable(reply):
            reply = reply(req)
        structured = reply if isinstance(reply, dict) else None
        text = json.dumps(reply) if isinstance(reply, dict) else reply
        if on_text is not None and structured is None:
            for word in re.findall(r"\S+\s*", text):
                on_text(word)
        return CompletionResult(
            text=text,
            structured=structured,
            usage=Usage(input_tokens=len(req.system) // 4, output_tokens=len(text) // 4),
            latency_s=0.0,
            backend=self.name,
            model=req.model,
        )

    async def health(self) -> BackendHealth:
        """Always healthy."""
        return BackendHealth(backend=self.name, ok=True, version="fake")
