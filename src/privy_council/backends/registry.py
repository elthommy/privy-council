"""Resolve which backend and model serve each persona, with local fallback."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from privy_council.backends.base import Backend, LocalOnlyError
from privy_council.backends.claude_cli import ClaudeCliBackend
from privy_council.backends.ollama import OllamaBackend
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.models.persona import BackendKind, Persona

LOCAL_ONLY_REFUSED = "local-only mode: this call would use Claude, so it was not made"


@dataclass(frozen=True)
class Route:
    """Where a persona's turn goes; `fallback` marks a re-routed Claude persona."""

    backend: Backend
    model: str
    kind: BackendKind
    fallback: bool = False


class BackendRegistry:
    """Creates one shared instance per backend kind, lazily.

    `overrides` replaces real backends (e.g. FakeBackend in tests). With
    `local_only`, the Claude backend is never handed out, overrides included: the
    hard guard of local-only mode (M15), whatever route a call takes.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: Ledger | None = None,
        overrides: Mapping[BackendKind, Backend] | None = None,
        local_only: bool = False,
    ) -> None:
        """Remember settings; backends are created on first use."""
        self.settings = settings
        self.ledger = ledger
        self.local_only = local_only
        self._backends: dict[BackendKind, Backend] = dict(overrides or {})

    def backend(self, kind: BackendKind) -> Backend:
        """Return the shared backend for `kind`, creating it if needed."""
        if self.local_only and kind is BackendKind.CLAUDE_CLI:
            raise LocalOnlyError(LOCAL_ONLY_REFUSED)
        if kind not in self._backends:
            if kind is BackendKind.CLAUDE_CLI:
                self._backends[kind] = ClaudeCliBackend(self.settings.claude, self.ledger)
            else:
                self._backends[kind] = OllamaBackend(self.settings.ollama)
        return self._backends[kind]

    def route(self, persona: Persona, claude_exhausted: bool = False) -> Route:
        """Route a persona; Claude personas go local when exhausted and fallback is on."""
        kind = persona.meta.backend
        claude = self.settings.claude
        if kind is BackendKind.CLAUDE_CLI and self.local_only:
            raise LocalOnlyError(LOCAL_ONLY_REFUSED)
        if kind is BackendKind.CLAUDE_CLI and claude_exhausted and claude.fallback_to_local:
            return Route(
                self.backend(BackendKind.OLLAMA),
                claude.fallback_model,
                BackendKind.OLLAMA,
                fallback=True,
            )
        return Route(self.backend(kind), persona.meta.model, kind)

    def required_ollama_models(self, personas: Iterable[Persona]) -> set[str]:
        """Return the Ollama models these personas (and the fallback) need pulled."""
        models = {p.meta.model for p in personas if p.meta.backend is BackendKind.OLLAMA}
        if self.settings.claude.fallback_to_local and not self.local_only:
            models.add(self.settings.claude.fallback_model)
        return models

    async def aclose(self) -> None:
        """Release backend resources (the Ollama HTTP client)."""
        for backend in self._backends.values():
            if isinstance(backend, OllamaBackend):
                await backend.aclose()
