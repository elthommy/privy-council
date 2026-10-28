"""Building chat sessions: always local-only, memory off, kept apart in data/chat/.

Whatever `config.toml` says: no call reaches Claude (the registry's local-only
guard), no persona memory is read or written, and sessions live under
`data/chat/`, so `council sessions` and the private memory stay separate. Chat
sessions search when local search is enabled (user decision, 2026-10-24).
"""

from __future__ import annotations

from datetime import UTC, datetime

from privy_council.backends.registry import BackendRegistry
from privy_council.chat.frontend import ChatSetupError
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.research import ResearchRequiredError, resolve_mode
from privy_council.engine.resume import ResumeError
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import LocalOnlyConfigError, apply_local_only
from privy_council.models.council import CouncilPreset, MemoryMode
from privy_council.models.session import SessionState
from privy_council.storage.sessions import SessionStore, list_sessions

CHAT_DATA = "chat"


def chat_settings(settings: Settings) -> Settings:
    """Return the settings chat sessions run with: data under data/chat/."""
    paths = settings.paths.model_copy(update={"data": settings.data_dir / CHAT_DATA})
    return settings.model_copy(update={"paths": paths})


class ChatSessions:
    """Creates and resumes chat engines; one registry for the bot's lifetime."""

    def __init__(self, settings: Settings) -> None:
        """Load personas and councils and seat them on local models (raises ChatSetupError)."""
        loaded, _ = load_personas(settings.personas_dir)
        councils, _ = load_councils(settings.councils_dir, loaded)
        try:
            self.settings, self.personas = apply_local_only(
                chat_settings(settings), loaded, {c.moderator for c in councils.values()}
            )
        except LocalOnlyConfigError as exc:
            raise ChatSetupError(f"config error: {exc}") from exc
        self.councils = councils
        data = self.settings.data_dir
        self.registry = BackendRegistry(self.settings, Ledger.in_data_dir(data), local_only=True)

    def _preset(self, name: str) -> CouncilPreset:
        """Return a council with memory forced off."""
        preset = self.councils.get(name)
        if preset is None:
            raise ChatSetupError(f"Unknown council {name!r}: {', '.join(sorted(self.councils))}.")
        return preset.model_copy(update={"memory": MemoryMode.OFF})

    def _nicks(self) -> dict[str, str]:
        """Return persona display names by slug, for the transcripts."""
        return {slug: p.name for slug, p in self.personas.items()}

    def new(self, council: str, question: str) -> DebateEngine:
        """Create a chat session on the owner's question."""
        preset = self._preset(council)
        flag = True if self.settings.research.local.enabled else None
        try:
            mode = resolve_mode(self.settings, preset, flag, local_only=True)
        except ResearchRequiredError as exc:  # can't happen with flag None/True; be safe
            raise ChatSetupError(str(exc)) from exc
        store = SessionStore.create(
            self.settings.data_dir, preset.name, question, datetime.now(UTC), self._nicks()
        )
        return DebateEngine(
            settings=self.settings,
            council=preset,
            personas=self.personas,
            registry=self.registry,
            topic=question,
            store=store,
            research=mode,
        )

    def resume(self) -> DebateEngine | None:
        """Reopen the most recent unfinished chat session, or None."""
        for info in list_sessions(self.settings.data_dir):
            meta = info.meta
            if (
                meta is None
                or meta.state is SessionState.CLOSED
                or meta.council not in self.councils
            ):
                continue
            try:
                return DebateEngine.resume(
                    settings=self.settings,
                    council=self._preset(meta.council),
                    personas=self.personas,
                    registry=self.registry,
                    store=SessionStore.open(info.directory, self._nicks()),
                )
            except ResumeError:
                continue
        return None
