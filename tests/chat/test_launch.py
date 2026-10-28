"""Chat sessions: local-only, memory off, data/chat/; the token stays secret; doctor --chat."""

from __future__ import annotations

import asyncio
from pathlib import Path

import discord
import pytest

from privy_council.backends.base import BackendHealth
from privy_council.chat import discord_bot
from privy_council.chat.frontend import ChatSetupError
from privy_council.chat.launch import ChatSessions
from privy_council.config import Settings
from privy_council.doctor import check_chat
from privy_council.engine.research import ResearchMode
from privy_council.models.council import MemoryMode
from tests.conftest import REPO_ROOT

CHAT = {"token": "very-secret", "guild": 1, "channel": 2, "owner": 3}


def settings(tmp_path: Path, **extra: object) -> Settings:
    """The repo's personas and councils, data in tmp_path."""
    return Settings.model_validate(
        {
            "root": tmp_path,
            "paths": {
                "personas": str(REPO_ROOT / "personas"),
                "councils": str(REPO_ROOT / "councils"),
                "data": str(tmp_path / "data"),
            },
            **extra,
        }
    )


def test_chat_sessions_are_local_only_memory_off_and_apart(tmp_path: Path) -> None:
    sessions = ChatSessions(settings(tmp_path))
    engine = sessions.new("finance", "Should I max my PEA?")
    assert engine.meta.local_only and sessions.registry.local_only
    assert engine.council.memory is MemoryMode.OFF
    assert engine.store is not None
    assert engine.store.directory.parent == tmp_path / "data" / "chat" / "sessions"
    assert engine.research_mode is ResearchMode.OFF  # local search off: no research
    resumed = sessions.resume()
    assert resumed is not None and resumed.meta.id == engine.meta.id
    with pytest.raises(ChatSetupError, match="Unknown council"):
        sessions.new("nope", "Q")


def test_chat_sessions_search_when_local_search_is_on(tmp_path: Path) -> None:
    on = settings(tmp_path, research={"local": {"enabled": True}})
    engine = ChatSessions(on).new("career", "Should I change jobs?")
    assert engine.research_mode is ResearchMode.AUTO


def test_token_never_shows(tmp_path: Path) -> None:
    s = settings(tmp_path, chat={"discord": CHAT})
    assert s.chat.discord is not None
    assert "very-secret" not in repr(s) and "very-secret" not in str(s.model_dump())
    assert s.chat.discord.token.get_secret_value() == "very-secret"


async def test_doctor_chat_without_section(tmp_path: Path) -> None:
    [health] = await check_chat(settings(tmp_path))
    assert not health.ok and "docs/discord-setup.md" in health.problems[0]


async def test_doctor_chat_warns_on_a_readable_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_probe(config: object) -> BackendHealth:
        """No network in tests."""
        return BackendHealth(backend="discord", ok=True)

    monkeypatch.setattr(discord_bot, "probe", fake_probe)
    local = tmp_path / "config.local.toml"
    local.write_text("")
    local.chmod(0o644)
    [health] = await check_chat(settings(tmp_path, chat={"discord": CHAT}))
    assert health.ok and any("chmod 600" in w for w in health.warnings)
    local.chmod(0o600)
    [health] = await check_chat(settings(tmp_path, chat={"discord": CHAT}))
    assert not health.warnings


def test_bot_never_pings() -> None:
    none = discord_bot.NO_MENTIONS
    assert not (none.everyone or none.users or none.roles or none.replied_user)


def test_no_voice_warnings_at_start(caplog: pytest.LogCaptureFixture) -> None:
    discord.Client(intents=discord.Intents.none())
    assert "voice will NOT be supported" not in caplog.text


async def test_doctor_chat_flags_an_unknown_default_council(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_probe(config: object) -> BackendHealth:
        """No network in tests."""
        return BackendHealth(backend="discord", ok=True)

    monkeypatch.setattr(discord_bot, "probe", fake_probe)
    [health] = await check_chat(settings(tmp_path, chat={"discord": {**CHAT, "council": "nope"}}))
    assert not health.ok and "'nope' is not a council" in health.problems[0]


def test_gacha_council_loads(tmp_path: Path) -> None:
    sessions = ChatSessions(settings(tmp_path))
    gacha = sessions.councils["gacha"]
    assert gacha.members == ["meta", "saver", "fan", "contrarian"]
    assert gacha.protocol.require_research
    seats = {slug: sessions.personas[slug].meta.model for slug in gacha.members}
    assert len(set(seats.values())) == 4  # four model families, no echo
    assert all(sessions.personas[slug].meta.backend.value == "ollama" for slug in gacha.members)


async def test_the_bot_disconnects_once_offline_posts_are_sent() -> None:
    class Frontend:
        """Just the stop signal and the drain the watcher waits for."""

        def __init__(self) -> None:
            """Not stopped yet."""
            self.stopped = asyncio.Event()
            self.drained = False

        async def drain(self) -> None:
            """Record the drain."""
            self.drained = True

    class Bot:
        """The watcher's view of CouncilBot."""

        def __init__(self) -> None:
            """Open."""
            self.frontend = Frontend()
            self.closed_after_drain: bool | None = None

        async def close(self) -> None:
            """Record whether the posts were sent first."""
            self.closed_after_drain = self.frontend.drained

    bot = Bot()
    watch = asyncio.create_task(discord_bot.CouncilBot._close_when_offline(bot))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert not watch.done()
    bot.frontend.stopped.set()
    await asyncio.wait_for(watch, 1)
    assert bot.closed_after_drain is True
