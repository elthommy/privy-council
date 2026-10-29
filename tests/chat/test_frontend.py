"""The chat front end with a fake transport and local-only fake engines (0 Claude calls)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import discord
import pytest
from pydantic import SecretStr

from privy_council.chat.discord_bot import CouncilBot
from privy_council.chat.frontend import (
    HANDED,
    LINE_LIMIT,
    NO_DEBATE,
    NOT_ALLOWED,
    NOT_HANDED,
    NOT_LISTED,
    OFFLINE,
    ONE_AT_A_TIME,
    OPEN_COMMANDS,
    OWNER_COMMANDS,
    TAKEN_BACK,
    TYPED_COMMAND,
    ChatFrontend,
    ChatSetupError,
)
from privy_council.chat.render import GENERIC_ERROR, Post
from privy_council.chat.store import ChatThread, ThreadStore
from privy_council.config import DiscordConfig
from privy_council.engine.protocol import DebateEngine
from privy_council.models.council import ProtocolSettings
from privy_council.models.persona import Persona
from privy_council.models.session import ErrorEvent, Event, MessageEvent, Phase
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas

OWNER, ANA, BEN, STRANGER = 1, 2, 3, 99
THREAD = 500


class FakeTransport:
    """Records threads, posts and typing."""

    def __init__(self) -> None:
        """Nothing sent yet."""
        self.threads: list[tuple[str, str]] = []
        self.posts: list[tuple[int, Post]] = []
        self.typing_events: list[tuple[int, bool]] = []

    async def open_thread(self, title: str, opening: str) -> int:
        """Return a fixed thread ID."""
        self.threads.append((title, opening))
        return THREAD

    async def post(self, thread: int, post: Post) -> None:
        """Record a post."""
        self.posts.append((thread, post))

    async def typing(self, thread: int, on: bool) -> None:
        """Record typing."""
        self.typing_events.append((thread, on))

    def texts(self, persona: str | None = None) -> list[str]:
        """Return the posted texts, by one persona or by the bot (None)."""
        return [p.text for _, p in self.posts if p.persona == persona]


def config(**extra: Any) -> DiscordConfig:
    """A config with the owner, Ana and Ben allowed."""
    return DiscordConfig(
        token=SecretStr("secret-token"),
        guild=10,
        channel=20,
        owner=OWNER,
        allowed_users=[ANA, BEN],
        **extra,
    )


def asking(question: str) -> dict[str, Any]:
    """A moderator decision that puts one question to a person."""
    return {**decision([("rigor", "go")]), "questions_for_user": [question]}


class Chat:
    """A front end over one local-only Harness engine."""

    def __init__(
        self,
        tmp_path: Path,
        personas: dict[str, Persona],
        decisions: list[dict[str, Any] | str] | None = None,
        **cfg: Any,
    ) -> None:
        """Build the engine lazily on /ask, like the real factory."""
        self.harness = Harness(
            tmp_path,
            personas,
            standard_responder(decisions or [decision([("rigor", "x")])] * 6),
            protocol=ProtocolSettings(max_rounds=2),
            local_only=True,
        )
        self.transport = FakeTransport()
        self.threads = ThreadStore(tmp_path / "chat")
        self.created: list[tuple[str, str]] = []
        self.resumable: DebateEngine | None = None
        self.fe = ChatFrontend(
            config(**cfg),
            self.transport,
            self.threads,
            new_engine=self.new_engine,
            resume_engine=lambda: self.resumable,
        )

    def new_engine(self, council: str, question: str) -> DebateEngine:
        """Return the Harness engine (or refuse an unknown council)."""
        if council != "finance":
            raise ChatSetupError(f"Unknown council {council!r}.")
        self.created.append((council, question))
        return self.harness.engine

    async def ask(self) -> str:
        """The owner starts a debate and its opening runs."""
        await self.fe.start()
        reply = await self.fe.command(OWNER, "ask", "Should I max my PEA?")
        await self.fe.drain()
        return reply


async def test_owner_asks_and_the_opening_posts_as_chair(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    reply = await chat.ask()
    assert reply.startswith(f"Started in <#{THREAD}>")
    title, opening = chat.transport.threads[0]
    assert title == "Should I max my PEA?" and "**User asks:** Should I max my PEA?" in opening
    assert chat.transport.texts("Chair")[0].startswith("**Chair** · *moderator*\nRestating")
    assert (THREAD, True) in chat.transport.typing_events
    assert chat.harness.claude.requests == []  # local-only: nothing reached Claude
    assert chat.harness.engine.phase is Phase.ROUND
    assert chat.threads.get(chat.harness.engine.meta.id) == ChatThread(thread=THREAD)
    assert await chat.fe.command(OWNER, "ask", "Another one?") == ONE_AT_A_TIME


async def test_permissions_and_the_command_table(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.fe.start()
    assert await chat.fe.command(STRANGER, "status") == NOT_LISTED
    assert await chat.fe.command(ANA, "status") == NO_DEBATE
    for name in ("ask", "resume", "handover", "takeback"):
        assert await chat.fe.command(ANA, name, "x") == NOT_ALLOWED
    for name in ("run", "close", "offline", "vote", "pause"):
        assert await chat.fe.command(ANA, name, "x") == NOT_HANDED
    for name in ("attach", "export", "research", "keep", "forget", "summon", "kick", "lang", "log"):
        assert (await chat.fe.command(OWNER, name, "x")).startswith("Unknown command")
    assert await chat.fe.command(OWNER, "ask", "Q?", "nope") == "Unknown council 'nope'."
    assert chat.created == []


async def test_people_speak_by_id_with_frozen_usernames(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas, names={str(BEN): "Ben"})
    await chat.ask()
    engine = chat.harness.engine
    chat.fe.on_message(ANA, "ana.martin", THREAD, "I can put in 9,000 EUR")
    chat.fe.on_message(ANA, "ana.renamed", THREAD, "and I drive a lot")
    chat.fe.on_message(BEN, "xx_ben_xx", THREAD, "me too")
    chat.fe.on_message(OWNER, "thomas", THREAD, "my own line")
    chat.fe.on_message(STRANGER, "eve", THREAD, "let me in")
    chat.fe.on_message(ANA, "ana.martin", 777, "elsewhere")  # not the debate's thread
    people = [(m.author, m.text) for m in engine.messages if m.speaker == "user"]
    assert people == [
        ("ana-martin", "I can put in 9,000 EUR"),
        ("ana-martin", "and I drive a lot"),
        ("ben", "me too"),
        (None, "my own line"),
    ]
    entry = chat.threads.get(engine.meta.id)
    assert entry is not None and entry.names == {ANA: "ana-martin", BEN: "ben"}
    assert not [t for t in chat.transport.texts("Chair") if "9,000" in t]  # not echoed


async def test_line_limit_per_round(tmp_path: Path, personas: dict[str, Persona]) -> None:
    chat = Chat(tmp_path, personas, max_lines_per_round=2, max_message_chars=5)
    await chat.ask()
    for _ in range(4):
        chat.fe.on_message(ANA, "ana", THREAD, "a long line")
    await chat.fe.drain()
    lines = [m.text for m in chat.harness.engine.messages if m.author == "ana"]
    assert lines == ["a lon", "a lon"]  # capped in size and count
    assert chat.transport.texts().count(LINE_LIMIT.format(name="ana", count=2)) == 1


async def test_run_waits_for_the_guest_asked_then_goes_on(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        asking("@ana how much can you put in?"),  # round 1 update
        decision([("rigor", "x")]),  # round 2 update
    ]
    chat = Chat(tmp_path, personas, decisions, question_timeout_s=30)
    await chat.ask()
    chat.fe.on_message(ANA, "ana", THREAD, "Hello")  # Ana is a participant now
    assert await chat.fe.command(OWNER, "run") == "Running 2 round(s)."

    async def waiting() -> None:
        """Return once the front end waits for Ana (after round 1's update)."""
        while ANA not in chat.fe._waiting:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(waiting(), 5)
    assert chat.harness.engine.meta.runtime.round == 1
    assert "/run paused" not in " ".join(chat.transport.texts())
    chat.fe.on_message(ANA, "ana", THREAD, "About 9,000 EUR")
    await chat.fe.drain()
    engine = chat.harness.engine
    assert engine.meta.runtime.round == 2
    assert any("Waiting up to 1 min for @ana" in t for t in chat.transport.texts())
    assert any(
        t.startswith("**Chair** · *question*\n@ana how much") for t in chat.transport.texts("Chair")
    )


async def test_unanswered_question_times_out(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [
        asking("@ana how much can you put in?"),
        decision([("rigor", "x")]),
    ]
    chat = Chat(tmp_path, personas, decisions, question_timeout_s=0.05)
    await chat.ask()
    chat.fe.on_message(ANA, "ana", THREAD, "Hello")
    await chat.fe.command(OWNER, "run")
    await chat.fe.drain()
    assert any("No answer from @ana" in t for t in chat.transport.texts())
    assert chat.harness.engine.meta.runtime.round == 2  # the debate went on


async def test_errors_are_generic_in_the_channel(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    chat.harness.engine._emit(ErrorEvent(text="boom at /home/thomas/secret/file"))
    await chat.fe.drain()
    assert GENERIC_ERROR in chat.transport.texts()
    assert not any("/home/thomas" in t for _, p in chat.transport.posts for t in [p.text])


async def test_offline_then_resume(tmp_path: Path, personas: dict[str, Persona]) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    assert await chat.fe.command(OWNER, "offline") == "Going offline."
    await chat.fe.drain()
    assert chat.fe.stopped.is_set() and OFFLINE in chat.transport.texts()

    # A new bot process: the front end finds the session's thread and names again.
    engine = chat.harness.engine
    chat.threads.put(engine.meta.id, ChatThread(thread=THREAD, names={ANA: "ana"}))
    again = Chat(tmp_path / "again", personas)
    again.threads = chat.threads
    again.fe.threads = chat.threads
    again.resumable = engine
    await again.fe.start()
    assert await again.fe.command(OWNER, "resume") == f"Resumed in <#{THREAD}>."
    again.fe.on_message(ANA, "ana.other", THREAD, "back")
    assert engine.messages[-1].author == "ana"  # the name survived


async def test_offline_after_the_debate_closed(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    assert await chat.fe.command(OWNER, "close") == "Closing: the synthesis comes next."
    await chat.fe.drain()
    assert chat.harness.engine.phase is Phase.DONE
    assert await chat.fe.command(OWNER, "offline") == "Going offline."
    await chat.fe.drain()
    assert chat.fe.stopped.is_set() and chat.transport.texts()[-1] == OFFLINE


async def test_offline_mid_run_stops_at_the_next_turn(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    assert await chat.fe.command(OWNER, "run") == "Running 2 round(s)."
    assert await chat.fe.command(OWNER, "offline") == "Going offline after the current turn."
    await asyncio.wait_for(chat.fe.drain(), 5)
    assert chat.fe.stopped.is_set() and chat.transport.texts()[-1] == OFFLINE
    assert chat.harness.engine.phase is not Phase.DONE  # paused, resumable


async def test_offline_during_a_step_that_asks_never_waits_for_answers(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [asking("@ana how much?"), decision([("rigor", "x")])]
    chat = Chat(tmp_path, personas, decisions, question_timeout_s=30)
    await chat.ask()
    chat.fe.on_message(ANA, "ana", THREAD, "Hello")

    def offline_on_the_question(event: Event) -> None:
        """/offline while the Chair's step runs: right as it posts its question."""
        if isinstance(event, MessageEvent) and event.message.to_user:
            chat.fe._request_offline()

    chat.harness.engine.subscribe(offline_on_the_question)
    await chat.fe.command(OWNER, "run")
    await asyncio.wait_for(chat.fe.drain(), 5)  # not the 30 s question wait
    assert chat.fe.stopped.is_set() and chat.transport.texts()[-1] == OFFLINE
    assert not any(t.startswith("Waiting up to") for t in chat.transport.texts())


async def test_offline_while_waiting_for_an_answer(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [asking("@ana how much?"), decision([("rigor", "x")])]
    chat = Chat(tmp_path, personas, decisions, question_timeout_s=30)
    await chat.ask()
    chat.fe.on_message(ANA, "ana", THREAD, "Hello")
    await chat.fe.command(OWNER, "run")
    while ANA not in chat.fe._waiting:
        await asyncio.sleep(0.01)
    assert await chat.fe.command(OWNER, "offline") == "Going offline after the current turn."
    await asyncio.wait_for(chat.fe.drain(), 5)
    assert chat.fe.stopped.is_set() and chat.transport.texts()[-1] == OFFLINE
    assert chat.harness.engine.meta.runtime.round == 1


async def test_a_command_typed_as_text_never_reaches_the_debate(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    before = len(chat.harness.engine.messages)
    chat.fe.on_message(OWNER, "owner", THREAD, "/offline")
    chat.fe.on_message(ANA, "ana", THREAD, "/status please")
    chat.fe.on_message(ANA, "ana", THREAD, "/etc/passwd is not a command")
    await chat.fe.drain()
    assert [m.text for m in chat.harness.engine.messages[before:]] == [
        "/etc/passwd is not a command"
    ]
    hints = [t for t in chat.transport.texts() if "plain message" in t]
    assert hints == [TYPED_COMMAND.format(command="/offline")]  # the owner's only
    assert not chat.fe.stopped.is_set()


async def test_discord_mentions_reach_the_council_as_names(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    chat.fe.on_message(ANA, "ana", THREAD, "Hello")
    chat.fe.on_message(OWNER, "owner", THREAD, f"<@{ANA}> and <@!{OWNER}>, ask <@{BEN}> in <#77>")
    assert chat.harness.engine.messages[-1].text == "@ana and @User, ask @someone in @someone"


async def test_commands_are_logged_without_their_argument(
    tmp_path: Path, personas: dict[str, Persona], caplog: pytest.LogCaptureFixture
) -> None:
    chat = Chat(tmp_path, personas)
    with caplog.at_level(logging.INFO, logger="privy_council.chat"):
        await chat.ask()
        await chat.fe.command(OWNER, "offline")
        await chat.fe.command(STRANGER, "run")
    logged = caplog.text
    assert f"chat command /offline from {OWNER} (owner)" in logged
    assert "chat command /offline: Going offline." in logged
    assert f"chat command /run from {STRANGER} (not listed)" in logged
    assert "Should I max my PEA?" not in logged  # the question is personal


async def test_resume_without_a_session(tmp_path: Path, personas: dict[str, Persona]) -> None:
    chat = Chat(tmp_path, personas)
    assert await chat.fe.command(OWNER, "resume") == "No unfinished chat session to resume."


@pytest.mark.parametrize("name", ["status", "flaws"])
async def test_friends_can_read_status_and_flaws(
    tmp_path: Path, personas: dict[str, Persona], name: str
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    reply = await chat.fe.command(ANA, name)
    assert reply and reply != NOT_ALLOWED


async def test_handover_lets_one_friend_run_and_close_the_debate(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    assert await chat.fe.command(ANA, "run") == NOT_HANDED
    reply = await chat.fe.command(OWNER, "handover", str(ANA), username="ana.m")
    assert reply.startswith("@ana-m can now run")
    await chat.fe.drain()
    assert chat.transport.texts()[-1] == HANDED.format(name="ana-m")
    await chat.fe.command(OWNER, "handover", str(ANA), username="ana.m")
    await chat.fe.drain()
    assert chat.transport.texts().count(HANDED.format(name="ana-m")) == 1  # announced once
    assert "Handed over to @ana-m." in await chat.fe.command(BEN, "status")

    for name in ("ask", "resume", "handover", "takeback"):
        assert await chat.fe.command(ANA, name, str(BEN)) == NOT_ALLOWED
    assert await chat.fe.command(BEN, "run") == NOT_HANDED
    assert await chat.fe.command(ANA, "run", "1") == "Running 1 round(s)."
    await chat.fe.drain()
    assert await chat.fe.command(ANA, "pause") == "Pausing at the next turn boundary."
    assert await chat.fe.command(ANA, "close") == "Closing: the synthesis comes next."
    await chat.fe.drain()
    assert chat.harness.engine.phase is Phase.DONE
    # The handover ended with its debate.
    assert await chat.fe.command(ANA, "run") == NOT_HANDED
    assert "Handed over" not in await chat.fe.command(ANA, "status")
    assert await chat.fe.command(OWNER, "handover", str(ANA)) == NO_DEBATE


async def test_handover_refuses_anyone_off_the_list(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.fe.start()
    assert await chat.fe.command(OWNER, "handover", str(ANA)) == NO_DEBATE
    await chat.ask()
    refused = "Only a friend on the council's list can take the floor."
    for arg in (str(STRANGER), "", "ana", "-2"):
        assert await chat.fe.command(OWNER, "handover", arg) == refused
    await chat.fe.drain()
    assert await chat.fe.command(OWNER, "handover", str(OWNER)) == "You already hold the floor."
    assert await chat.fe.command(STRANGER, "handover", str(STRANGER)) == NOT_LISTED
    assert await chat.fe.command(STRANGER, "run") == NOT_LISTED
    assert HANDED.split("@")[0] not in "".join(chat.transport.texts())


async def test_takeback_and_a_new_debate_end_the_handover(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    assert await chat.fe.command(OWNER, "takeback") == "Nobody else holds the floor."
    await chat.fe.command(OWNER, "handover", str(ANA), username="ana")
    await chat.fe.command(OWNER, "handover", str(BEN), username="ben")
    assert await chat.fe.command(OWNER, "takeback") == "You hold the floor alone again."
    await chat.fe.drain()
    assert chat.transport.texts()[-1] == TAKEN_BACK
    assert await chat.fe.command(BEN, "vote") == NOT_HANDED

    await chat.fe.command(OWNER, "handover", str(ANA), username="ana")
    chat.fe._attach(chat.harness.engine, THREAD, None)  # what /ask and /resume do
    assert await chat.fe.command(ANA, "run") == NOT_HANDED


async def test_a_friend_handed_over_can_take_the_bot_offline(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    chat = Chat(tmp_path, personas)
    await chat.ask()
    await chat.fe.command(OWNER, "handover", str(ANA), username="ana")
    chat.fe.on_message(ANA, "ana", THREAD, "/run")
    await chat.fe.drain()
    assert chat.transport.texts()[-1] == TYPED_COMMAND.format(command="/run")
    assert await chat.fe.command(ANA, "offline") == "Going offline."
    await chat.fe.drain()
    assert chat.fe.stopped.is_set() and chat.transport.texts()[-1] == OFFLINE


async def test_the_bot_declares_exactly_the_front_ends_commands() -> None:
    bot = CouncilBot(config(), lambda transport: None, {"finance": "Money"})  # type: ignore[arg-type,return-value]
    commands = bot.tree.get_commands(guild=discord.Object(id=10))
    assert {c.name for c in commands} == OWNER_COMMANDS | OPEN_COMMANDS
    payload = {c.name: c.to_dict(bot.tree) for c in commands}
    assert all(len(p["description"]) <= 100 for p in payload.values())
    (friend,) = payload["handover"]["options"]
    assert friend["type"] == discord.AppCommandOptionType.user.value and friend["required"]
    await bot.close()
