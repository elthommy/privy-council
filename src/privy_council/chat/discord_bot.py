"""The thin discord.py adapter: messages in and out, slash commands, the doctor probe.

All decisions live in `frontend.py`; this module only translates. Every send uses
`AllowedMentions.none()` (rule 3: model text writing "@everyone" or `<@id>` must
not ping anyone). Personas post through one webhook with their name per message.
The council channel may be a text channel (one thread per question) or a forum
(one post per question).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Literal

import discord
from discord import app_commands

from privy_council.backends.base import BackendHealth
from privy_council.chat.frontend import ChatFrontend, ChatTransport, council_choices
from privy_council.chat.render import Post
from privy_council.chat.webhooks import WebhookPool
from privy_council.config import DiscordConfig

log = logging.getLogger(__name__)

# The bot never joins voice, so discord.py's "PyNaCl/davey is not installed" warnings
# at every client start are noise; these flags are the ones it clears after warning.
discord.VoiceClient.warn_nacl = False
discord.VoiceClient.warn_dave = False

NO_MENTIONS = discord.AllowedMentions.none()
WEBHOOK_NAME = "council"
TYPING_EVERY_S = 8  # Discord shows "is typing" for about 10 s per trigger
ARCHIVE_AFTER_MIN: Literal[1440] = 1440  # a day without messages
WRONG_PLACE = "Use the council's commands in its channel."
# What the bot needs in the council channel (docs/discord-setup.md, step 6).
NEEDED = (
    "view_channel",
    "send_messages",
    "send_messages_in_threads",
    "create_public_threads",
    "read_message_history",
    "manage_webhooks",
    "embed_links",
)

Channel = discord.TextChannel | discord.ForumChannel
MakeFrontend = Callable[[ChatTransport], ChatFrontend]
AvatarFor = Callable[[str], bytes | None]  # persona display name -> avatar PNG


class DiscordTransport:
    """`ChatTransport` over a connected client and the council channel."""

    def __init__(self, client: discord.Client, channel_id: int, avatar_for: AvatarFor) -> None:
        """Nothing is fetched until the first send; `avatar_for`: persona name -> PNG."""
        self.client = client
        self.channel_id = channel_id
        self._webhook: discord.Webhook | None = None
        self.pool: WebhookPool[discord.Webhook] = WebhookPool(_Hooks(self), avatar_for)
        self._typing: dict[int, asyncio.Task[None]] = {}

    async def _channel(self) -> Channel:
        """Return the council channel (a text or forum channel)."""
        channel = self.client.get_channel(self.channel_id)
        if channel is None:
            channel = await self.client.fetch_channel(self.channel_id)
        if not isinstance(channel, discord.TextChannel | discord.ForumChannel):
            raise TypeError("the council channel must be a text or forum channel")
        return channel

    async def _thread(self, thread_id: int) -> discord.Thread:
        """Return a thread of the council channel."""
        thread = self.client.get_channel(thread_id)
        if thread is None:
            thread = await self.client.fetch_channel(thread_id)
        if not isinstance(thread, discord.Thread):
            raise TypeError(f"{thread_id} is not a thread")
        return thread

    async def open_thread(self, title: str, opening: str) -> int:
        """Start a forum post, or a public thread with the opening as its first message."""
        channel = await self._channel()
        if isinstance(channel, discord.ForumChannel):
            created = await channel.create_thread(
                name=title,
                content=opening,
                auto_archive_duration=ARCHIVE_AFTER_MIN,
                allowed_mentions=NO_MENTIONS,
            )
            return int(created.thread.id)
        thread = await channel.create_thread(
            name=title,
            type=discord.ChannelType.public_thread,
            auto_archive_duration=ARCHIVE_AFTER_MIN,
        )
        await thread.send(opening, allowed_mentions=NO_MENTIONS)
        return thread.id

    async def _hook(self) -> discord.Webhook:
        """Return the bot's webhook in the council channel, creating it once."""
        if self._webhook is None:
            channel = await self._channel()
            me = self.client.user
            hooks = [h for h in await channel.webhooks() if h.name == WEBHOOK_NAME]
            mine = [
                h for h in hooks if me is not None and h.user is not None and h.user.id == me.id
            ]
            self._webhook = mine[0] if mine else await channel.create_webhook(name=WEBHOOK_NAME)
        return self._webhook

    async def post(self, thread: int, post: Post) -> None:
        """Send as a persona (webhook) or as the bot; links stay unfurled-free."""
        if post.persona is None:
            target = await self._thread(thread)
            await target.send(post.text, allowed_mentions=NO_MENTIONS, suppress_embeds=True)
            return
        hook = await self.pool.hook_for(post.persona) or await self._hook()
        await hook.send(
            post.text,
            username=post.persona[:80],
            thread=discord.Object(id=thread),
            allowed_mentions=NO_MENTIONS,
            suppress_embeds=True,
        )

    async def typing(self, thread: int, on: bool) -> None:
        """Keep "is typing" up while a slow local turn runs, or stop it."""
        task = self._typing.pop(thread, None)
        if task is not None:
            task.cancel()
        if on:
            target = await self._thread(thread)
            self._typing[thread] = asyncio.create_task(_keep_typing(target))


class _Hooks:
    """`HookChannel` over the council channel; only lists and changes its webhooks."""

    def __init__(self, transport: DiscordTransport) -> None:
        """Use the transport's channel lookup."""
        self.transport = transport

    async def list_hooks(self) -> list[discord.Webhook]:
        """Return the channel's webhooks."""
        return list(await (await self.transport._channel()).webhooks())

    async def create_hook(self, name: str, avatar: bytes | None) -> discord.Webhook:
        """Create a webhook with its avatar."""
        channel = await self.transport._channel()
        if avatar is None:
            return await channel.create_webhook(name=name)
        return await channel.create_webhook(name=name, avatar=avatar)

    async def set_avatar(self, hook: discord.Webhook, avatar: bytes) -> None:
        """Replace a webhook's avatar."""
        await hook.edit(avatar=avatar)

    async def delete_hook(self, hook: discord.Webhook) -> None:
        """Delete a webhook (the pool only passes its own `council-*` ones)."""
        await hook.delete(reason="council: making room for another persona")


async def _keep_typing(thread: discord.Thread) -> None:
    """Re-trigger the typing indicator until cancelled (a turn can take minutes)."""
    with contextlib.suppress(discord.HTTPException):
        while True:
            await thread.typing()
            await asyncio.sleep(TYPING_EVERY_S)


class CouncilBot(discord.Client):
    """The bot: passes listed people's lines and commands to the front end."""

    def __init__(
        self,
        config: DiscordConfig,
        make_frontend: MakeFrontend,
        councils: Mapping[str, str],
        avatar_for: AvatarFor = lambda name: None,
    ) -> None:
        """Set up intents (message content, to read people's lines) and the commands.

        `councils`: name -> description, offered as the `/ask` dropdown.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, allowed_mentions=NO_MENTIONS)
        self.config = config
        self.frontend = make_frontend(DiscordTransport(self, config.channel, avatar_for))
        self.tree = app_commands.CommandTree(self)
        register_commands(self.tree, self, council_choices(councils))

    async def setup_hook(self) -> None:
        """Sync the commands to the one server; watch for /offline."""
        await self.tree.sync(guild=discord.Object(id=self.config.guild))
        self._watch = asyncio.create_task(self._close_when_offline())

    async def on_ready(self) -> None:
        """Start sending."""
        await self.frontend.start()
        log.info("chat bot online as %s", self.user)

    def in_council(self, channel: object) -> int | None:
        """Return the thread ID if `channel` is a thread of the council channel."""
        if isinstance(channel, discord.Thread) and channel.parent_id == self.config.channel:
            return channel.id
        return None

    async def on_message(self, message: discord.Message) -> None:
        """Pass people's lines in council threads on; ignore bots, webhooks, DMs, elsewhere."""
        if message.author.bot or message.webhook_id is not None or message.guild is None:
            return
        if message.guild.id != self.config.guild:
            return
        thread = self.in_council(message.channel)
        if thread is not None:
            self.frontend.on_message(
                message.author.id, message.author.name, thread, message.content
            )

    async def _close_when_offline(self) -> None:
        """After /offline: send what's queued, then disconnect."""
        await self.frontend.stopped.wait()
        await self.frontend.drain()
        log.info("chat bot disconnecting")
        await self.close()


def register_commands(
    tree: app_commands.CommandTree[CouncilBot],
    bot: CouncilBot,
    choices: list[tuple[str, str]],
) -> None:
    """Declare the slash commands: the front end's table, nothing else (rule 1)."""
    guild = discord.Object(id=bot.config.guild)
    council_choice = [app_commands.Choice(name=label, value=value) for label, value in choices]

    async def reply(
        interaction: discord.Interaction,
        name: str,
        arg: str = "",
        council: str = "",
        username: str = "",
    ) -> None:
        """Answer privately; commands outside the council channel are refused."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel = interaction.channel
        here = interaction.channel_id == bot.config.channel or bot.in_council(channel) is not None
        text = (
            await bot.frontend.command(interaction.user.id, name, arg, council, username)
            if here
            else WRONG_PLACE
        )
        await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @tree.command(name="ask", description="Start a debate on your question (owner).", guild=guild)
    @app_commands.describe(question="Your question", council="Default: the bot's default council")
    @app_commands.choices(council=council_choice)
    async def ask(
        interaction: discord.Interaction,
        question: str,
        council: app_commands.Choice[str] | None = None,
    ) -> None:
        """/ask."""
        await reply(interaction, "ask", question, council.value if council else "")

    @tree.command(
        name="run",
        description="Run rounds (owner, or handed over); default: the rest.",
        guild=guild,
    )
    async def run(interaction: discord.Interaction, rounds: int | None = None) -> None:
        """/run."""
        await reply(interaction, "run", str(rounds) if rounds else "")

    @tree.command(
        name="handover",
        description="Let a friend /run, /pause, /vote, /close and /offline this debate (owner).",
        guild=guild,
    )
    @app_commands.describe(friend="A friend on the council's list")
    async def handover(interaction: discord.Interaction, friend: discord.User) -> None:
        """/handover: by user ID; the username only names them in the debate."""
        await reply(interaction, "handover", str(friend.id), username=friend.name)

    simple: dict[str, str] = {
        "pause": "Pause at the next turn boundary (owner, or handed over).",
        "vote": "Ask every member for a final position (owner, or handed over).",
        "close": "End the debate with the synthesis (owner, or handed over).",
        "offline": "Stop after the current turn, save, disconnect (owner, or handed over).",
        "resume": "Reopen the latest unfinished debate (owner).",
        "takeback": "End the handover: only you run the debate again (owner).",
        "status": "Where the debate is, and who it waits for.",
        "flaws": "The errors caught so far.",
    }
    for name, description in simple.items():
        tree.command(name=name, description=description, guild=guild)(_simple(reply, name))


def _simple(
    reply: Callable[[discord.Interaction, str], Awaitable[None]], name: str
) -> Callable[[discord.Interaction], Coroutine[Any, Any, None]]:
    """Return the callback of an argument-free command."""

    async def callback(interaction: discord.Interaction) -> None:
        """Run the command."""
        await reply(interaction, name)

    callback.__name__ = name
    return callback


def run_bot(
    config: DiscordConfig,
    make_frontend: MakeFrontend,
    councils: Mapping[str, str],
    on_exit: Callable[[], Awaitable[None]],
    avatar_for: AvatarFor = lambda name: None,
) -> None:
    """Connect and serve until /offline or Ctrl+C."""

    async def main() -> None:
        """Run the client, then close the backends."""
        try:
            async with CouncilBot(config, make_frontend, councils, avatar_for) as bot:
                await bot.start(config.token.get_secret_value())
        finally:
            await on_exit()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
    log.info("chat bot stopped")


async def probe(config: DiscordConfig) -> BackendHealth:
    """`doctor --chat`: log in over HTTP (no gateway) and check the channel and permissions."""
    health = BackendHealth(backend="discord", ok=True)
    client = discord.Client(intents=discord.Intents.none())
    try:
        await client.login(config.token.get_secret_value())
        await _probe_checks(client, config, health)
    except discord.LoginFailure:
        health.problems.append("token rejected: Bot → Reset Token, then update config.local.toml")
    except discord.NotFound:
        health.problems.append("server or channel not found: check the IDs in [chat.discord]")
    except discord.Forbidden:
        health.problems.append("the bot can't see the channel: add it to the private channel")
    except discord.HTTPException as exc:
        health.problems.append(f"Discord error {exc.status}")
    finally:
        await client.close()
    health.ok = not health.problems
    return health


async def _probe_checks(
    client: discord.Client, config: DiscordConfig, health: BackendHealth
) -> None:
    """Check the intent, the channel's kind and server, and the bot's permissions."""
    app = await client.application_info()
    flags = app.flags
    if not (flags.gateway_message_content or flags.gateway_message_content_limited):
        health.problems.append(
            "Message Content intent is off: Bot tab → Privileged Gateway Intents"
        )
    channel = await client.fetch_channel(config.channel)
    if not isinstance(channel, discord.TextChannel | discord.ForumChannel):
        health.problems.append("the council channel must be a text or forum channel")
        return
    if channel.guild.id != config.guild:
        health.problems.append("the channel is not in the configured server")
        return
    guild = await client.fetch_guild(config.guild)
    assert client.user is not None
    member = await guild.fetch_member(client.user.id)
    kind = "forum" if isinstance(channel, discord.ForumChannel) else "text channel"
    health.details.append(f"bot {client.user.name} in {guild.name}, #{channel.name} ({kind})")
    try:
        # A channel fetched over HTTP points at a server without roles, which would
        # drop the bot's role permissions; use the fully fetched server instead.
        channel.guild = guild
        perms = channel.permissions_for(member)
    except Exception:  # partial objects from HTTP fetches; not worth failing doctor
        health.warnings.append("couldn't compute the bot's permissions; the live run will tell")
        return
    missing = [p for p in NEEDED if not getattr(perms, p)]
    if missing:
        health.problems.append("missing permissions: " + ", ".join(missing))
