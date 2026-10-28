"""`council chat discord`: serve debates to a private Discord channel (M16 part B)."""

from __future__ import annotations

import asyncio

import typer

from privy_council.chat.avatars import persona_avatars
from privy_council.chat.frontend import ChatFrontend, ChatSetupError, ChatTransport
from privy_council.chat.launch import ChatSessions
from privy_council.chat.store import ThreadStore
from privy_council.config import Settings
from privy_council.search.container import SearxngContainer, stopping
from privy_council.util.logs import setup_logging

chat_app = typer.Typer(help="Chat front ends (local-only, memory off).", no_args_is_help=True)

NOT_SET_UP = (
    "no [chat.discord] section in config.local.toml: see docs/discord-setup.md, "
    "then check with `council doctor --chat`"
)


@chat_app.command("discord")
def discord_command(ctx: typer.Context) -> None:
    """Run the council bot on this PC until /offline (or Ctrl+C); no Claude call is made."""
    from privy_council.chat.discord_bot import run_bot  # imports discord.py only when used

    settings: Settings = ctx.obj
    config = settings.chat.discord
    if config is None:
        typer.secho(NOT_SET_UP, fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    try:
        sessions = ChatSessions(settings)
    except ChatSetupError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if config.council not in sessions.councils:
        names = ", ".join(sorted(sessions.councils))
        typer.secho(
            f"[chat.discord] council = {config.council!r} is not a council: {names}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    data = sessions.settings.data_dir
    setup_logging(data, warnings_from=("discord",))
    threads = ThreadStore(data)

    def make_frontend(transport: ChatTransport) -> ChatFrontend:
        """Build the front end once the client exists."""
        return ChatFrontend(
            config, transport, threads, new_engine=sessions.new, resume_engine=sessions.resume
        )

    search = "on (SearXNG)" if settings.research.local.enabled else "off"
    typer.echo(f"Council bot starting: local-only, memory off, local search {search}.")
    typer.echo(f"Sessions in {data / 'sessions'}. Stop with /offline in Discord, or Ctrl+C.")
    councils = {name: c.description for name, c in sessions.councils.items()}
    avatar_for = persona_avatars(sessions.personas.values(), settings.root)
    # Chat sessions search by default: SearXNG starts with the bot, not at the first /ask.
    container = SearxngContainer(sessions.settings.research.local, settings.root)
    if line := asyncio.run(container.ensure_up()):
        typer.echo(line)
    on_exit = stopping(sessions.registry.aclose, container)
    run_bot(config, make_frontend, councils, on_exit, avatar_for)
    typer.echo("Council bot stopped.")
