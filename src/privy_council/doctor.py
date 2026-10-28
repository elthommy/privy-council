"""`council doctor`: check config, the claude CLI and Ollama without spending quota."""

from __future__ import annotations

import asyncio

from privy_council.backends.base import BackendHealth
from privy_council.backends.claude_cli import ClaudeCliBackend
from privy_council.backends.ollama import OllamaBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import LOCAL_CONFIG_FILE, Settings
from privy_council.engine.budget import Ledger, quota_warnings
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import LocalOnlyConfigError, required_models
from privy_council.search import SearxngProvider


def check_config(settings: Settings, local_only: bool = False) -> tuple[BackendHealth, set[str]]:
    """Validate personas and councils; also return the Ollama models they need.

    In local-only mode, those are the seat map's models, and the map is checked too.
    """
    personas, p_errors = load_personas(settings.personas_dir)
    councils, c_errors = load_councils(settings.councils_dir, personas)
    health = BackendHealth(backend="config", ok=not (p_errors or c_errors))
    health.details.append(f"{len(personas)} personas, {len(councils)} councils")
    health.problems += [str(e) for e in p_errors + c_errors]
    if not local_only:
        return health, BackendRegistry(settings).required_ollama_models(personas.values())
    moderators = {c.moderator for c in councils.values()}
    try:
        models = required_models(settings, personas, moderators)
    except LocalOnlyConfigError as exc:
        health.ok = False
        health.problems.append(str(exc))
        return health, set()
    health.details.append(f"local-only seat map: {len(models)} model(s)")
    return health, models


def check_budget(settings: Settings) -> BackendHealth:
    """Summarize today's Claude usage against the daily soft cap."""
    report = Ledger.in_data_dir(settings.data_dir).report()
    cap = settings.claude.daily_soft_cap_calls
    health = BackendHealth(backend="budget", ok=True)
    health.details.append(f"{report.today.calls}/{cap or '∞'} Claude calls today")
    if cap and report.today.calls >= cap:
        health.warnings.append("daily soft cap reached")
    if report.quota is not None and report.quota_ts is not None:
        q = report.quota
        health.details.append(
            f"Pro quota 5h {q.five_hour or 0:.0%} / 7d {q.seven_day or 0:.0%} "
            f"(as of {report.quota_ts.astimezone():%a %H:%M})"
        )
        health.warnings += quota_warnings(q, settings.claude.quota_warn_utilization)
    return health


async def check_chat(settings: Settings) -> list[BackendHealth]:
    """`doctor --chat`: the [chat.discord] section, config.local.toml's mode, the bot."""
    config = settings.chat.discord
    if config is None:
        missing = "no [chat.discord] section in config.local.toml (docs/discord-setup.md)"
        return [BackendHealth(backend="discord", ok=False, problems=[missing])]
    from privy_council.chat.discord_bot import probe  # discord.py only when asked

    health = await probe(config)
    local = settings.root / LOCAL_CONFIG_FILE
    if local.is_file() and local.stat().st_mode & 0o077:
        health.warnings.append(f"{LOCAL_CONFIG_FILE} holds the bot token: chmod 600 it")
    people = len(config.allowed_users)
    health.details.append(f"owner + {people} friend(s) allowed, default council {config.council}")
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    if config.council not in councils:
        names = ", ".join(sorted(councils))
        health.problems.append(
            f"[chat.discord] council = {config.council!r} is not a council: {names}"
        )
        health.ok = False
    return [health]


async def check_search(settings: Settings) -> BackendHealth:
    """Check local web search (SearXNG) when it's enabled; "off" is not a problem."""
    if not settings.research.local.enabled:
        return BackendHealth(backend="local search", ok=True, details=["off"])
    health = await SearxngProvider(settings.research.local).health()
    use = {"claude": "local-only sessions", "local": "every session", "auto": "fallback"}
    provider = settings.research.provider
    health.details.append(f"used by: {use[provider]} ([research] provider = {provider!r})")
    return health


async def run_doctor(settings: Settings, local_only: bool = False) -> list[BackendHealth]:
    """Run every check; backend checks run concurrently. Local-only skips Claude."""
    config_health, models = check_config(settings, local_only)
    ollama = OllamaBackend(settings.ollama)
    try:
        if local_only:
            ollama_health, search_health = await asyncio.gather(
                ollama.health(models), check_search(settings)
            )
            return [config_health, ollama_health, search_health]
        claude_health, ollama_health, search_health = await asyncio.gather(
            ClaudeCliBackend(settings.claude).health(),
            ollama.health(models),
            check_search(settings),
        )
    finally:
        await ollama.aclose()
    return [config_health, claude_health, ollama_health, search_health, check_budget(settings)]


def render(results: list[BackendHealth]) -> list[tuple[str, str]]:
    """Render results as (style, line) pairs; style is ok, warn or fail."""
    lines = []
    for h in results:
        version = f" {h.version}" if h.version else ""
        summary = ", ".join(h.details)
        lines.append(
            ("ok" if h.ok else "fail", f"{'✓' if h.ok else '✗'} {h.backend}{version}: {summary}")
        )
        lines += [("warn", f"    ! {w}") for w in h.warnings]
        lines += [("fail", f"    ✗ {p}") for p in h.problems]
    return lines
