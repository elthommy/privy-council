"""Typer CLI entry point (`council ...`)."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from privy_council import persona_draft
from privy_council.backends.base import BackendError, CompletionRequest
from privy_council.backends.claude_cli import api_keys_in_env
from privy_council.backends.registry import BackendRegistry
from privy_council.chat.cli import chat_app
from privy_council.compare.cli import compare_app
from privy_council.config import ConfigError, Settings, load_settings
from privy_council.doctor import check_chat, render, run_doctor
from privy_council.engine.budget import Ledger, QuotaSnapshot
from privy_council.engine.dossier import ALLOWED_TEXT, DossierError, read_attachable
from privy_council.engine.protocol import DebateEngine
from privy_council.engine.research import ResearchMode, ResearchRequiredError, resolve_mode
from privy_council.engine.resume import ResumeError
from privy_council.export import EXPORTS_DIR, ExportError, ExportFormat, export_session
from privy_council.loaders.councils import load_councils
from privy_council.loaders.errors import LoadError
from privy_council.loaders.personas import load_personas
from privy_council.local_only import LocalOnlyConfigError, apply_local_only
from privy_council.memory.store import SLUG_PATTERN, MemoryStore, size_bytes
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona, Role
from privy_council.models.session import Phase, SessionMeta
from privy_council.repl import run_repl
from privy_council.search.container import SearxngContainer, StartingSearch, stopping
from privy_council.storage.sessions import (
    SYNTHESIS_MD,
    SessionInfo,
    SessionLookupError,
    SessionStore,
    find_session,
    list_sessions,
    remove_session,
)
from privy_council.tui.app import CouncilApp
from privy_council.util.language import LANGUAGE_NAMES, Language, parse_language
from privy_council.util.logs import setup_logging

app = typer.Typer(help="An IRC-style council of LLM personas.", no_args_is_help=True)
personas_app = typer.Typer(help="Inspect and validate persona files.", no_args_is_help=True)
councils_app = typer.Typer(help="Inspect council presets.", no_args_is_help=True)
memory_app = typer.Typer(
    help="Show or reset persona memory (files under data/memory/).", no_args_is_help=True
)
app.add_typer(personas_app, name="personas")
app.add_typer(councils_app, name="councils")
sessions_app = typer.Typer(help="List, inspect or prune saved sessions.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
app.add_typer(sessions_app, name="sessions")
app.add_typer(compare_app, name="compare")
app.add_typer(chat_app, name="chat")

ROLE_BADGES = {Role.MODERATOR: "@", Role.SCRIBE: "@"}


@app.callback()
def main(
    ctx: typer.Context,
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Directory holding config.toml (default: cwd)."),
    ] = None,
) -> None:
    """Load settings once for every subcommand."""
    try:
        ctx.obj = load_settings(root)
    except ConfigError as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


def _report_errors(errors: list[LoadError]) -> None:
    """Print load errors to stderr, one problem per line."""
    for err in errors:
        typer.secho(f"✗ {err.path}", fg=typer.colors.RED, err=True)
        for problem in err.problems:
            typer.echo(f"    {problem}", err=True)


def _personas(settings: Settings) -> tuple[dict[str, Persona], list[LoadError]]:
    """Load personas from the configured directory."""
    return load_personas(settings.personas_dir)


def _nick(persona: Persona) -> str:
    """Return the nick with IRC-style badges (@ moderator, + Claude-backed)."""
    badge = ROLE_BADGES.get(persona.meta.role, "")
    if not badge and persona.meta.backend is BackendKind.CLAUDE_CLI:
        badge = "+"
    return f"{badge}{persona.name}"


@personas_app.command("list")
def personas_list(ctx: typer.Context) -> None:
    """List personas with role and backend/model."""
    personas, errors = _personas(ctx.obj)
    for p in personas.values():
        backend = f"{p.meta.backend.value}:{p.meta.model}"
        typer.echo(f"{p.slug:<14} {_nick(p):<14} {p.meta.role.value:<15} {backend}")
    _report_errors(errors)
    if errors:
        raise typer.Exit(1)


@personas_app.command("show")
def personas_show(ctx: typer.Context, slug: str) -> None:
    """Show one persona's metadata and prompt body."""
    personas, _ = _personas(ctx.obj)
    persona = personas.get(slug)
    if persona is None:
        typer.secho(f"unknown or invalid persona {slug!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    for key, value in persona.meta.model_dump(mode="json").items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"file: {persona.path}\n")
    typer.echo(persona.body)


@personas_app.command("validate")
def personas_validate(ctx: typer.Context) -> None:
    """Validate all persona and council files; exit 1 on any error."""
    settings: Settings = ctx.obj
    personas, p_errors = _personas(settings)
    councils, c_errors = load_councils(settings.councils_dir, personas)
    _report_errors(p_errors + c_errors)
    if p_errors or c_errors:
        raise typer.Exit(1)
    typer.secho(f"✓ {len(personas)} personas, {len(councils)} councils OK", fg=typer.colors.GREEN)


@personas_app.command("new")
def personas_new(
    ctx: typer.Context,
    description: Annotated[str, typer.Argument(help="One line describing the persona.")],
    slug: Annotated[str | None, typer.Option(help="Slug (file name) to use.")] = None,
    model: Annotated[
        str, typer.Option(help="Model of the new seat.")
    ] = persona_draft.DEFAULT_SEAT_MODEL,
    backend: Annotated[
        BackendKind, typer.Option(help="Backend of the new seat.")
    ] = BackendKind.OLLAMA,
    tags: Annotated[
        str | None, typer.Option(help="Comma-separated tags (default: drafted).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Write without asking.")] = False,
) -> None:
    """Draft a persona file with one Claude call, show it, then write personas/<slug>.md."""
    settings: Settings = ctx.obj
    if settings.local_only.enabled:
        raise _fail("personas new drafts with Claude: not available with [local_only] enabled")
    personas, _ = _personas(settings)
    if slug is not None and (slug in personas or (settings.personas_dir / f"{slug}.md").exists()):
        raise _fail(f"a persona with slug {slug!r} already exists")
    councils, _ = _councils(settings)
    request = persona_draft.draft_request(
        description, personas, sorted(councils), settings.claude.default_model
    )
    typer.echo(f"Drafting with Claude ({settings.claude.default_model}, 1 call)…")
    draft = _run_draft(settings, request)
    if slug is not None:
        draft = draft.model_copy(update={"slug": slug})
    seat = persona_draft.Seat(backend=backend, model=model)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags is not None else None
    text = persona_draft.render_persona(draft, seat, tag_list)
    typer.echo(f"\n{text}")
    try:
        persona_draft.check_persona(text, draft.slug, settings.personas_dir, personas)
    except (LoadError, ValueError) as exc:
        saved = _save_draft(settings, draft.slug, text)
        raise _fail(f"not written: {exc}\ndraft kept in {saved}") from exc
    path = settings.personas_dir / f"{draft.slug}.md"
    if not yes and not typer.confirm(f"Write {path}?", default=False):
        typer.echo(f"Not written; draft kept in {_save_draft(settings, draft.slug, text)}")
        return
    persona_draft.write_persona(text, path)
    typer.secho(f"✓ wrote {path}", fg=typer.colors.GREEN)
    typer.echo(f"To seat it, add {draft.slug} to a council's members in {settings.councils_dir}/")


def _run_draft(settings: Settings, request: CompletionRequest) -> persona_draft.PersonaDraft:
    """Run the drafting call (recorded in the ledger) or exit with an actionable error."""
    registry = BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir))
    try:
        return asyncio.run(
            persona_draft.draft_persona(registry.backend(BackendKind.CLAUDE_CLI), request)
        )
    except persona_draft.DraftError as exc:
        saved = _save_draft(settings, "unnamed", exc.raw)
        raise _fail(f"{exc}\nraw draft kept in {saved}") from exc
    except BackendError as exc:
        raise _fail(f"Claude call failed: {exc}") from exc


def _save_draft(settings: Settings, slug: str, text: str) -> Path:
    """Keep a draft that wasn't written, so the Claude call isn't wasted."""
    if not SLUG_PATTERN.fullmatch(slug):  # a --slug typo must not pick a path outside exports/
        slug = "unnamed"
    path = settings.data_dir / EXPORTS_DIR / f"persona-draft-{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _councils(settings: Settings) -> tuple[dict[str, CouncilPreset], list[LoadError]]:
    """Load councils, checking rosters against the loaded personas."""
    personas, _ = _personas(settings)
    return load_councils(settings.councils_dir, personas)


@councils_app.command("list")
def councils_list(ctx: typer.Context) -> None:
    """List council presets."""
    councils, errors = _councils(ctx.obj)
    for c in councils.values():
        roster = ", ".join(c.members)
        typer.echo(f"{c.name:<12} @{c.moderator} + {roster}  — {c.description}")
    _report_errors(errors)
    if errors:
        raise typer.Exit(1)


@councils_app.command("show")
def councils_show(ctx: typer.Context, name: str) -> None:
    """Show one council's roster and protocol settings."""
    councils, _ = _councils(ctx.obj)
    council = councils.get(name)
    if council is None:
        typer.secho(f"unknown or invalid council {name!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"name: {council.name}\ndescription: {council.description}")
    typer.echo(f"moderator: {council.moderator}\nmembers: {', '.join(council.members)}")
    typer.echo(f"memory: {council.memory}")
    for key, value in council.protocol.model_dump(mode="json").items():
        typer.echo(f"protocol.{key}: {value}")


def _echo_memory_file(title: str, path: Path, text: str) -> None:
    """Print one memory file with its path and size."""
    typer.secho(f"# {title}  ({path}, {size_bytes(text)} bytes)", bold=True)
    typer.echo(text.rstrip() or "(empty)")
    typer.echo()


def _check_slug(slug: str | None) -> None:
    """Exit with an error on a persona slug that can't name a memory directory."""
    if slug is not None and not SLUG_PATTERN.fullmatch(slug):
        typer.secho(f"invalid persona slug {slug!r}", fg="red", err=True)
        raise typer.Exit(2)


@memory_app.command("show")
def memory_show(
    ctx: typer.Context,
    persona: Annotated[
        str | None, typer.Option("--persona", help="Show only this persona's notes.")
    ] = None,
) -> None:
    """Show the shared user facts and every persona's private notes."""
    settings: Settings = ctx.obj
    store = MemoryStore(settings.data_dir)
    _check_slug(persona)
    if persona is not None:
        _echo_memory_file(persona, store.private_path(persona), store.read_private(persona))
        return
    _echo_memory_file("shared", store.shared_path, store.read_shared())
    for slug in store.personas_with_notes():
        _echo_memory_file(slug, store.private_path(slug), store.read_private(slug))


@memory_app.command("reset")
def memory_reset(
    ctx: typer.Context,
    persona: Annotated[
        str | None, typer.Option("--persona", help="Wipe this persona's private notes.")
    ] = None,
    shared: Annotated[bool, typer.Option("--shared", help="Wipe the shared user facts.")] = False,
    all_: Annotated[bool, typer.Option("--all", help="Wipe all memory.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
) -> None:
    """Wipe memory: exactly one of --persona SLUG, --shared or --all."""
    if sum([persona is not None, shared, all_]) != 1:
        typer.secho("choose exactly one of --persona SLUG, --shared, --all", fg="red", err=True)
        raise typer.Exit(2)
    _check_slug(persona)
    store = MemoryStore(ctx.obj.data_dir)
    if all_:
        what, wipe = "all memory", store.reset_all
    elif shared:
        what, wipe = "the shared user facts", store.reset_shared
    else:
        slug = persona or ""
        what, wipe = f"{slug}'s private notes", lambda: store.reset_private(slug)
    if not yes and not typer.confirm(f"Delete {what}?"):
        raise typer.Exit(1)
    typer.echo(f"Deleted {what}." if wipe() else f"Nothing to delete: {what} was empty.")


STYLE_COLORS = {"ok": typer.colors.GREEN, "warn": typer.colors.YELLOW, "fail": typer.colors.RED}


@app.command()
def doctor(
    ctx: typer.Context,
    local_only: Annotated[
        bool,
        typer.Option("--local-only", help="Check the local-only seat map's models; skip Claude."),
    ] = False,
    chat: Annotated[
        bool,
        typer.Option("--chat", help="Also check the Discord bot setup ([chat.discord])."),
    ] = False,
) -> None:
    """Check config, claude CLI login/flags and Ollama models (uses no quota)."""
    results = asyncio.run(run_doctor(ctx.obj, local_only))
    if chat:
        results += asyncio.run(check_chat(ctx.obj))
    for style, line in render(results):
        typer.secho(line, fg=STYLE_COLORS[style])
    if not all(r.ok for r in results):
        raise typer.Exit(1)


@app.command()
def budget(ctx: typer.Context) -> None:
    """Show Claude usage today and over the last 7 days (from data/usage.jsonl)."""
    settings: Settings = ctx.obj
    report = Ledger.in_data_dir(settings.data_dir).report()
    cap = settings.claude.daily_soft_cap_calls
    for label, summary, limit in [("today", report.today, cap), ("7 days", report.week, 0)]:
        calls = f"{summary.calls}/{limit}" if limit else str(summary.calls)
        typer.echo(
            f"{label:<7} {calls:>7} calls ({summary.failed} failed)  "
            f"{summary.input_tokens:,} in / {summary.output_tokens:,} out tokens"
        )
    _echo_quota(report.quota, report.quota_ts)
    typer.echo("Note: call counts cover council only; the quota covers all Claude use.")


def _echo_quota(quota: QuotaSnapshot | None, ts: datetime | None) -> None:
    """Print the latest Pro quota snapshot recorded in the ledger."""
    if quota is None or ts is None:
        typer.echo("Pro quota: unknown until council makes a Claude call")
        return
    for label, used, resets in [
        ("5-hour", quota.five_hour, quota.five_hour_resets_at),
        ("7-day", quota.seven_day, quota.seven_day_resets_at),
    ]:
        when = f", resets {resets.astimezone():%a %H:%M}" if resets else ""
        pct = f"{used:.0%}" if used is not None else "?"
        typer.echo(f"Pro quota {label}: {pct}{when}")
    typer.echo(f"  (as of {ts.astimezone():%a %H:%M}, status {quota.status})")


def _new_engine(
    settings: Settings,
    preset: CouncilPreset,
    personas: dict[str, Persona],
    registry: BackendRegistry,
    topic: str,
    research: bool | None,
    lang: Language | None = None,
) -> tuple[DebateEngine, SessionStore]:
    """Create the session directory and its engine ("" topic: asked in the session)."""
    try:
        research_mode = resolve_mode(settings, preset, research, registry.local_only)
    except ResearchRequiredError as exc:
        raise _fail(str(exc)) from exc
    nicks = {slug: p.name for slug, p in personas.items()}
    store = SessionStore.create(settings.data_dir, preset.name, topic, datetime.now(UTC), nicks)
    engine = DebateEngine(
        settings=settings,
        council=preset,
        personas=personas,
        registry=registry,
        topic=topic,
        store=store,
        research=research_mode,
        lang=lang,
    )
    return engine, store


def _local_setup(
    settings: Settings, personas: dict[str, Persona], local_only: bool
) -> tuple[Settings, dict[str, Persona]]:
    """Return the settings and personas a session runs with (seat map in local-only mode)."""
    if not local_only:
        return settings, personas
    councils, _ = load_councils(settings.councils_dir, personas)
    try:
        return apply_local_only(settings, personas, {c.moderator for c in councils.values()})
    except LocalOnlyConfigError as exc:
        raise _fail(f"config error: {exc}", 2) from exc


LangOption = Annotated[
    str | None,
    typer.Option("--lang", help="Debate language: en or fr. Default: detected from the question."),
]
LocalOnlyOption = Annotated[
    bool | None,
    typer.Option(
        "--local-only/--no-local-only",
        help="Local-only: every Claude seat runs on a local model (\\[local_only] in "
        "config.toml), no call reaches Claude. Nothing leaves the PC unless local search is "
        "enabled (\\[research.local]: web search via SearXNG). Default: \\[local_only] enabled.",
    ),
]


def _language(lang: str | None) -> Language | None:
    """Parse `--lang`, or exit on an unknown language; None means detect it."""
    if lang is None:
        return None
    language = parse_language(lang)
    if language is None:
        raise _fail(f"unknown language {lang!r}: use en or fr")
    return language


@app.command()
def new(
    ctx: typer.Context,
    council: Annotated[str, typer.Argument(help="Council preset name, e.g. finance.")],
    topic: Annotated[
        str | None,
        typer.Argument(help="The question to debate; omit it to type it in the session."),
    ] = None,
    no_tui: Annotated[bool, typer.Option("--no-tui", help="Plain-text REPL, no TUI.")] = False,
    research: Annotated[
        bool | None,
        typer.Option(
            "--research/--no-research",
            help="--research: the moderator searches the web before its opening "
            "(/research stays available). --no-research: no web research at all, refused "
            "by councils that require it. Default: the research section of config.toml "
            "and the council's auto_research / require_research.",
        ),
    ] = None,
    attach: Annotated[
        list[Path] | None,
        typer.Option(
            "--attach",
            help=f"Add a file to the dossier ({ALLOWED_TEXT}); repeat for several. "
            "Large files are summarized once by the moderator.",
        ),
    ] = None,
    lang: LangOption = None,
    local_only: LocalOnlyOption = None,
) -> None:
    """Start a session on a topic with a council (spends Claude quota for Claude personas)."""
    settings: Settings = ctx.obj
    language = _language(lang)
    local = settings.local_only.enabled if local_only is None else local_only
    for path in attach or []:
        try:
            read_attachable(path.expanduser(), settings.dossier.max_file_bytes)
        except DossierError as exc:
            raise _fail(str(exc)) from exc
    personas, _ = _personas(settings)
    councils, _ = load_councils(settings.councils_dir, personas)
    preset = councils.get(council)
    if preset is None:
        typer.secho(f"unknown or invalid council {council!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    settings, personas = _local_setup(settings, personas, local)
    for var in [] if local else api_keys_in_env():
        typer.secho(f"warning: {var} is set; it is removed for claude calls", fg="yellow")
    setup_logging(settings.data_dir)
    registry = BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir), local_only=local)
    engine, store = _new_engine(
        settings, preset, personas, registry, topic or "", research, language
    )
    _launch(engine, store, registry, no_tui, [p.expanduser().resolve() for p in attach or []])


def _launch(
    engine: DebateEngine,
    store: SessionStore,
    registry: BackendRegistry,
    no_tui: bool,
    attach: list[Path] | None = None,
) -> None:
    """Run a session in the TUI or the REPL, then say where its files are.

    A local-only session that may search starts SearXNG first if it's down (and
    stops it at exit); a normal session with `[research] provider` "local" or "auto"
    starts it at its first local search.
    """
    container = SearxngContainer(engine.settings.research.local, engine.settings.root)
    searches = engine.search is not None and engine.research_mode is not ResearchMode.OFF
    if engine.meta.local_only and searches and (line := asyncio.run(container.ensure_up())):
        typer.echo(line)
    elif searches and engine.search and engine.settings.research.provider != "claude":
        engine.search = StartingSearch(engine.search, container)
    on_exit = stopping(registry.aclose, container)
    if no_tui:
        run_repl(engine, on_exit=on_exit, attach=attach or [])
    else:
        CouncilApp(engine, on_exit=on_exit, attach=attach or []).run()
    if engine.awaiting_topic:  # left before typing a question: drop the empty session
        store.discard()
        return
    typer.echo(f"Session files: {store.directory}")
    if engine.phase is not Phase.DONE:
        typer.echo(f"Resume with: council resume {store.id}")


def _fail(text: str, code: int = 1) -> typer.Exit:
    """Print an error in red and return the Exit to raise."""
    typer.secho(text, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def _find(settings: Settings, ref: str | None) -> Path:
    """Resolve a session reference or exit with the lookup error."""
    try:
        return find_session(settings.data_dir, ref)
    except SessionLookupError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def resume(
    ctx: typer.Context,
    session: Annotated[
        str | None,
        typer.Argument(help="Session id, or a unique part of it (default: the latest)."),
    ] = None,
    no_tui: Annotated[bool, typer.Option("--no-tui", help="Plain-text REPL, no TUI.")] = False,
) -> None:
    """Reopen a saved session where it stopped (a turn cut by a crash runs again)."""
    settings: Settings = ctx.obj
    directory = _find(settings, session)
    personas, _ = _personas(settings)
    nicks = {slug: p.name for slug, p in personas.items()}
    try:
        store = SessionStore.open(directory, nicks)
        meta = store.load_meta()
    except (OSError, ValueError) as exc:
        raise _fail(f"can't read {directory.name}/session.json: {exc}") from exc
    councils, _ = load_councils(settings.councils_dir, personas)
    preset = councils.get(meta.council)
    if preset is None:
        raise _fail(f"session {meta.id} uses council {meta.council!r}, which is missing or invalid")
    settings, personas = _local_setup(settings, personas, meta.local_only)
    setup_logging(settings.data_dir)
    registry = BackendRegistry(
        settings, Ledger.in_data_dir(settings.data_dir), local_only=meta.local_only
    )
    try:
        engine = DebateEngine.resume(
            settings=settings, council=preset, personas=personas, registry=registry, store=store
        )
    except ResumeError as exc:
        raise _fail(str(exc)) from exc
    _launch(engine, store, registry, no_tui)


def _session_line(info: SessionInfo) -> str:
    """Return one `council sessions list` row."""
    when = f"{info.updated.astimezone():%Y-%m-%d %H:%M}"
    if info.meta is None:
        return f"{info.id}  {when}  (unreadable session.json)"
    return f"{info.id}  {when}  {_progress(info.meta)}"


def _progress(meta: SessionMeta) -> str:
    """Say where a session stands, e.g. `paused · in round 2, 1 turn(s) left`."""
    rt = meta.runtime
    if rt.phase is Phase.DONE:
        return "closed" + (" · shared facts await /keep" if rt.memory_review.pending else "")
    if rt.phase is Phase.OPENING:
        where = "not started"
    elif rt.round_in_progress:
        where = f"in round {rt.round}, {len(rt.pending)} turn(s) left"
    elif rt.phase is Phase.SYNTHESIS:
        where = f"after round {rt.round}, synthesis next"
    else:
        where = f"after round {rt.round}"
    return f"{meta.state.value.lower()} · {where}"


@sessions_app.command("list")
def sessions_list(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Show at most N sessions.")] = 20,
) -> None:
    """List saved sessions, most recently updated first."""
    infos = list_sessions(ctx.obj.data_dir)
    if not infos:
        typer.echo("No saved sessions.")
        return
    for info in infos[:limit]:
        typer.echo(_session_line(info))
    if len(infos) > limit:
        typer.echo(f"… {len(infos) - limit} more (--limit)")


@sessions_app.command("show")
def sessions_show(
    ctx: typer.Context,
    session: Annotated[str | None, typer.Argument(help="Session id or a unique part.")] = None,
) -> None:
    """Show one session: topic, council, progress and files."""
    directory = _find(ctx.obj, session)
    store = SessionStore(directory)
    try:
        meta = store.load_meta()
    except (OSError, ValueError) as exc:
        raise _fail(f"can't read {directory.name}/session.json: {exc}") from exc
    rt = meta.runtime
    messages = sum(1 for e in store.events() if e.kind == "message")
    typer.echo(f"id: {meta.id}\ntopic: {meta.topic}\ncouncil: {meta.council}")
    typer.echo(f"moderator: {meta.moderator}\nmembers: {', '.join(meta.members)}")
    typer.echo(f"started: {meta.created_at.astimezone():%Y-%m-%d %H:%M}")
    typer.echo(f"language: {LANGUAGE_NAMES[meta.language]}")
    typer.echo(f"state: {_progress(meta)}")
    if rt.round_in_progress:
        typer.echo(f"pending turns: {', '.join(t.speaker for t in rt.pending)}")
    typer.echo(f"messages: {messages}\ntranscript: {store.transcript_md}")
    if (directory / SYNTHESIS_MD).is_file():
        typer.echo(f"synthesis: {directory / SYNTHESIS_MD}")
    if rt.phase is not Phase.DONE or rt.memory_review.pending:
        typer.echo(f"resume: council resume {meta.id}")


@sessions_app.command("export")
def sessions_export(
    ctx: typer.Context,
    session: Annotated[str | None, typer.Argument(help="Session id or a unique part.")] = None,
    fmt: Annotated[
        ExportFormat, typer.Option("--format", "-f", help="md, html, or all (both).")
    ] = ExportFormat.MD,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output file, or a directory (default: data/exports/)."),
    ] = None,
    flaws: Annotated[
        bool, typer.Option("--flaws/--no-flaws", help="Include the errors caught.")
    ] = True,
) -> None:
    """Write a readable report of a session: synthesis, vote, errors caught, transcript."""
    settings: Settings = ctx.obj
    directory = _find(settings, session)
    personas, _ = _personas(settings)
    try:
        paths = export_session(
            directory,
            personas,
            user_name=settings.user_name,
            exports_dir=settings.data_dir / EXPORTS_DIR,
            fmt=fmt,
            out=out,
            flaws=flaws,
        )
    except ExportError as exc:
        raise _fail(str(exc)) from exc
    for path in paths:
        typer.echo(f"wrote {path}")


def parse_age(text: str) -> timedelta | None:
    """Parse an age like `90d` or `12w`; None if malformed."""
    match = re.fullmatch(r"(\d+)([dw])", text.strip())
    if match is None:
        return None
    days = int(match[1]) * (7 if match[2] == "w" else 1)
    return timedelta(days=days)


@sessions_app.command("prune")
def sessions_prune(
    ctx: typer.Context,
    older_than: Annotated[
        str, typer.Option("--older-than", help="Age since the last update, e.g. 90d or 12w.")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
) -> None:
    """Delete sessions not updated for a while (memory is not touched)."""
    age = parse_age(older_than)
    if age is None:
        raise _fail("--older-than takes a number of days or weeks, e.g. 90d or 12w", 2)
    cutoff = datetime.now(UTC) - age
    old = [i for i in list_sessions(ctx.obj.data_dir) if i.updated < cutoff]
    if not old:
        typer.echo(f"No sessions older than {older_than}.")
        return
    for info in old:
        typer.echo(_session_line(info))
    if not yes and not typer.confirm(f"Delete these {len(old)} session(s)?"):
        raise typer.Exit(1)
    for info in old:
        remove_session(info.directory)
    typer.echo(f"Deleted {len(old)} session(s).")
