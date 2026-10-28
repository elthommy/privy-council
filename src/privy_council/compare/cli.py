"""`council compare ...`: run one question through several council variants (M14)."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated

import typer

from privy_council.backends.base import BackendError
from privy_council.backends.registry import BackendRegistry
from privy_council.compare import estimate
from privy_council.compare.render import build_compare_report, duration, write_reports
from privy_council.compare.runner import CompareRunner, Outcome, run_judge, variant_research
from privy_council.compare.spec import (
    Variant,
    VariantError,
    apply_variant,
    check_variants,
    parse_variant,
)
from privy_council.compare.store import (
    CompareLookupError,
    CompareState,
    CompareStore,
    VariantRun,
    VariantStatus,
    find_compare,
    list_compares,
)
from privy_council.config import Settings
from privy_council.engine.budget import Ledger, cap_warnings, quota_warnings
from privy_council.engine.research import ResearchRequiredError, resolve_mode
from privy_council.export import ExportError
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import LocalOnlyConfigError, apply_local_only
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import BackendKind, Persona
from privy_council.util.language import parse_language
from privy_council.util.logs import setup_logging

compare_app = typer.Typer(
    help="Run one question through several variants of a council and compare them.",
    no_args_is_help=True,
)

DEFAULT_ROUNDS = 3


def _fail(text: str, code: int = 1) -> typer.Exit:
    """Print an error in red and return the Exit to raise."""
    typer.secho(text, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def _load(
    settings: Settings, council: str, local_only: bool
) -> tuple[Settings, CouncilPreset, dict[str, Persona]]:
    """Return the settings, council preset and personas (local seats if asked), or exit."""
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    preset = councils.get(council)
    if preset is None:
        raise _fail(f"unknown or invalid council {council!r}")
    if local_only:
        try:
            moderators = {c.moderator for c in councils.values()}
            settings, personas = apply_local_only(settings, personas, moderators)
        except LocalOnlyConfigError as exc:
            raise _fail(f"config error: {exc}", 2) from exc
    return settings, preset, personas


def _check(
    preset: CouncilPreset,
    personas: Mapping[str, Persona],
    variants: list[Variant],
    rounds: int,
    local_only: bool = False,
) -> None:
    """Exit if a variant doesn't fit the council, or seats Claude in local-only mode."""
    try:
        check_variants(variants)
        for variant in variants:
            council, seated = apply_variant(preset, personas, variant, rounds)
            claude = [
                slug
                for slug in [council.moderator, *council.members]
                if seated[slug].meta.backend is BackendKind.CLAUDE_CLI
            ]
            if local_only and claude:
                raise VariantError(
                    f"variant {variant.name!r} seats {', '.join(claude)} on Claude: "
                    "not in local-only mode"
                )
    except VariantError as exc:
        raise _fail(str(exc)) from exc


VariantOption = Annotated[
    list[str],
    typer.Option(
        "--variant",
        "-v",
        help="NAME[:CHANGE,...], repeat for each variant (two or more). A CHANGE is "
        "slug=model or slug=backend:model (another model for a seat), +slug (add a "
        "member) or -slug (drop one); a bare NAME is the council as it is. Example: "
        "-v base -v qwen:rigor=qwen3.6:27b",
    ),
]
RoundsOption = Annotated[
    int, typer.Option("--rounds", min=1, max=20, help="Rounds every variant debates.")
]
ResearchOption = Annotated[
    bool | None,
    typer.Option(
        "--research/--no-research",
        help="Web research by the moderator, once, shared by every variant. "
        "Default: as for `council new`.",
    ),
]
JudgeOption = Annotated[
    bool, typer.Option("--judge", help="End with one Claude call comparing the syntheses.")
]
LocalOnlyOption = Annotated[
    bool | None,
    typer.Option(
        "--local-only/--no-local-only",
        help="Every variant runs local-only (no Claude call, no --judge; web research only "
        "with local search, \\[research.local]). "
        "Default: \\[local_only] enabled.",
    ),
]
LangOption = Annotated[
    str | None, typer.Option("--lang", help="en or fr. Default: detected from the question.")
]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Run without asking.")]
CompareArg = Annotated[
    str | None, typer.Argument(help="Comparison id, or a unique part of it (default: latest).")
]


@compare_app.command("new")
def compare_new(
    ctx: typer.Context,
    council: Annotated[str, typer.Argument(help="Council preset name, e.g. geopolitics.")],
    topic: Annotated[str, typer.Argument(help="The question every variant debates.")],
    variant: VariantOption,
    rounds: RoundsOption = DEFAULT_ROUNDS,
    research: ResearchOption = None,
    judge: JudgeOption = False,
    lang: LangOption = None,
    local_only: LocalOnlyOption = None,
    yes: YesOption = False,
) -> None:
    """Debate one question with each variant in turn, then write a comparison report.

    Spends Claude quota: the estimate is shown before anything runs. Ctrl-C (or
    creating the STOP file) stops at the next turn boundary; `resume` goes on.
    """
    settings: Settings = ctx.obj
    local = settings.local_only.enabled if local_only is None else local_only
    if local and judge:
        raise _fail("--judge is one Claude call: not in local-only mode")
    settings, preset, personas = _load(settings, council, local)
    state = _new_state(settings, preset, personas, topic, variant, rounds, research, lang, local)
    state.judge = judge
    if not _confirm(settings, preset, personas, state, yes):
        raise typer.Exit(0)
    store = CompareStore.create(settings.data_dir, preset.name, topic, datetime.now(UTC))
    state.id = store.id
    store.save(state)
    typer.echo(f"Comparison {store.id}")
    _run(settings, preset, personas, store)


def _new_state(
    settings: Settings,
    preset: CouncilPreset,
    personas: Mapping[str, Persona],
    topic: str,
    variant: list[str],
    rounds: int,
    research: bool | None,
    lang: str | None,
    local_only: bool,
) -> CompareState:
    """Check the command line and return the comparison to run (id set later), or exit."""
    try:
        variants = [parse_variant(text) for text in variant]
    except VariantError as exc:
        raise _fail(str(exc)) from exc
    _check(preset, personas, variants, rounds, local_only)
    language = parse_language(lang) if lang is not None else None
    if lang is not None and language is None:
        raise _fail(f"unknown language {lang!r}: use en or fr")
    try:
        mode = resolve_mode(settings, preset, research, local_only)
    except ResearchRequiredError as exc:
        raise _fail(str(exc)) from exc
    return CompareState(
        id="",
        council=preset.name,
        topic=topic,
        rounds=rounds,
        lang=language,
        research=mode.value,
        variants=[VariantRun(variant=v) for v in variants],
        local_only=local_only,
    )


@compare_app.command("resume")
def compare_resume(
    ctx: typer.Context,
    compare: CompareArg = None,
    yes: YesOption = False,
) -> None:
    """Go on with a stopped or failed comparison where it stopped."""
    settings: Settings = ctx.obj
    store = _find(settings, compare)
    state = _state(store)
    settings, preset, personas = _load(settings, state.council, state.local_only)
    _check(preset, personas, [r.variant for r in state.variants], state.rounds, state.local_only)
    if state.done and (state.judged or not state.judge):
        typer.echo("Every variant is done; writing the report.")
    elif not _confirm(settings, preset, personas, state, yes):
        raise typer.Exit(0)
    _run(settings, preset, personas, store)


@compare_app.command("report")
def compare_report(
    ctx: typer.Context,
    compare: CompareArg = None,
) -> None:
    """Write the comparison report again (Markdown and HTML), from the saved sessions."""
    settings: Settings = ctx.obj
    _report(settings, _find(settings, compare))


@compare_app.command("list")
def compare_list(ctx: typer.Context) -> None:
    """List saved comparisons, newest first."""
    stores = list_compares(ctx.obj.data_dir)
    if not stores:
        typer.echo("No comparisons yet.")
        return
    for store in stores:
        try:
            state = store.load()
        except (OSError, ValueError):
            typer.echo(f"{store.id}  (unreadable compare.json)")
            continue
        done = sum(r.status is VariantStatus.DONE for r in state.variants)
        typer.echo(f"{store.id}  {done}/{len(state.variants)} variants done · {state.topic}")


def _find(settings: Settings, ref: str | None) -> CompareStore:
    """Resolve a comparison reference or exit."""
    try:
        return find_compare(settings.data_dir, ref)
    except CompareLookupError as exc:
        raise _fail(str(exc)) from exc


def _state(store: CompareStore) -> CompareState:
    """Read compare.json or exit."""
    try:
        return store.load()
    except (OSError, ValueError) as exc:
        raise _fail(f"can't read {store.id}/compare.json: {exc}") from exc


def _confirm(
    settings: Settings,
    preset: CouncilPreset,
    personas: Mapping[str, Persona],
    state: CompareState,
    yes: bool,
) -> bool:
    """Show the estimate for the variants still to run, with budget warnings; ask."""
    ledger = Ledger.in_data_dir(settings.data_dir)
    history = estimate.turn_history(settings.data_dir, personas)
    moderator = personas[preset.moderator]
    moderator_s = estimate.moderator_history(ledger, moderator)
    rows: list[estimate.VariantEstimate] = []
    typer.echo(f"{state.council} · {state.rounds} rounds · {state.topic}")
    for index, run in enumerate(state.variants):
        if run.status is VariantStatus.DONE:
            continue
        council, seated = apply_variant(preset, personas, run.variant, state.rounds)
        est = estimate.estimate_variant(
            council,
            seated,
            rounds=state.rounds,
            research=variant_research(state, index),
            history=history,
            moderator_s=moderator_s,
        )
        rows.append(est)
        typer.echo(
            f"  {run.variant.name:<14} ≤{est.claude_calls:>3} Claude calls  "
            f"~{duration(est.seconds):>12}  {run.variant.label()}"
        )
    judge = state.judge and not state.judged
    calls = estimate.total_calls(rows, judge)
    seconds = sum(e.seconds for e in rows)
    typer.echo(
        f"  total: ≤{calls} Claude calls{' (judge included)' if judge else ''}, "
        f"~{duration(seconds)} (resumable at any turn: Ctrl-C or a STOP file)"
    )
    report = ledger.report()
    claude = settings.claude
    biggest = max((e.claude_calls for e in rows), default=0)
    warnings = cap_warnings(report, calls, 0, claude.daily_soft_cap_calls)
    warnings += cap_warnings(report, biggest, claude.session_soft_cap_calls, 0)
    warnings += quota_warnings(report.quota, claude.quota_warn_utilization)
    for text in warnings:
        typer.secho(f"! budget: {text}", fg=typer.colors.YELLOW)
    return yes or typer.confirm("Run it?", default=False)


def _run(
    settings: Settings,
    preset: CouncilPreset,
    personas: Mapping[str, Persona],
    store: CompareStore,
) -> None:
    """Run the comparison's remaining variants and judge, then write the report."""
    setup_logging(settings.data_dir)
    local_only = store.load().local_only
    registry = BackendRegistry(
        settings, Ledger.in_data_dir(settings.data_dir), local_only=local_only
    )
    runner = CompareRunner(
        settings=settings,
        preset=preset,
        personas=personas,
        registry=registry,
        store=store,
        progress=typer.echo,
    )
    typer.echo(f"To stop after the current turn: Ctrl-C, or touch {store.stop_file}")
    try:
        outcome = asyncio.run(_drive(runner, registry, settings, store))
    except asyncio.CancelledError:
        typer.secho("Stopped at once; the turn in flight runs again on resume.", fg="yellow")
        outcome = Outcome.STOPPED
    _report(settings, store)
    if outcome is not Outcome.DONE:
        typer.echo(f"Resume with: council compare resume {store.id}")


async def _drive(
    runner: CompareRunner, registry: BackendRegistry, settings: Settings, store: CompareStore
) -> Outcome:
    """Run the variants (Ctrl-C: stop at a turn boundary; twice: at once), then the judge."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()

    def on_sigint() -> None:
        """Ask for a clean stop; a second Ctrl-C cancels the call in flight."""
        if runner.stopping and task is not None:
            task.cancel()
        else:
            runner.request_stop()

    loop.add_signal_handler(signal.SIGINT, on_sigint)
    try:
        outcome = await runner.run()
        state = store.load()
        if outcome is Outcome.DONE and state.judge and not state.judged:
            typer.echo("Judge: comparing the syntheses (1 Claude call)…")
            try:
                await run_judge(store, settings, registry)
            except BackendError as exc:
                typer.secho(f"!! judge failed: {exc} (resume retries it)", fg="red")
                return Outcome.FAILED
        return outcome
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await registry.aclose()


def _report(settings: Settings, store: CompareStore) -> None:
    """Write report.md and report.html, or exit."""
    personas, _ = load_personas(settings.personas_dir)
    ledger = Ledger.in_data_dir(settings.data_dir)
    try:
        report = build_compare_report(
            store, settings.data_dir, personas, settings.user_name, ledger
        )
        paths = write_reports(report, store.directory)
    except (OSError, ValueError, ExportError) as exc:
        raise _fail(f"can't write the report: {exc}") from exc
    for path in paths:
        typer.echo(f"Report: {path}")
