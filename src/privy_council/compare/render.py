"""The comparison report, in Markdown and HTML (built from saved files, no model calls).

The report's own words are English: it is the user's tool, not a debate.
Model text (syntheses, the judge, vote positions) stays in the debate language.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from html import escape
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from privy_council.compare.judge import letters
from privy_council.compare.metrics import VariantMetrics, variant_metrics
from privy_council.compare.store import CompareState, CompareStore
from privy_council.engine.budget import Ledger
from privy_council.export.html import CSS, markdown_html
from privy_council.export.markdown import demote
from privy_council.export.report import flaw_text, model_cell
from privy_council.models.persona import Persona
from privy_council.storage.sessions import SESSIONS_DIR

REPORT_STEM = "report"
EXTRA_CSS = """
.num { text-align: right; white-space: nowrap; }
.judge { background: var(--panel); border-radius: 6px; padding: .2rem 1rem; }
"""


class CompareReport(BaseModel):
    """A comparison, ready to render."""

    model_config = ConfigDict(frozen=True)

    state: CompareState
    variants: list[VariantMetrics]
    judge: str | None


def build_compare_report(
    store: CompareStore,
    data_dir: Path,
    personas: Mapping[str, Persona],
    user_name: str,
    ledger: Ledger | None,
) -> CompareReport:
    """Read a comparison and its sessions (raises OSError, ValueError or ExportError)."""
    state = store.load()
    sessions_dir = data_dir / SESSIONS_DIR
    variants = [
        variant_metrics(r, sessions_dir, personas, user_name, ledger) for r in state.variants
    ]
    return CompareReport(state=state, variants=variants, judge=store.load_judge())


def write_reports(report: CompareReport, directory: Path) -> list[Path]:
    """Write report.md and report.html into the comparison directory."""
    paths = []
    for ext, text in (("md", render_markdown(report)), ("html", render_html(report))):
        path = directory / f"{REPORT_STEM}.{ext}"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def duration(seconds: float) -> str:
    """Format seconds as "45 s", "3 min 20 s" or "1 h 05 min"."""
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def _confidence(v: VariantMetrics) -> str:
    """Return the vote's confidence spread, "min-max (mean)"."""
    spread = v.confidence()
    if spread is None:
        return "—"
    low, mean, high = spread
    return f"{low}%" if low == high else f"{low}-{high}% (mean {mean:.0f}%)"


def _errors(v: VariantMetrics) -> str:
    """Return the flaw total and its non-zero members, e.g. "3 (Rigor 2, Value 1)"."""
    counts = v.flaw_counts()
    if v.report is None:
        return "—"
    total = sum(n for _, n in counts)
    named = ", ".join(f"{nick} {n}" for nick, n in counts if n)
    return f"{total} ({named})" if named else str(total)


def summary_rows(report: CompareReport) -> list[list[str]]:
    """Return the summary table's rows, one per variant."""
    return [
        [
            v.name,
            v.label,
            v.status.value,
            str(v.rounds),
            str(v.claude_calls),
            duration(v.run_s) if v.report else "—",
            _errors(v),
            _confidence(v),
            " → ".join(v.consensus) or "—",
        ]
        for v in report.variants
    ]


SUMMARY_HEAD = [
    "Variant",
    "Changes",
    "Status",
    "Rounds",
    "Claude calls",
    "Run time",
    "Errors caught",
    "Confidence",
    "Consensus by round",
]
SEAT_HEAD = ["Member", "Model", "Turns", "Median turn", "Slowest", "Errors caught"]
VOTE_HEAD = ["Member", "Confidence", "Position"]


def seat_rows(v: VariantMetrics) -> list[list[str]]:
    """Return a variant's members with their model, turn times and flaws."""
    if v.report is None:
        return []
    counts = dict(v.flaw_counts())
    times = {t.seat.slug: t for t in v.times}
    rows = []
    for seat in v.report.seats:
        if seat.slug == v.report.meta.moderator:
            continue
        t = times.get(seat.slug)
        timing = [str(t.turns), duration(t.median_s), duration(t.max_s)] if t else ["0", "—", "—"]
        rows.append([seat.nick, model_cell(seat), *timing, str(counts.get(seat.nick, 0))])
    return rows


def vote_rows(v: VariantMetrics) -> list[list[str]]:
    """Return the final vote's rows: member, confidence, position (or why there is none)."""
    if v.report is None or v.vote is None:
        return []
    return [
        [v.report.nick(x.speaker), "—" if x.error else f"{x.confidence}%", x.error or x.position]
        for x in v.vote.votes
    ]


def _facts(report: CompareReport) -> list[tuple[str, str]]:
    """Return the header's label/value pairs."""
    state = report.state
    notes = state.shared_research
    research = f"{len(notes)} note(s), shared by every variant" if notes else "none"
    reports = [v.report for v in report.variants if v.report]
    moderator = reports[0].seat(reports[0].meta.moderator) if reports else None
    return [
        ("Council:", state.council),
        ("Started:", f"{state.created_at.astimezone():%Y-%m-%d %H:%M}"),
        ("Rounds:", str(state.rounds)),
        ("Moderator:", f"{moderator.nick} ({model_cell(moderator)})" if moderator else "—"),
        ("Research:", research),
        ("Comparison:", state.id),
    ]


def _letters_line(report: CompareReport) -> str:
    """Return which letter the judge saw for each variant."""
    names = [v.name for v in report.variants if v.report and v.report.synthesis]
    return ", ".join(f"{letter} = {name}" for name, letter in letters(names).items())


# --- Markdown ----------------------------------------------------------------------


def _md_cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _md_table(head: list[str], rows: list[list[str]]) -> str:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(report: CompareReport, now: datetime | None = None) -> str:
    """Return the whole comparison report as Markdown."""
    parts = [f"# Comparison: {report.state.topic}", ""]
    parts += [f"- **{k}** {v}" for k, v in _facts(report)]
    parts += ["", "## Summary", "", _md_table(SUMMARY_HEAD, summary_rows(report)), ""]
    if report.judge:
        parts += ["## Judge", "", f"*Variants shown to the judge as {_letters_line(report)}.*", ""]
        parts += [demote(report.judge, 1), ""]
    for v in report.variants:
        parts += _md_variant(v)
    when = f"{(now or datetime.now().astimezone()).astimezone():%Y-%m-%d %H:%M}"
    parts += [
        "---",
        "",
        f"*Written {when}. Full transcripts: `council sessions export <session>`.*",
    ]
    return "\n".join(parts) + "\n"


def _md_variant(v: VariantMetrics) -> list[str]:
    """Return one variant's section."""
    parts = [f"## {v.name}: {v.label}", ""]
    if v.report is None:
        return [*parts, f"*Not run yet ({v.status.value}).*", ""]
    parts += [f"Session `{v.report.meta.id}`" + (f". Error: {v.error}" if v.error else ""), ""]
    parts += [_md_table(SEAT_HEAD, seat_rows(v)), ""]
    if v.vote is not None:
        parts += ["### Final vote", "", _md_table(VOTE_HEAD, vote_rows(v)), ""]
    parts += ["### Synthesis", ""]
    parts += [demote(v.report.synthesis, 3) if v.report.synthesis else "*No synthesis yet.*", ""]
    if v.report.flaws:
        parts += ["### Errors caught", ""]
        parts += [f"- {v.report.nick(f.member)}: {flaw_text(f)}" for f in v.report.flaws]
        parts.append("")
    return parts


# --- HTML --------------------------------------------------------------------------


def _html_table(
    head: list[str], rows: list[list[str]], numeric: frozenset[int] = frozenset()
) -> str:
    """Return an HTML table; `numeric` columns are right-aligned."""
    th = "".join(f"<th>{escape(h)}</th>" for h in head)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{escape(c)}</td>' if i in numeric else f"<td>{escape(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: CompareReport, now: datetime | None = None) -> str:
    """Return the whole comparison report as one HTML page (inline CSS, no scripts)."""
    topic = report.state.topic
    items = "".join(f"<li><b>{escape(k)}</b> {escape(v)}</li>" for k, v in _facts(report))
    parts = [
        f"<h1>Comparison: {escape(topic)}</h1>",
        f'<ul class="facts">{items}</ul>',
        "<h2>Summary</h2>",
        _html_table(SUMMARY_HEAD, summary_rows(report), frozenset({3, 4, 5})),
    ]
    if report.judge:
        parts += [
            "<h2>Judge</h2>",
            f'<p class="note">Variants shown to the judge as {escape(_letters_line(report))}.</p>',
            f'<div class="judge">{markdown_html(report.judge, 1)}</div>',
        ]
    parts += [_html_variant(v) for v in report.variants]
    when = f"{(now or datetime.now().astimezone()).astimezone():%Y-%m-%d %H:%M}"
    parts.append(
        f"<footer>Written {escape(when)}. Full transcripts: "
        "<code>council sessions export &lt;session&gt;</code>.</footer>"
    )
    lang = report.state.lang or "en"
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Comparison: {escape(topic)}</title>\n<style>{CSS}{EXTRA_CSS}</style>\n</head>\n"
        "<body>\n<main>\n" + "\n".join(parts) + "\n</main>\n</body>\n</html>\n"
    )


def _html_variant(v: VariantMetrics) -> str:
    """Return one variant's section."""
    parts = [f"<h2>{escape(v.name)}: {escape(v.label)}</h2>"]
    if v.report is None:
        return parts[0] + f'<p class="note">Not run yet ({escape(v.status.value)}).</p>'
    error = f" Error: {escape(v.error)}" if v.error else ""
    parts.append(f'<p class="note">Session <code>{escape(v.report.meta.id)}</code>.{error}</p>')
    parts.append(_html_table(SEAT_HEAD, seat_rows(v), frozenset({2, 3, 4, 5})))
    if v.vote is not None:
        parts += ["<h3>Final vote</h3>", _html_table(VOTE_HEAD, vote_rows(v), frozenset({1}))]
    parts.append("<h3>Synthesis</h3>")
    if v.report.synthesis:
        parts.append(markdown_html(v.report.synthesis, 3))
    else:
        parts.append('<p class="note">No synthesis yet.</p>')
    if v.report.flaws:
        items = "".join(
            f"<li>{escape(v.report.nick(f.member))}: {escape(flaw_text(f))}</li>"
            for f in v.report.flaws
        )
        parts.append(f'<h3>Errors caught</h3><ul class="flaws">{items}</ul>')
    return "\n".join(parts)
