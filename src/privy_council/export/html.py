"""Render a Report as one self-contained HTML file (inline CSS, no scripts, printable)."""

from __future__ import annotations

from datetime import datetime
from html import escape

from markdown_it import MarkdownIt

from privy_council.export.labels import EXPORT_LABELS, ROLE_NAMES
from privy_council.export.markdown import demote
from privy_council.export.report import (
    Report,
    ShownEvent,
    facts,
    flaw_text,
    flaws_by_member,
    model_cell,
    vote_rows,
)
from privy_council.models.session import (
    DossierAdded,
    ErrorEvent,
    MessageEvent,
    ResearchAdded,
    RoundStarted,
    VoteTaken,
)

# Model output is Markdown; raw HTML in it is shown as text, never run.
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")

CSS = """
:root { color-scheme: light dark; --bg: #fdfdfb; --fg: #1f2328; --muted: #656d76;
  --line: #d8dee4; --panel: #f3f4f1; --accent: #2f6f9f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1e2127; --fg: #d7dae0; --muted: #8b929c; --line: #3a3f4b;
    --panel: #262a31; --accent: #61afef; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 52rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 1rem; }
h2 { font-size: 1.25rem; margin: 2.5rem 0 1rem; padding-bottom: .3rem;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 1.8rem 0 .6rem; }
h4, h5, h6 { font-size: 1rem; margin: 1.2rem 0 .4rem; }
a { color: var(--accent); }
code { font-size: .9em; }
.facts { display: flex; flex-wrap: wrap; gap: .3rem 1.5rem; margin: 0 0 1rem; padding: 0;
  list-style: none; color: var(--muted); }
.facts b { color: var(--fg); font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .95rem; }
th, td { text-align: left; vertical-align: top; padding: .35rem .6rem;
  border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
.dot { display: inline-block; width: .7em; height: .7em; border-radius: 50%;
  background: var(--c); margin-right: .45em; }
.note { color: var(--muted); font-size: .9rem; }
.msg { border-left: 3px solid var(--c); padding: .1rem 0 .1rem .8rem; margin: 1rem 0; }
.msg .who { font-weight: 600; }
.msg .when, .flag { color: var(--muted); font-size: .85rem; font-weight: 400; }
.flag { margin-left: .5em; }
.msg .body > :first-child { margin-top: .2rem; }
.msg .body > :last-child { margin-bottom: 0; }
.system { color: var(--muted); font-style: italic; font-size: .9rem; margin: .5rem 0; }
.error { color: #c0392b; }
details { background: var(--panel); border-radius: 6px; padding: .5rem .8rem; margin: 1rem 0; }
summary { cursor: pointer; font-weight: 600; }
.flaws li { margin: .3rem 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
@media print {
  body { font-size: 11pt; background: #fff; color: #000; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; }
  .msg, details, tr { break-inside: avoid; }
}
"""


def markdown_html(text: str, levels: int = 0) -> str:
    """Render model Markdown to HTML, its headings pushed down `levels`."""
    rendered: str = _MD.render(demote(text, levels) if levels else text)
    return rendered


def render_html(report: Report, flaws: bool = True, now: datetime | None = None) -> str:
    """Return the whole report as one HTML page."""
    labels = EXPORT_LABELS[report.meta.language]
    parts = [_header(report, labels), _synthesis(report, labels)]
    if report.final_vote is not None:
        title = labels["final_vote"].format(round=report.final_vote.round)
        parts.append(f"<h2>{escape(title)}</h2>{_vote(report, report.final_vote, labels)}")
    if flaws and report.flaws:
        parts.append(_flaws(report, labels))
    parts.append(_transcript(report, labels))
    when = f"{(now or datetime.now().astimezone()).astimezone():%Y-%m-%d %H:%M}"
    parts.append(f"<footer>{escape(labels['exported'].format(when=when))}</footer>")
    return (
        f'<!doctype html>\n<html lang="{report.meta.language}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(report.meta.topic)}</title>\n<style>{CSS}</style>\n</head>\n"
        "<body>\n<main>\n" + "\n".join(parts) + "\n</main>\n</body>\n</html>\n"
    )


def _header(report: Report, labels: dict[str, str]) -> str:
    """Return the title, the session facts and the members table."""
    items = "".join(f"<li><b>{escape(k)}</b> {escape(v)}</li>" for k, v in facts(report, labels))
    roles = ROLE_NAMES[report.meta.language]
    rows = []
    for seat in report.seats:
        mark = "" if seat.recorded else " *"
        rows.append(
            f'<tr><td><span class="dot" style="--c:{seat.color}"></span>{escape(seat.nick)}</td>'
            f"<td>{escape(roles.get(seat.role, seat.role))}</td>"
            f"<td><code>{escape(model_cell(seat))}</code>{mark}</td></tr>"
        )
    note = ""
    if any(not s.recorded for s in report.seats):
        note = f'<p class="note">* {escape(labels["model_unrecorded"])}</p>'
    head = "".join(f"<th>{escape(labels[k])}</th>" for k in ("member", "role", "model"))
    return (
        f'<h1>{escape(report.meta.topic)}</h1>\n<ul class="facts">{items}</ul>\n'
        f"<h2>{escape(labels['members'])}</h2>\n"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"
    )


def _synthesis(report: Report, labels: dict[str, str]) -> str:
    """Return the synthesis section, its headings nested under it."""
    if report.synthesis:
        body = markdown_html(report.synthesis, 1)
    else:
        body = f'<p class="note">{escape(labels["no_synthesis"])}</p>'
    return f"<h2>{escape(labels['synthesis'])}</h2>\n{body}"


def _vote(report: Report, vote: VoteTaken, labels: dict[str, str]) -> str:
    """Return a vote as a table."""
    head = "".join(f"<th>{escape(labels[k])}</th>" for k in ("member", "confidence", "position"))
    rows = "".join(
        f"<tr><td>{escape(n)}</td><td>{escape(c)}</td><td>{escape(p)}</td></tr>"
        for n, c, p in vote_rows(report, vote, labels)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _flaws(report: Report, labels: dict[str, str]) -> str:
    """Return the flaws section, grouped by member."""
    parts = [
        f"<h2>{escape(labels['errors'])}</h2>",
        f'<p class="note">{escape(labels["errors_note"])}</p>',
    ]
    for nick, flaws in flaws_by_member(report):
        items = "".join(f"<li>{escape(flaw_text(f))}</li>" for f in flaws)
        parts.append(f'<h3>{escape(nick)}</h3><ul class="flaws">{items}</ul>')
    return "\n".join(parts)


def _transcript(report: Report, labels: dict[str, str]) -> str:
    """Return the transcript by round."""
    parts = [f"<h2>{escape(labels['transcript'])}</h2>"]
    if report.events and not isinstance(report.events[0], RoundStarted):
        parts.append(f"<h3>{escape(labels['opening'])}</h3>")
    parts += [_event(report, e, labels) for e in report.events]
    return "\n".join(parts)


def _event(report: Report, event: ShownEvent, labels: dict[str, str]) -> str:
    """Render one transcript event."""
    if isinstance(event, RoundStarted):
        return f"<h3>{escape(event.label or f'Round {event.round}')}</h3>"
    if isinstance(event, ResearchAdded):
        note = event.note
        searches = "; ".join(note.searches)
        source = labels[f"source_{note.source}"]
        searched = f" · {labels['searches']} {searches}" if searches else ""
        meta = f'<p class="note">{escape(source + searched)}</p>'
        summary = f"<summary>{escape(labels['research'])} {escape(note.focus)}</summary>"
        return f"<details>{summary}{meta}{markdown_html(note.text, 3)}</details>"
    if isinstance(event, DossierAdded):
        text = labels["attached"].format(name=event.file.name, tokens=event.file.tokens)
        return f'<p class="system">{escape(text)}</p>'
    if isinstance(event, VoteTaken):
        title = labels["vote_after"].format(round=event.round)
        return f"<h4>{escape(title)}</h4>{_vote(report, event, labels)}"
    if isinstance(event, ErrorEvent):
        return f'<p class="system error"><b>{escape(labels["error"])}</b> {escape(event.text)}</p>'
    return _message(report, event, labels)


def _message(report: Report, event: MessageEvent, labels: dict[str, str]) -> str:
    """Render one chat message (a system line muted)."""
    msg = event.message
    if msg.speaker == "system":
        return f'<p class="system">{escape(msg.text)}</p>'
    flags = [labels[k] for k, on in [("fallback", msg.fallback), ("cut_off", msg.truncated)] if on]
    flags += [labels["to_user"]] if msg.to_user else []
    flag_html = "".join(f'<span class="flag">[{escape(f)}]</span>' for f in flags)
    return (
        f'<div class="msg" style="--c:{report.color(msg.speaker)}">'
        f'<div><span class="when">{msg.ts.astimezone():%H:%M}</span> '
        f'<span class="who">{escape(msg.display_nick(report.nick))}</span>{flag_html}</div>'
        f'<div class="body">{markdown_html(msg.text, 3)}</div></div>'
    )
