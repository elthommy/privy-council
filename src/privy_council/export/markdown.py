"""Render a Report as one Markdown file."""

from __future__ import annotations

import re
from datetime import datetime

from privy_council.export.labels import EXPORT_LABELS, ROLE_NAMES
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

_HEADING = re.compile(r"^(#{1,5}) ", re.MULTILINE)


def demote(text: str, levels: int) -> str:
    """Push Markdown headings down so a model's `## X` nests under the report's own."""
    return _HEADING.sub(lambda m: "#" * min(len(m.group(1)) + levels, 6) + " ", text)


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def render_markdown(report: Report, flaws: bool = True, now: datetime | None = None) -> str:
    """Return the whole report as Markdown."""
    labels = EXPORT_LABELS[report.meta.language]
    parts = [_header(report, labels), _synthesis(report, labels)]
    if report.final_vote is not None:
        title = labels["final_vote"].format(round=report.final_vote.round)
        parts.append(_vote(report, report.final_vote, f"## {title}", labels))
    if flaws and report.flaws:
        parts.append(_flaws(report, labels))
    parts.append(_transcript(report, labels))
    when = f"{(now or datetime.now().astimezone()).astimezone():%Y-%m-%d %H:%M}"
    parts.append(f"---\n\n*{labels['exported'].format(when=when)}*\n")
    return "\n".join(parts)


def _header(report: Report, labels: dict[str, str]) -> str:
    """Return the title, the session facts and the members table."""
    lines = [f"# {report.meta.topic}", ""]
    lines += [f"- **{label}** {value}" for label, value in facts(report, labels)]
    roles = ROLE_NAMES[report.meta.language]
    lines += ["", f"## {labels['members']}", ""]
    lines += [f"| {labels['member']} | {labels['role']} | {labels['model']} |", "|---|---|---|"]
    unrecorded = False
    for seat in report.seats:
        mark = "" if seat.recorded else " \\*"
        unrecorded = unrecorded or not seat.recorded
        role = roles.get(seat.role, seat.role)
        lines.append(f"| {seat.nick} | {role} | `{model_cell(seat)}`{mark} |")
    if unrecorded:
        lines += ["", f"\\* {labels['model_unrecorded']}"]
    return "\n".join(lines) + "\n"


def _synthesis(report: Report, labels: dict[str, str]) -> str:
    """Return the synthesis section, its headings nested under it."""
    body = demote(report.synthesis, 1) if report.synthesis else f"*{labels['no_synthesis']}*"
    return f"## {labels['synthesis']}\n\n{body}\n"


def _vote(report: Report, vote: VoteTaken, title: str, labels: dict[str, str]) -> str:
    """Return a vote as a titled table."""
    lines = [title, "", f"| {labels['member']} | {labels['confidence']} | {labels['position']} |"]
    lines.append("|---|---|---|")
    lines += [f"| {n} | {c} | {_cell(p)} |" for n, c, p in vote_rows(report, vote, labels)]
    return "\n".join(lines) + "\n"


def _flaws(report: Report, labels: dict[str, str]) -> str:
    """Return the flaws section, grouped by member."""
    lines = [f"## {labels['errors']}", "", f"*{labels['errors_note']}*", ""]
    for nick, flaws in flaws_by_member(report):
        lines += [f"### {nick}", ""]
        lines += [f"- {_cell(flaw_text(f))}" for f in flaws]
        lines.append("")
    return "\n".join(lines)


def _transcript(report: Report, labels: dict[str, str]) -> str:
    """Return the transcript by round."""
    lines = [f"## {labels['transcript']}\n"]
    if report.events and not isinstance(report.events[0], RoundStarted):
        lines.append(f"### {labels['opening']}\n")
    lines += [_event(report, e, labels) for e in report.events]
    return "\n".join(lines)


def _event(report: Report, event: ShownEvent, labels: dict[str, str]) -> str:
    """Render one transcript event."""
    if isinstance(event, RoundStarted):
        return f"### {event.label or f'Round {event.round}'}\n"
    if isinstance(event, ResearchAdded):
        note = event.note
        searches = "; ".join(note.searches)
        lines = [f"> **{labels['research']} {note.focus}**"]
        source = labels[f"source_{note.source}"]
        lines.append(f"> *{source}{f' · {labels["searches"]} {searches}' if searches else ''}*")
        lines += [">", *(f"> {line}" for line in demote(note.text, 4).splitlines())]
        return "\n".join(lines) + "\n"
    if isinstance(event, DossierAdded):
        f = event.file
        return f"*{labels['attached'].format(name=f.name, tokens=f.tokens)}*\n"
    if isinstance(event, VoteTaken):
        return _vote(
            report, event, f"#### {labels['vote_after'].format(round=event.round)}", labels
        )
    if isinstance(event, ErrorEvent):
        return f"> **{labels['error']}** {event.text}\n"
    return _message(report, event, labels)


def _message(report: Report, event: MessageEvent, labels: dict[str, str]) -> str:
    """Render one chat message (a system line in italics)."""
    msg = event.message
    if msg.speaker == "system":
        return f"*{msg.text}*\n"
    flags = [labels[k] for k, on in [("fallback", msg.fallback), ("cut_off", msg.truncated)] if on]
    flags += [labels["to_user"]] if msg.to_user else []
    flag_text = "".join(f" *[{f}]*" for f in flags)
    when = f"{msg.ts.astimezone():%H:%M}"
    return f"**[{when}] \\<{msg.display_nick(report.nick)}\\>**{flag_text} {demote(msg.text, 4)}\n"
