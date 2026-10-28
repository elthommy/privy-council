"""Engine events as chat posts: who posts, what text, split to the platform's limit.

Rule 4 of the M16 security review: no local paths or raw errors reach the channel.
Errors go to the local log and the channel gets a generic line.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass

from privy_council.engine.notices import RESEARCH_SOURCE_LABELS
from privy_council.models.persona import Role
from privy_council.models.session import (
    ErrorEvent,
    Event,
    MessageEvent,
    ResearchAdded,
    RoundStarted,
    VoteTaken,
)
from privy_council.tui.commands import VOTE_TITLE, vote_rows

log = logging.getLogger(__name__)

GENERIC_ERROR = "Something went wrong on the host; the details are in its local log."
HIDDEN_PATH = "(local path hidden)"
# An absolute or home-relative path with at least two parts: /home/x/…, ~/data/…
_PATH = re.compile(r"(?:~|(?<![\w.~]))/[\w.-]+(?:/[\w.-]+)+/?")
_HEADING = re.compile(r"\n(?=#{1,6} )")
# A REPL hint closing a notice: ". /step to continue, /close to wrap up".
_HINT = re.compile(r"(?:[.;:,]\s*|\s+)(?:(?:so|or)\s+)?/(?P<cmd>[a-z]+)\b")


ROLE_LABELS = {
    Role.ADVISOR: "advisor",
    Role.DEVIL_ADVOCATE: "devil's advocate",
    Role.RISK_OFFICER: "risk officer",
    Role.MODERATOR: "moderator",
    Role.SCRIBE: "scribe",
}
QUESTION_LABEL = "question"
MIN_PART = 100  # the smallest piece a turn is cut into
CONTINUED = "*({name}, continued)*"
RoleOf = Callable[[str], Role | None]  # persona slug -> its role, None if unknown


@dataclass(frozen=True)
class Post:
    """One message to send: as a persona (by its display name), or as the bot."""

    text: str
    persona: str | None = None
    question: bool = False  # the moderator asks people something (the front end waits)


def hide_paths(text: str) -> str:
    """Replace local file paths in a notice."""
    return _PATH.sub(HIDDEN_PATH, text)


def chat_notice(text: str, commands: Collection[str]) -> str:
    """Return a notice without its REPL hint, cut at the first command the chat lacks.

    "Next round: Fan, Meta. /step to continue, /close to wrap up" names a command
    nobody in the channel can type; the people who can act see their own commands
    in Discord's popup. A hint naming only chat commands ("or /pause") stays.
    """
    for m in _HINT.finditer(text):
        if m["cmd"] not in commands:
            head = text[: m.start()].rstrip(" ,;:")
            return f"{head}." if head and head[-1] not in ".!?" else head
    return text


def _hard_split(text: str, limit: int) -> list[str]:
    """Cut text into pieces of at most `limit` characters, at spaces when possible."""
    limit = max(limit, MIN_PART)  # a tiny or negative room would never shrink the text
    out: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [*out, text] if text else out


def split_text(text: str, limit: int) -> list[str]:
    """Split a turn under `limit` characters: at headings, then paragraphs, lines, spaces."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    for pattern in (_HEADING, re.compile(r"\n\s*\n"), re.compile(r"\n")):
        parts = [p.strip() for p in pattern.split(text) if p.strip()]
        if len(parts) > 1:
            return _pack(parts, limit, "\n\n" if pattern is not _HEADING else "\n")
    return _hard_split(text, limit)


def _pack(parts: list[str], limit: int, joiner: str) -> list[str]:
    """Join consecutive parts while they fit; split any part that alone is too long."""
    out: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else part
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            out.append(current)
        if len(part) <= limit:
            current = part
        else:
            out += split_text(part, limit)
            current = ""
    return [*out, current] if current else out


def turn_posts(name: str, label: str, text: str, limit: int, question: bool = False) -> list[Post]:
    """Post a turn under its persona, each part headed so the speaker shows in the text too.

    Webhook names are easy to miss (same default avatar, mobile, previews, and a
    split turn's later parts show no name), so the first part opens with
    "**Name** · *role*" and the others with "*(Name, continued)*".
    """
    header = f"**{name}** · *{label}*" if label else f"**{name}**"
    more = CONTINUED.format(name=name)
    room = max(limit - max(len(header), len(more)) - 1, MIN_PART)
    parts = split_text(text, room)
    return [
        Post(f"{header if i == 0 else more}\n{part}", persona=name, question=question)
        for i, part in enumerate(parts)
    ]


def _message_posts(
    event: MessageEvent,
    nick_of: Callable[[str], str],
    role_of: RoleOf,
    limit: int,
    commands: Collection[str],
) -> list[Post]:
    """A persona turn under its name, a notice from the bot; people's own lines aren't echoed."""
    msg = event.message
    if msg.speaker == "user":
        return []  # already in the channel, typed by the person
    if msg.speaker == "system":
        text = chat_notice(hide_paths(msg.text), commands)
        return [Post(p) for p in split_text(text, limit)]
    role = role_of(msg.speaker)
    label = QUESTION_LABEL if msg.to_user else ROLE_LABELS.get(role, "") if role else ""
    return turn_posts(nick_of(msg.speaker), label, msg.text, limit, question=msg.to_user)


def _vote_text(event: VoteTaken, nick_of: Callable[[str], str]) -> str:
    """Render a vote as an aligned code block (Discord has no tables)."""
    rows = vote_rows(event.votes, nick_of)
    width = max(len(nick) for nick, _, _ in rows)
    lines = [f"{nick:<{width}}  {conf:>4}  {pos}" for nick, conf, pos in rows]
    text = VOTE_TITLE.format(round=event.round) + "\n```\n" + "\n".join(lines) + "\n```"
    return hide_paths(text)  # a vote row may carry an error message


def render_event(
    event: Event,
    nick_of: Callable[[str], str],
    moderator: str,
    limit: int,
    role_of: RoleOf = lambda slug: None,
    commands: Collection[str] = frozenset(),
) -> list[Post]:
    """Return the posts for one engine event (most events post nothing).

    `commands`: the chat's own slash commands; notices keep hints naming only those.
    """
    if isinstance(event, MessageEvent):
        return _message_posts(event, nick_of, role_of, limit, commands)
    if isinstance(event, RoundStarted):
        return [Post(f"**— {event.label or f'Round {event.round}'} —**")]
    if isinstance(event, ResearchAdded):
        note = event.note
        source = RESEARCH_SOURCE_LABELS[note.source]
        text = f"**Research ({source}): {note.focus}**\n{note.text}"
        return [Post(p, persona=moderator) for p in split_text(text, limit)]
    if isinstance(event, VoteTaken):
        return [Post(p) for p in split_text(_vote_text(event, nick_of), limit)]
    if isinstance(event, ErrorEvent):
        log.warning("engine error (not shown in the chat): %s", event.text)
        return [Post(GENERIC_ERROR)]
    return []
