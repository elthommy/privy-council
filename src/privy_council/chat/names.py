"""Debate names for the people in a chat: by user ID, frozen once per session.

Permissions never look at names (rule 7): the ID decides who may do what. The name
only labels a person's lines in the prompts and the transcript, so it must be a
valid `@name` for the engine, distinct from the owner and the personas, and stable
for the session even if the person renames themselves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_INVALID = re.compile(r"[^a-z0-9_-]+")
FALLBACK_NAME = "guest"


def clean_name(username: str) -> str:
    """Turn a Discord username into an `@name` the engine parses whole.

    Usernames are lowercase letters, digits, `_` and `.`; `@name` parsing stops at
    a dot, so "ana.martin" becomes "ana-martin". The first character must be a
    letter or digit.
    """
    name = _INVALID.sub("", username.strip().lower().replace(".", "-"))
    return name.lstrip("_-")[:32] or FALLBACK_NAME


class NameBook:
    """Names given so far in one session, by user ID."""

    def __init__(self, reserved: Iterable[str], given: Mapping[int, str] | None = None) -> None:
        """`reserved`: the owner's and the personas' names; `given`: names from a resume."""
        self.reserved = {r.casefold() for r in reserved}
        self.by_id: dict[int, str] = dict(given or {})

    def name_for(self, user_id: int, username: str, override: str | None = None) -> str:
        """Return this person's name, giving one (suffixed on a collision) the first time."""
        if user_id in self.by_id:
            return self.by_id[user_id]
        base = clean_name(override) if override else clean_name(username)
        taken = self.reserved | {n.casefold() for n in self.by_id.values()}
        name, n = base, 1
        while name.casefold() in taken:
            n += 1
            name = f"{base}-{n}"
        self.by_id[user_id] = name
        return name

    def user_of(self, name: str) -> int | None:
        """Return the user ID behind a name, or None."""
        folded = name.casefold()
        return next((uid for uid, n in self.by_id.items() if n.casefold() == folded), None)
