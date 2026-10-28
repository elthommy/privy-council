"""One webhook per persona, each with its own avatar, within the platform's limit.

Webhooks are named `council-<persona>` and reused across debates and bot runs.
The first post of each run refreshes the avatar (a persona may have changed
colour). Discord allows 15 webhooks per channel: past that, the least recently
used `council-*` webhook is deleted; no other webhook is ever touched. Any error
returns None, and the adapter posts through its shared webhook instead: a debate
never stalls over an avatar.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from typing import Protocol

from privy_council.chat.names import clean_name

log = logging.getLogger(__name__)

PREFIX = "council-"
CHANNEL_LIMIT = 15  # Discord's webhooks per channel


class NamedHook(Protocol):
    """A webhook as the pool sees it."""

    @property
    def name(self) -> str | None:
        """The webhook's own name."""
        ...


class HookChannel[H: NamedHook](Protocol):
    """What the pool needs from the platform."""

    async def list_hooks(self) -> list[H]:
        """Return every webhook in the council channel."""
        ...

    async def create_hook(self, name: str, avatar: bytes | None) -> H:
        """Create a webhook with an avatar."""
        ...

    async def set_avatar(self, hook: H, avatar: bytes) -> None:
        """Replace a webhook's avatar."""
        ...

    async def delete_hook(self, hook: H) -> None:
        """Delete a webhook."""
        ...


def hook_name(persona: str) -> str:
    """Return the webhook name for a persona's display name."""
    return PREFIX + clean_name(persona)


class WebhookPool[H: NamedHook]:
    """Finds or makes each persona's webhook; None means "use the shared one"."""

    def __init__(
        self,
        channel: HookChannel[H],
        avatar_for: Callable[[str], bytes | None],
        limit: int = CHANNEL_LIMIT,
    ) -> None:
        """`avatar_for` maps a persona's display name to its avatar (None: none)."""
        self.channel = channel
        self.avatar_for = avatar_for
        self.limit = limit
        self._known: dict[str, H] | None = None  # the channel's webhooks, by name
        self._refreshed: set[str] = set()
        self._clock = itertools.count()
        self._used: dict[str, int] = {}

    async def hook_for(self, persona: str) -> H | None:
        """Return the persona's webhook, creating or refreshing it; None on any error."""
        name = hook_name(persona)
        try:
            hook = await self._find_or_create(name, persona)
        except Exception:  # the platform's errors are many; fall back, never stall
            log.warning("persona webhook for %s unavailable; using the shared one", persona)
            return None
        self._used[name] = next(self._clock)
        return hook

    async def _find_or_create(self, name: str, persona: str) -> H:
        """Return the named webhook, refreshing its avatar once per run."""
        known = await self._webhooks()
        hook = known.get(name)
        if hook is None:
            await self._make_room(known)
            hook = await self.channel.create_hook(name, self.avatar_for(persona))
            known[name] = hook
        elif name not in self._refreshed:
            avatar = self.avatar_for(persona)
            if avatar is not None:
                await self.channel.set_avatar(hook, avatar)
        self._refreshed.add(name)
        return hook

    async def _webhooks(self) -> dict[str, H]:
        """Load the channel's webhooks once."""
        if self._known is None:
            self._known = {h.name: h for h in await self.channel.list_hooks() if h.name}
        return self._known

    async def _make_room(self, known: dict[str, H]) -> None:
        """Delete the least recently used `council-*` webhook while the channel is full."""
        while len(known) >= self.limit:
            ours = [n for n in known if n.startswith(PREFIX)]
            if not ours:
                raise RuntimeError("the channel's webhook limit is taken by others")
            oldest = min(ours, key=lambda n: self._used.get(n, -1))
            await self.channel.delete_hook(known.pop(oldest))
            self._used.pop(oldest, None)
            self._refreshed.discard(oldest)
