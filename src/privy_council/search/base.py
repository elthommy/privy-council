"""What a search provider returns, and the interface the engine uses."""

from __future__ import annotations

from typing import Protocol

from privy_council.backends.base import BackendHealth
from privy_council.models.research import SearchResult

__all__ = ["SearchProvider", "SearchResult"]


class SearchProvider(Protocol):
    """A web search engine; errors are the typed `BackendError`s."""

    name: str

    async def search(self, query: str, lang: str) -> list[SearchResult]:
        """Return the hits for one query, best first."""
        ...

    async def health(self) -> BackendHealth:
        """Check the provider answers queries."""
        ...
