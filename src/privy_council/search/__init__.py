"""Web search outside Claude, for the moderator's research in local-only sessions."""

from privy_council.search.base import SearchProvider, SearchResult
from privy_council.search.searxng import SearxngProvider

__all__ = ["SearchProvider", "SearchResult", "SearxngProvider"]
