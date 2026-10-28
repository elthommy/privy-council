"""SearXNG's JSON API (self-hosted, run with `scripts/searxng.sh` in Podman)."""

from __future__ import annotations

from typing import Any

import httpx

from privy_council.backends.base import (
    BackendHealth,
    BackendOutputError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from privy_council.config import LocalSearchConfig
from privy_council.search.base import SearchResult

START_HINT = "start it with scripts/searxng.sh start (needs Podman)"
JSON_HINT = "SearXNG refused JSON output: add json to search.formats in its settings.yml"
MAX_SNIPPET_CHARS = 400


def parse_results(payload: Any, limit: int) -> list[SearchResult]:
    """Turn a SearXNG JSON reply into results, skipping hits without a URL."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise BackendOutputError("SearXNG: no results list in the reply")
    out: list[SearchResult] = []
    for hit in payload["results"]:
        if not isinstance(hit, dict) or not isinstance(hit.get("url"), str):
            continue
        snippet = " ".join(str(hit.get("content") or "").split())
        published = hit.get("publishedDate")
        out.append(
            SearchResult(
                title=" ".join(str(hit.get("title") or "").split()),
                url=hit["url"],
                snippet=snippet[:MAX_SNIPPET_CHARS],
                published=str(published)[:10] if published else None,
            )
        )
        if len(out) >= limit:
            break
    return out


class SearxngProvider:
    """Queries one SearXNG instance; a short-lived client per call (a few per session)."""

    name = "searxng"

    def __init__(self, config: LocalSearchConfig) -> None:
        """Keep the config; nothing connects until a search."""
        self.config = config

    async def _get(self, query: str, lang: str) -> Any:
        """Run one query and return the decoded JSON, raising typed errors."""
        params = {"q": query, "format": "json", "language": lang}
        url = self.config.base_url.rstrip("/") + "/search"
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"SearXNG: no answer in {self.config.timeout_s:g} s") from exc
        except httpx.TransportError as exc:
            raise BackendUnavailableError(
                f"SearXNG not reachable at {self.config.base_url}: {START_HINT}"
            ) from exc
        if response.status_code == 403:
            raise BackendUnavailableError(JSON_HINT)
        if response.status_code != 200:
            raise BackendUnavailableError(f"SearXNG: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise BackendOutputError("SearXNG: the reply is not JSON") from exc

    async def search(self, query: str, lang: str) -> list[SearchResult]:
        """Return up to `results_per_query` hits for one query."""
        return parse_results(await self._get(query, lang), self.config.results_per_query)

    async def health(self) -> BackendHealth:
        """Run one test query; problems carry the fix (start it, enable JSON)."""
        health = BackendHealth(backend="local search", ok=True)
        try:
            payload = await self._get("council doctor", "en")
            parse_results(payload, 1)
        except (BackendUnavailableError, BackendTimeoutError, BackendOutputError) as exc:
            health.ok = False
            health.problems.append(str(exc))
            return health
        health.details.append(f"SearXNG at {self.config.base_url}")
        dead = payload.get("unresponsive_engines") or []
        if dead:
            names = ", ".join(str(e[0]) if isinstance(e, list) and e else str(e) for e in dead)
            health.warnings.append(f"engines not answering: {names}")
        return health
