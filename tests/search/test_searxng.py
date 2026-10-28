"""SearXNG client with pytest-httpx: parsing, typed errors, health."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from privy_council.backends.base import (
    BackendOutputError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from privy_council.config import LocalSearchConfig
from privy_council.search.searxng import (
    JSON_HINT,
    MAX_SNIPPET_CHARS,
    SearxngProvider,
    parse_results,
)

BASE = "http://127.0.0.1:8888"


def hit(n: int) -> dict[str, object]:
    """One SearXNG result."""
    return {"title": f"Title  {n}", "url": f"https://example.org/{n}", "content": f"Fact {n}."}


@pytest.fixture
def provider() -> SearxngProvider:
    """A provider with 2 results per query."""
    return SearxngProvider(LocalSearchConfig(enabled=True, base_url=BASE, results_per_query=2))


async def test_search_sends_query_and_parses_hits(
    provider: SearxngProvider, httpx_mock: HTTPXMock
) -> None:
    results = [{**hit(1), "publishedDate": "2026-10-20T08:00:00"}, hit(2), hit(3)]
    httpx_mock.add_response(json={"results": results})
    found = await provider.search("ecb deposit rate", "fr")
    request = httpx_mock.get_request()
    assert request is not None
    assert request.url.params["q"] == "ecb deposit rate"
    assert request.url.params["format"] == "json"
    assert request.url.params["language"] == "fr"
    assert [r.url for r in found] == ["https://example.org/1", "https://example.org/2"]
    assert found[0].title == "Title 1" and found[0].published == "2026-10-20"
    assert found[1].published is None


def test_parse_skips_hits_without_url_and_trims_snippets() -> None:
    long = {"title": "t", "url": "https://x.org", "content": "word " * 200}
    found = parse_results({"results": [{"title": "no url"}, "junk", long]}, 5)
    assert len(found) == 1
    assert len(found[0].snippet) == MAX_SNIPPET_CHARS


@pytest.mark.parametrize("payload", [[], {"results": "none"}, {"answers": []}])
def test_parse_rejects_a_reply_without_results(payload: object) -> None:
    with pytest.raises(BackendOutputError):
        parse_results(payload, 5)


async def test_json_format_disabled_gives_the_hint(
    provider: SearxngProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=403)
    with pytest.raises(BackendUnavailableError, match=r"search\.formats"):
        await provider.search("x", "en")


async def test_unreachable_server_gives_the_start_hint(
    provider: SearxngProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    with pytest.raises(BackendUnavailableError, match=r"scripts/searxng\.sh start"):
        await provider.search("x", "en")


async def test_timeout_and_bad_json(provider: SearxngProvider, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))
    with pytest.raises(BackendTimeoutError):
        await provider.search("x", "en")
    httpx_mock.add_response(text="<html>")
    with pytest.raises(BackendOutputError, match="not JSON"):
        await provider.search("x", "en")
    httpx_mock.add_response(status_code=500)
    with pytest.raises(BackendUnavailableError, match="HTTP 500"):
        await provider.search("x", "en")


async def test_health_reports_dead_engines(
    provider: SearxngProvider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        json={"results": [hit(1)], "unresponsive_engines": [["google", "CAPTCHA"]]}
    )
    health = await provider.health()
    assert health.ok and health.details == [f"SearXNG at {BASE}"]
    assert health.warnings == ["engines not answering: google"]


async def test_health_fails_with_the_fix(provider: SearxngProvider, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=403)
    health = await provider.health()
    assert not health.ok and health.problems == [JSON_HINT]
