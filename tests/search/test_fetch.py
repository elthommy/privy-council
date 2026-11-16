"""The document reader's URL guard and fetch (M21): no real network, DNS faked."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from privy_council.config import DocumentsConfig
from privy_council.search.fetch import DocumentRefused, check_url, fetch_page

PUBLIC = "93.184.215.14"
ADDRESSES = {
    "example.org": [PUBLIC],
    "intranet.example.org": ["192.168.1.20"],
    "mixed.example.org": [PUBLIC, "10.0.0.5"],
}


async def fake_resolve(host: str, port: int) -> list[str]:
    """Resolve from the table above; an unknown name has no address."""
    return ADDRESSES.get(host, [])


def config(**changes: object) -> DocumentsConfig:
    """Reader settings for tests."""
    return DocumentsConfig(enabled=True, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://10.1.2.3/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]:8080/",
        "http://[::ffff:192.168.0.1]/",
        "http://[2002:7f00:1::1]/",
        "http://[64:ff9b::a00:1]/",
        "http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/",
        "http://0.0.0.0/",
        "http://intranet.example.org/",
        "http://mixed.example.org/",
        "http://unknown.example.org/",
        "file:///etc/passwd",
        "ftp://example.org/file.txt",
        "http://user:secret@example.org/",
    ],
)
async def test_guard_refuses(url: str) -> None:
    with pytest.raises(DocumentRefused):
        await check_url(url, fake_resolve)


async def test_guard_allows_a_public_host() -> None:
    await check_url("https://example.org/filing.htm", fake_resolve)


async def test_fetch_sends_only_the_url(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://example.org/a",
        status_code=302,
        headers={"location": "/b", "set-cookie": "track=1"},
    )
    httpx_mock.add_response(
        url="https://example.org/b",
        headers={"content-type": "text/html; charset=utf-8"},
        text="<p>Debt due 2027: 1,234</p>",
    )
    page = await fetch_page("https://example.org/a", config(), fake_resolve)
    assert page.url == "https://example.org/b"
    assert "1,234" in page.text
    first, second = httpx_mock.get_requests()
    assert second.headers["user-agent"] == config().user_agent
    for request in (first, second):
        assert "cookie" not in request.headers
        assert "referer" not in request.headers


async def test_a_redirect_to_a_private_host_is_refused(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://example.org/a",
        status_code=301,
        headers={"location": "http://intranet.example.org/secret"},
    )
    with pytest.raises(DocumentRefused, match="not a public address"):
        await fetch_page("https://example.org/a", config(), fake_resolve)
    assert len(httpx_mock.get_requests()) == 1  # the private host was never contacted


async def test_too_many_redirects(httpx_mock: HTTPXMock) -> None:
    for i in range(6):
        httpx_mock.add_response(
            url=f"https://example.org/{i}", status_code=302, headers={"location": f"/{i + 1}"}
        )
    with pytest.raises(DocumentRefused, match="redirects"):
        await fetch_page("https://example.org/0", config(), fake_resolve)


async def test_an_oversize_body_is_refused(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://example.org/big", headers={"content-type": "text/plain"}, text="x" * 5000
    )
    with pytest.raises(DocumentRefused, match="over 1,000 bytes"):
        await fetch_page("https://example.org/big", config(max_bytes=1000), fake_resolve)


async def test_a_pdf_is_refused(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://example.org/f.pdf", headers={"content-type": "application/pdf"}, content=b"%"
    )
    with pytest.raises(DocumentRefused, match="application/pdf"):
        await fetch_page("https://example.org/f.pdf", config(), fake_resolve)
