"""Fetch one public web page for the document reader (M21), behind a URL guard.

The guard: http(s) only, no credentials in the URL, and every hop's host must
resolve to public addresses only (no private, loopback, link-local or reserved
ones). Redirects are followed by hand and re-checked. The body is streamed with a
size cap, the whole fetch has a time cap, and only HTML, plain text and PDF are read
(a PDF's text is extracted in a capped child process, `search/pdf_text.py`).

The request carries only the URL: no cookies (a fresh client per hop), no referer,
no proxy or netrc settings from the environment.

Accepted risk (security model): DNS can answer differently between the check and
the request (rebinding); pinning the checked address would close it.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from urllib.parse import urljoin, urlsplit

import httpx

from privy_council.backends.base import (
    BackendOutputError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from privy_council.config import DocumentsConfig
from privy_council.search.pdf_text import pdf_text

log = logging.getLogger(__name__)

PDF_TYPE = "application/pdf"
MAX_REDIRECTS = 5
READABLE_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain", PDF_TYPE})
ACCEPT = "text/html, text/plain, application/pdf"
Resolver = Callable[[str, int], Awaitable[list[str]]]


class DocumentRefused(BackendOutputError):
    """The page can't be read: the guard refused it, or it is not text."""


@dataclass(frozen=True)
class FetchedPage:
    """A fetched page: where it ended up (after redirects), its type and its text."""

    url: str
    content_type: str
    text: str


async def resolve_host(host: str, port: int) -> list[str]:
    """Return the addresses a host name resolves to."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise BackendUnavailableError(f"can't resolve {host}") from exc
    return [str(info[4][0]) for info in infos]


NAT64 = ipaddress.ip_network("64:ff9b::/96")


def check_address(address: str) -> None:
    """Refuse an address that isn't public (private, loopback, link-local, reserved...).

    An IPv6 address that carries an IPv4 one (mapped, 6to4, Teredo, NAT64) is also
    judged by that IPv4 address, which a gateway may reach on the owner's network.
    """
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    for candidate in (ip, *_embedded(ip)):
        if not candidate.is_global or candidate.is_multicast:
            raise DocumentRefused(f"refused: {address} is not a public address")


def _embedded(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """Return the IPv4 addresses an IPv6 address carries, if any."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return []
    out = [a for a in (ip.ipv4_mapped, ip.sixtofour) if a is not None]
    if ip.teredo is not None:
        out += list(ip.teredo)
    if ip in NAT64:
        out.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return out


async def check_url(url: str, resolve: Resolver = resolve_host) -> None:
    """Refuse a URL the reader must not fetch; resolve its host and check every address."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise DocumentRefused(f"refused: only http(s) pages can be read, not {url}")
    if parts.username or parts.password:
        raise DocumentRefused("refused: a URL with credentials")
    host = parts.hostname
    if not host:
        raise DocumentRefused(f"refused: no host in {url}")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise DocumentRefused(f"refused: bad port in {url}") from exc
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = await resolve(host, port)
    if not addresses:
        raise DocumentRefused(f"refused: {host} has no address")
    for address in addresses:
        check_address(address)


async def fetch_page(
    url: str, config: DocumentsConfig, resolve: Resolver = resolve_host
) -> FetchedPage:
    """Fetch a page through the guard, following up to MAX_REDIRECTS redirects."""
    try:
        async with asyncio.timeout(config.timeout_s):
            page, body = await _follow(url, config, resolve)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise BackendTimeoutError(f"the page took over {config.timeout_s:g} s: {url}") from exc
    except httpx.TransportError as exc:
        raise BackendUnavailableError(f"can't reach {url}") from exc
    if page.content_type == PDF_TYPE:  # outside the fetch's time cap: it has its own
        return replace(page, text=await pdf_text(body))
    return page


async def _follow(
    url: str, config: DocumentsConfig, resolve: Resolver
) -> tuple[FetchedPage, bytes]:
    """Fetch hop by hop: check each URL, then request it with a fresh client."""
    for _ in range(MAX_REDIRECTS + 1):
        await check_url(url, resolve)
        log.info("document fetch: %s", url)
        headers = {"User-Agent": config.user_agent, "Accept": ACCEPT}
        async with (
            httpx.AsyncClient(
                headers=headers, follow_redirects=False, trust_env=False, timeout=config.timeout_s
            ) as client,
            client.stream("GET", url) as response,
        ):
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            return await _read(url, response, config.max_bytes)
    raise DocumentRefused(f"refused: more than {MAX_REDIRECTS} redirects")


async def _read(url: str, response: httpx.Response, max_bytes: int) -> tuple[FetchedPage, bytes]:
    """Read a response's body, refusing other types and oversized bodies.

    A PDF comes back as bytes with no text yet.
    """
    if response.status_code != 200:
        raise BackendUnavailableError(f"HTTP {response.status_code} for {url}")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in READABLE_TYPES:
        raise DocumentRefused(f"refused: {content_type or 'no type'} is not HTML, text or PDF")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise DocumentRefused(f"refused: the page is over {max_bytes:,} bytes")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > max_bytes:
            raise DocumentRefused(f"refused: the page is over {max_bytes:,} bytes")
    if content_type == PDF_TYPE:
        return FetchedPage(url=url, content_type=content_type, text=""), bytes(body)
    try:
        text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
    except LookupError:  # a charset Python doesn't know
        text = bytes(body).decode("utf-8", errors="replace")
    return FetchedPage(url=url, content_type=content_type, text=text), b""
