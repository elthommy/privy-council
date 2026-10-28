"""Start the local SearXNG container when a session needs it, and stop it after.

Through `scripts/searxng.sh`, the one place that knows Podman: fixed arguments, no
shell, nothing typed by a person on the command line. Only a SearXNG on this PC
(localhost) is managed, and only one this process started is stopped: a SearXNG the
owner runs by hand keeps running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from privy_council.backends.base import BackendHealth
from privy_council.config import LocalSearchConfig
from privy_council.models.research import SearchResult
from privy_council.search.base import SearchProvider

log = logging.getLogger(__name__)

SCRIPT = Path("scripts") / "searxng.sh"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
START_TIMEOUT_S = 300.0  # the first start pulls the image
STOP_TIMEOUT_S = 60.0
READY_WAIT_S = 30.0  # after the container starts, until SearXNG answers
POLL_S = 0.5

# (script path, argument, extra env) -> (exit code, output)
RunScript = Callable[[Path, str, dict[str, str], float], Awaitable[tuple[int, str]]]
Probe = Callable[[str], Awaitable[bool]]


async def run_script(
    script: Path, arg: str, env: dict[str, str], timeout: float
) -> tuple[int, str]:
    """Run the script with one argument; return its exit code and output (killed on timeout)."""
    proc = await asyncio.create_subprocess_exec(
        str(script),
        arg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **env},
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, f"no result in {timeout:g} s"
    return proc.returncode or 0, out.decode(errors="replace").strip()


async def healthy(base_url: str) -> bool:
    """Whether SearXNG answers its health check (no search: nothing leaves the PC)."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(base_url.rstrip("/") + "/healthz")
    except httpx.HTTPError:
        return False
    return response.status_code == 200


class SearxngContainer:
    """Starts SearXNG if it's down when a session begins; stops it at exit if it did."""

    def __init__(
        self,
        config: LocalSearchConfig,
        root: Path,
        *,
        run: RunScript = run_script,
        probe: Probe = healthy,
        ready_wait_s: float = READY_WAIT_S,
    ) -> None:
        """Keep the config and the repo root (where `scripts/searxng.sh` is)."""
        self.config = config
        self.script = root / SCRIPT
        self.run = run
        self.probe = probe
        self.ready_wait_s = ready_wait_s
        self.started = False

    @property
    def managed(self) -> bool:
        """Whether this process may start SearXNG: on, opted in, and on this PC."""
        url = urlsplit(self.config.base_url)
        local = url.hostname in LOCAL_HOSTS
        return self.config.enabled and self.config.manage_container and local

    def _env(self) -> dict[str, str]:
        """The script's port: the one `base_url` names."""
        port = urlsplit(self.config.base_url).port
        return {"SEARXNG_PORT": str(port)} if port else {}

    async def ensure_up(self) -> str:
        """Start SearXNG if it's down; return a line for the terminal ("" if nothing to say)."""
        if not self.managed or await self.probe(self.config.base_url):
            return ""
        if not self.script.is_file():
            return f"SearXNG is down and {SCRIPT} is missing: local search will fail."
        code, out = await self.run(self.script, "start", self._env(), START_TIMEOUT_S)
        if code != 0:
            log.warning("searxng start failed (%s): %s", code, out)
            return f"Couldn't start SearXNG ({out or f'exit {code}'}): local search will fail."
        self.started = True
        log.info("searxng started")
        if not await self._ready():
            return f"SearXNG started but isn't answering after {self.ready_wait_s:g} s."
        return "SearXNG started; it stops again when this session ends."

    async def _ready(self) -> bool:
        """Wait until SearXNG answers, up to `ready_wait_s`."""
        deadline = time.monotonic() + self.ready_wait_s
        while time.monotonic() < deadline:
            if await self.probe(self.config.base_url):
                return True
            await asyncio.sleep(POLL_S)
        return await self.probe(self.config.base_url)

    async def stop(self) -> None:
        """Stop SearXNG if this process started it; a failure is only logged."""
        if not self.started:
            return
        self.started = False
        code, out = await self.run(self.script, "stop", self._env(), STOP_TIMEOUT_S)
        if code != 0:
            log.warning("searxng stop failed (%s): %s", code, out)
        else:
            log.info("searxng stopped")


def stopping(
    on_exit: Callable[[], Awaitable[None]], container: SearxngContainer
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Return an exit hook that runs `on_exit`, then stops SearXNG if it was started."""

    async def both() -> None:
        """Close the backends, then the container, even if closing failed."""
        try:
            await on_exit()
        finally:
            await container.stop()

    return both


class StartingSearch:
    """A search provider that starts SearXNG before its first query.

    For normal sessions with `[research] provider = "auto"` or `"local"`: most never
    search locally, so Podman stays untouched unless a search needs it.
    """

    def __init__(self, inner: SearchProvider, container: SearxngContainer) -> None:
        """Wrap `inner`; `container` starts SearXNG (and `stopping` stops it)."""
        self.inner = inner
        self.container = container
        self.name = inner.name
        self.checked = False

    async def search(self, query: str, lang: str) -> list[SearchResult]:
        """Start SearXNG once if it's down, then search."""
        if not self.checked:
            self.checked = True
            if line := await self.container.ensure_up():
                log.warning("%s", line)
        return await self.inner.search(query, lang)

    async def health(self) -> BackendHealth:
        """The wrapped provider's health."""
        return await self.inner.health()
