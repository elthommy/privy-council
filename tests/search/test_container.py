"""Starting and stopping SearXNG for a session (scripts/searxng.sh faked, no Podman)."""

from __future__ import annotations

from pathlib import Path

import pytest

from privy_council.config import LocalSearchConfig
from privy_council.search.container import SCRIPT, SearxngContainer, StartingSearch, stopping


class Script:
    """A fake scripts/searxng.sh: records calls; `start` brings SearXNG up."""

    def __init__(self, up: bool = False, code: int = 0) -> None:
        """SearXNG up or down at first; the exit code every call returns."""
        self.up = up
        self.code = code
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def run(
        self, script: Path, arg: str, env: dict[str, str], timeout: float
    ) -> tuple[int, str]:
        """Record the call; a successful start brings SearXNG up."""
        assert script.name == "searxng.sh"
        self.calls.append((arg, env))
        if arg == "start" and self.code == 0:
            self.up = True
        return self.code, "" if self.code == 0 else "podman not found: install it first"

    async def probe(self, base_url: str) -> bool:
        """Answer the health check."""
        return self.up


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repo root with the script in place."""
    (tmp_path / SCRIPT).parent.mkdir(parents=True)
    (tmp_path / SCRIPT).write_text("#!/bin/sh\n")
    return tmp_path


def container(root: Path, script: Script, **config: object) -> SearxngContainer:
    """A container manager over the fake script."""
    local = LocalSearchConfig.model_validate({"enabled": True, **config})
    return SearxngContainer(local, root, run=script.run, probe=script.probe, ready_wait_s=0.05)


async def test_starts_when_down_and_stops_what_it_started(root: Path) -> None:
    script = Script()
    c = container(root, script, base_url="http://127.0.0.1:8899")
    assert await c.ensure_up() == "SearXNG started; it stops again when this session ends."
    await stopping(_nothing, c)()
    assert script.calls == [("start", {"SEARXNG_PORT": "8899"}), ("stop", {"SEARXNG_PORT": "8899"})]


async def test_a_searxng_already_running_is_left_alone(root: Path) -> None:
    script = Script(up=True)
    c = container(root, script)
    assert await c.ensure_up() == ""
    await c.stop()
    assert script.calls == []


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": False},
        {"manage_container": False},
        {"base_url": "https://search.example.org"},  # not on this PC
    ],
)
async def test_nothing_is_started_when_not_managed(root: Path, config: dict[str, object]) -> None:
    script = Script()
    c = container(root, script, **config)
    assert await c.ensure_up() == ""
    assert script.calls == []


async def test_a_failed_start_says_why_and_stops_nothing(root: Path) -> None:
    script = Script(code=1)
    c = container(root, script)
    line = await c.ensure_up()
    assert "Couldn't start SearXNG (podman not found" in line
    await c.stop()
    assert [arg for arg, _ in script.calls] == ["start"]


async def test_started_but_not_answering(root: Path) -> None:
    script = Script()
    c = container(root, script)

    async def never(base_url: str) -> bool:
        """SearXNG never answers."""
        return False

    c.probe = never
    assert "isn't answering after" in await c.ensure_up()
    await c.stop()
    assert [arg for arg, _ in script.calls] == ["start", "stop"]  # it did start the container


async def test_missing_script(tmp_path: Path) -> None:
    script = Script()
    assert "is missing" in await container(tmp_path, script).ensure_up()
    assert script.calls == []


async def test_the_container_stops_even_if_closing_the_backends_fails(root: Path) -> None:
    script = Script()
    c = container(root, script)
    await c.ensure_up()

    async def broken() -> None:
        """Closing the backends fails."""
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await stopping(broken, c)()
    assert script.calls[-1][0] == "stop"


async def _nothing() -> None:
    """An exit hook with nothing to close."""


async def test_starting_search_starts_searxng_once_before_the_first_query(root: Path) -> None:
    from tests.engine.test_local_research import FakeSearch

    script = Script()
    inner = FakeSearch()
    search = StartingSearch(inner, container(root, script))
    await search.search("a", "en")
    await search.search("b", "en")
    assert [call for call, _ in script.calls] == ["start"]
    assert [q for q, _ in inner.queries] == ["a", "b"]
