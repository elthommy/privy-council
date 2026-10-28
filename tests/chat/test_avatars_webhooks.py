"""Persona avatars (pure-Python PNGs) and the per-persona webhook pool, with fakes."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from privy_council.chat.avatars import BACKGROUND, SIZE, persona_avatars, seat_avatar
from privy_council.chat.webhooks import WebhookPool, hook_name
from privy_council.models.persona import Persona
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import personas as personas


def pixels(png: bytes) -> tuple[int, int, bytes]:
    """Decode our own RGB PNGs (filter 0 rows): width, height, raw rows."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    idat = png.index(b"IDAT")
    size = struct.unpack(">I", png[idat - 4 : idat])[0]
    return width, height, zlib.decompress(png[idat + 4 : idat + 4 + size])


def at(raw: bytes, width: int, x: int, y: int) -> str:
    """Return the colour at (x, y) as #rrggbb."""
    i = y * (1 + 3 * width) + 1 + 3 * x
    return "#{:02x}{:02x}{:02x}".format(*raw[i : i + 3])


def test_seat_avatar_is_the_seat_on_the_logo_background() -> None:
    width, height, raw = pixels(seat_avatar("#f0c674"))
    assert (width, height) == (SIZE, SIZE)
    assert at(raw, width, SIZE // 2, SIZE // 2) == "#f0c674"  # the seat
    assert at(raw, width, 2, 2) == BACKGROUND  # outside the disc
    ring = at(raw, width, SIZE // 2 + int(0.27 * SIZE), SIZE // 2)
    assert ring not in ("#f0c674", BACKGROUND)  # the darker pad ring


def test_moderators_wear_the_logo_others_their_seat(personas: dict[str, Persona]) -> None:
    avatar = persona_avatars(personas.values(), REPO_ROOT)
    assert avatar("Chair") == (REPO_ROOT / "docs/assets/council-logo.png").read_bytes()
    assert avatar("Rigor") == seat_avatar(personas["rigor"].meta.color)
    assert avatar("Nobody") is None


def test_without_the_logo_the_moderator_gets_its_seat(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    avatar = persona_avatars(personas.values(), tmp_path)
    assert avatar("Chair") == seat_avatar(personas["chair"].meta.color)


@dataclass
class FakeHook:
    """A webhook: its name and avatar."""

    name: str | None
    avatar: bytes | None = None


@dataclass
class FakeChannel:
    """Records every webhook change; `fail` makes create raise."""

    hooks: list[FakeHook] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    fail: bool = False

    async def list_hooks(self) -> list[FakeHook]:
        """Return the hooks."""
        self.log.append("list")
        return list(self.hooks)

    async def create_hook(self, name: str, avatar: bytes | None) -> FakeHook:
        """Create one (or fail)."""
        if self.fail:
            raise RuntimeError("Missing Permissions")
        hook = FakeHook(name, avatar)
        self.hooks.append(hook)
        self.log.append(f"create {name}")
        return hook

    async def set_avatar(self, hook: FakeHook, avatar: bytes) -> None:
        """Change an avatar."""
        hook.avatar = avatar
        self.log.append(f"avatar {hook.name}")

    async def delete_hook(self, hook: FakeHook) -> None:
        """Delete one."""
        self.hooks.remove(hook)
        self.log.append(f"delete {hook.name}")


def avatar_of(name: str) -> bytes:
    """A distinct fake avatar per persona."""
    return name.encode()


async def test_pool_creates_once_with_the_avatar_then_reuses() -> None:
    channel = FakeChannel()
    pool = WebhookPool(channel, avatar_of)
    first = await pool.hook_for("Saver")
    assert first is not None and first.name == "council-saver" and first.avatar == b"Saver"
    assert await pool.hook_for("Saver") is first
    assert channel.log == ["list", "create council-saver"]  # one listing, one creation


async def test_existing_hooks_get_their_avatar_refreshed_once_per_run() -> None:
    old = FakeHook("council-saver", b"old colour")
    channel = FakeChannel([old, FakeHook("council")])
    pool = WebhookPool(channel, avatar_of)
    assert await pool.hook_for("Saver") is old
    await pool.hook_for("Saver")
    assert old.avatar == b"Saver" and channel.log == ["list", "avatar council-saver"]


async def test_a_full_channel_loses_its_least_recently_used_council_hook() -> None:
    others = [FakeHook("someone-else"), FakeHook("council")]
    channel = FakeChannel(others + [FakeHook(hook_name(f"P{i}")) for i in range(3)])
    pool = WebhookPool(channel, avatar_of, limit=5)
    for name in ("P1", "P0", "P2"):  # P1 is now the least recently used
        await pool.hook_for(name)
    await pool.hook_for("New")
    assert "delete council-p1" in channel.log
    names = {h.name for h in channel.hooks}
    assert {"someone-else", "council", "council-new"} <= names and len(names) == 5


async def test_pool_never_deletes_other_hooks_and_falls_back() -> None:
    channel = FakeChannel([FakeHook(f"other-{i}") for i in range(5)])
    pool = WebhookPool(channel, avatar_of, limit=5)
    assert await pool.hook_for("Saver") is None  # the adapter uses its shared webhook
    assert len(channel.hooks) == 5 and not any(e.startswith("delete") for e in channel.log)


async def test_pool_falls_back_on_platform_errors() -> None:
    pool = WebhookPool(FakeChannel(fail=True), avatar_of)
    assert await pool.hook_for("Saver") is None
