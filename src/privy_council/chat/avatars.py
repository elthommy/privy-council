"""Persona avatars for the chat: each persona's seat from the logo, as PNG bytes.

Discord takes a webhook's avatar as image bytes, so nothing is hosted. Drawn in
pure Python (no image library): a coloured disc with a darker pad ring on the
logo's background, like the seats in `docs/assets/council-logo.svg`, smoothed by
drawing at 3x and averaging down. Discord crops avatars to a circle.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

from privy_council.models.persona import Persona, Role

BACKGROUND = "#1e2130"  # the logo's background
SIZE = 256
_SCALE = 3  # supersampling factor
_DISC = 0.40  # seat radius, as a share of the size
_RING = 0.27  # pad ring radius
_RING_WIDTH = 0.045
_RING_SHADE = 0.55  # how far the ring darkens the seat colour toward the background
LOGO = Path("docs/assets/council-logo.png")


def _rgb(color: str) -> tuple[int, int, int]:
    """Parse "#rrggbb"."""
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"not a #rrggbb colour: {color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Blend a toward b by t (0 = a, 1 = b)."""
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def _pixel(
    x: float,
    y: float,
    seat: tuple[int, int, int],
    ring: tuple[int, int, int],
    bg: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the colour at a point, in units of the image size (0..1)."""
    d = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
    if d > _DISC:
        return bg
    if abs(d - _RING) <= _RING_WIDTH / 2:
        return ring
    return seat


def _png(width: int, rows: list[bytes]) -> bytes:
    """Encode RGB rows as a PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        """One PNG chunk with its CRC."""
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, len(rows), 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + row for row in rows)  # filter type 0 per row
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


@lru_cache(maxsize=64)
def seat_avatar(color: str, size: int = SIZE) -> bytes:
    """Return a persona's avatar: its seat colour, as in the logo, as PNG bytes."""
    seat, bg = _rgb(color), _rgb(BACKGROUND)
    ring = _mix(seat, bg, _RING_SHADE)
    big = size * _SCALE
    rows: list[bytes] = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            r = g = b = 0
            for sy in range(_SCALE):
                for sx in range(_SCALE):
                    x = (px * _SCALE + sx + 0.5) / big
                    y = (py * _SCALE + sy + 0.5) / big
                    c = _pixel(x, y, seat, ring, bg)
                    r, g, b = r + c[0], g + c[1], b + c[2]
            n = _SCALE * _SCALE
            row += bytes((round(r / n), round(g / n), round(b / n)))
        rows.append(bytes(row))
    return _png(size, rows)


def logo_avatar(root: Path) -> bytes | None:
    """Return the project logo (the Chair's avatar), or None if it's missing."""
    path = root / LOGO
    return path.read_bytes() if path.is_file() else None


def persona_avatars(personas: Iterable[Persona], root: Path) -> Callable[[str], bytes | None]:
    """Return a lookup from a persona's display name to its avatar.

    Moderators wear the project logo (their seat colour if it's missing); everyone
    else their seat. Unknown names get None (the webhook keeps its default).
    """
    by_name = {p.name: p for p in personas}
    logo = logo_avatar(root)

    def avatar(name: str) -> bytes | None:
        """Return the avatar PNG for one persona name."""
        persona = by_name.get(name)
        if persona is None:
            return None
        if persona.meta.role is Role.MODERATOR and logo is not None:
            return logo
        return seat_avatar(persona.meta.color)

    return avatar
