"""Text of a fetched PDF for the document reader, extracted in a capped child process.

A PDF is untrusted: parsing one can loop or eat memory, and it would block the event
loop. The child (`python -I -m privy_council.search.pdf_text`) gets the bytes on stdin
and an empty environment, writes no file, and runs under memory and CPU limits.
"""

from __future__ import annotations

import asyncio
import io
import resource
import sys

from privy_council.backends.base import BackendOutputError, BackendTimeoutError

TIMEOUT_S = 90.0  # a 119-page filing takes about 4 s
MEMORY_BYTES = 1_500_000_000
MAX_PAGES = 600
MAX_CHARS = 3_000_000


async def pdf_text(body: bytes, timeout_s: float = TIMEOUT_S) -> str:
    """Return a PDF's text; a file that can't be read raises a typed error."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "privy_council.search.pdf_text",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={},
    )
    try:
        async with asyncio.timeout(timeout_s):
            out, _ = await process.communicate(body)
    except TimeoutError as exc:
        raise BackendTimeoutError(f"the PDF took over {timeout_s:g} s to read") from exc
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    text = out.decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise BackendOutputError("the PDF can't be read (damaged, locked or too large)")
    if not text.strip():
        raise BackendOutputError("the PDF holds no text (a scan?)")
    return text


def extract(body: bytes) -> str:
    """Return the text of a PDF's pages in order, up to MAX_PAGES and MAX_CHARS."""
    from pypdf import PdfReader  # only the child needs it

    parts: list[str] = []
    size = 0
    for page in PdfReader(io.BytesIO(body)).pages[:MAX_PAGES]:
        text = page.extract_text()
        parts.append(text)
        size += len(text)
        if size >= MAX_CHARS:
            break
    return "\n".join(parts)[:MAX_CHARS]


def _limit() -> None:
    """Cap this process: memory, CPU time, and no file writes."""
    cpu = int(TIMEOUT_S)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def main() -> None:
    """Read a PDF on stdin and write its text on stdout (the child process)."""
    _limit()
    text = extract(sys.stdin.buffer.read())
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
