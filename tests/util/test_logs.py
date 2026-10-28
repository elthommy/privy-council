"""The log file setup."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from privy_council.util.logs import setup_logging


@pytest.fixture
def clean_loggers() -> Iterator[None]:
    """Remove the handlers the test adds, so other tests don't write to its file."""
    names = ("privy_council", "test-library")
    before = {n: list(logging.getLogger(n).handlers) for n in names}
    yield
    for name in names:
        logger = logging.getLogger(name)
        for handler in set(logger.handlers) - set(before[name]):
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.usefixtures("clean_loggers")
def test_a_librarys_warnings_reach_the_file_but_not_its_info(tmp_path: Path) -> None:
    setup_logging(tmp_path, warnings_from=("test-library",))
    library = logging.getLogger("test-library")
    library.info("connected")
    library.warning("Ignoring exception in command 'offline'")
    logging.getLogger("privy_council.chat").info("chat command /offline from 1 (owner)")
    text = (tmp_path / "logs" / "council.log").read_text()
    assert "Ignoring exception in command 'offline'" in text
    assert "chat command /offline" in text
    assert "connected" not in text
