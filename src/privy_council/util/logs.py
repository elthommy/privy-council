"""Log file setup shared by the commands that run debates."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(data_dir: Path, *, warnings_from: tuple[str, ...] = ()) -> None:
    """Log to data/logs/council.log (INFO: no prompt content is logged at INFO).

    `warnings_from`: libraries whose warnings and errors go to the file too ("discord":
    a failed slash command is otherwise only a traceback in the terminal).
    """
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "council.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("privy_council")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    for name in warnings_from:
        library = logging.getLogger(name)
        library.setLevel(logging.WARNING)
        library.addHandler(handler)
