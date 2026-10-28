"""Which chat thread holds which session, and the names given there (for resume)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

THREADS_JSON = "threads.json"


class ChatThread(BaseModel):
    """One session's thread and the people's names in it, by user ID."""

    thread: int
    names: dict[int, str] = Field(default_factory=dict)


class ThreadIndex(BaseModel):
    """All chat sessions' threads, by session ID."""

    sessions: dict[str, ChatThread] = Field(default_factory=dict)


class ThreadStore:
    """`threads.json` in the chat data directory, rewritten atomically."""

    def __init__(self, directory: Path) -> None:
        """Load the index, or start an empty one."""
        self.path = directory / THREADS_JSON
        self.index = (
            ThreadIndex.model_validate_json(self.path.read_text(encoding="utf-8"))
            if self.path.is_file()
            else ThreadIndex()
        )

    def get(self, session_id: str) -> ChatThread | None:
        """Return a session's thread, or None."""
        return self.index.sessions.get(session_id)

    def put(self, session_id: str, entry: ChatThread) -> None:
        """Record a session's thread and names, and save."""
        self.index.sessions[session_id] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self.index.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
