"""Local web search: the moderator's queries (strict JSON), the hits, the saved record."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ResearchQueries(BaseModel):
    """What the moderator returns before a local search: the queries to run."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1)  # extra ones are cut by the engine


class SearchResult(BaseModel):
    """One web search hit: the snippet is all the moderator reads (no page fetching)."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""
    published: str | None = None  # as the engine gave it, when it gave one


class LocalSearchRecord(BaseModel):
    """One local search as saved in `research_raw.jsonl`, to check its note by hand."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    focus: str
    queries: list[str]
    results: list[SearchResult]
    summary: str  # the moderator's summary before the check
    dropped: list[str] = Field(default_factory=list)  # bullets the check removed
