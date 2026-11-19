"""Local web search: the moderator's queries (strict JSON), the hits, the saved record."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ResearchQueries(BaseModel):
    """What the moderator returns before a local search: the queries to run."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1)  # extra ones are cut by the engine


class ReadPick(BaseModel):
    """A local moderator's choice of a page to read (0: none fits)."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=0)  # a number from the listed pages; out of range means none
    look_for: str = ""


class SearchResult(BaseModel):
    """One web search hit: the snippet is all the moderator reads (no page fetching)."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""
    published: str | None = None  # as the engine gave it, when it gave one


class DocumentRecord(BaseModel):
    """One page read, as saved in `documents_raw.jsonl`, to check its note by hand."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    url: str  # where the page ended up, after redirects
    look_for: str
    passages: list[str]  # what the summary read, best first
    summary: str  # the moderator's summary before the check
    dropped: list[str] = Field(default_factory=list)  # bullets the check removed


class LocalSearchRecord(BaseModel):
    """One local search as saved in `research_raw.jsonl`, to check its note by hand."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    focus: str
    queries: list[str]
    results: list[SearchResult]
    summary: str  # the moderator's summary before the check
    dropped: list[str] = Field(default_factory=list)  # bullets the check removed
