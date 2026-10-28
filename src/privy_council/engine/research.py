"""Web research by the moderator: when it's allowed, and what it searched.

The claude CLI's own WebSearch/WebFetch tools do the searching (docs/spike-notes.md,
spike 3); in local-only sessions, SearXNG (`engine/local_research.py`). Only the
moderator researches; members see the result as shared facts.
"""

from __future__ import annotations

from enum import StrEnum

from privy_council.backends.base import CompletionResult
from privy_council.config import Settings
from privy_council.models.council import CouncilPreset

WEB_TOOLS = ["WebSearch", "WebFetch"]
SEARCH_TOOL = "WebSearch"
FETCH_TOOL = "WebFetch"


class ResearchMode(StrEnum):
    """Whether a session may research, and whether it does so before the opening."""

    OFF = "off"
    ON_DEMAND = "on_demand"  # `/research` only
    AUTO = "auto"  # before the opening, and `/research`
    REQUIRED = "required"  # AUTO, but the debate waits until that first search succeeds


class ResearchRequiredError(ValueError):
    """A research flag the session can't honour.

    `--no-research` on a council that requires research, or `--research` in
    local-only mode.
    """


def searches_first(mode: ResearchMode) -> bool:
    """Whether the moderator searches before its opening."""
    return mode in (ResearchMode.AUTO, ResearchMode.REQUIRED)


def resolve_mode(
    settings: Settings, council: CouncilPreset, flag: bool | None, local_only: bool = False
) -> ResearchMode:
    """Combine the CLI flag (`--research`/`--no-research`), config and council preset.

    The flag wins; otherwise `[research] enabled` and the council's `auto_research`.
    A council with `require_research` ignores the config and refuses `--no-research`.
    Local-only sessions research only with local search (`[research.local]`, which
    is also the user's consent to queries leaving the PC); then the rules above
    apply. Without it they never research and refuse `--research`;
    `require_research` councils run anyway, with a notice (M15).
    """
    if local_only and not settings.research.local.enabled:
        if flag is True:
            raise ResearchRequiredError(
                "--research in local-only mode needs local search: set "
                "[research.local] enabled = true and start SearXNG (scripts/searxng.sh start)"
            )
        return ResearchMode.OFF
    if council.protocol.require_research:
        if flag is False:
            raise ResearchRequiredError(
                f"council {council.name!r} requires web research; drop --no-research"
            )
        return ResearchMode.REQUIRED
    if flag is True:
        return ResearchMode.AUTO
    if flag is False or not settings.research.enabled:
        return ResearchMode.OFF
    return ResearchMode.AUTO if council.protocol.auto_research else ResearchMode.ON_DEMAND


def searches(result: CompletionResult) -> list[str]:
    """Return the web search queries a research call ran (shown to the user)."""
    calls = result.raw.get("tool_calls") or []
    return [
        str(call["input"]["query"])
        for call in calls
        if call.get("name") == SEARCH_TOOL
        and isinstance(call.get("input"), dict)
        and call["input"].get("query")
    ]


def fetches(result: CompletionResult) -> list[str]:
    """Return the URLs a research call read with WebFetch (shown to the user)."""
    calls = result.raw.get("tool_calls") or []
    return [
        str(call["input"]["url"])
        for call in calls
        if call.get("name") == FETCH_TOOL
        and isinstance(call.get("input"), dict)
        and call["input"].get("url")
    ]
