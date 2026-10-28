"""The optional judge: one Claude call comparing the variants' syntheses (`--judge`).

The judge sees the variants under letters, never their models or rosters, so
its verdict can't lean on a model's reputation; the report maps letters back.
"""

from __future__ import annotations

import string

from privy_council.backends.base import Backend, CallTag, ChatMessage, CompletionRequest
from privy_council.engine import context, prompts
from privy_council.models.session import ResearchNote
from privy_council.util.language import LANGUAGE_NAMES, Language

JUDGE_PURPOSE = "compare_judge"
JUDGE_MAX_TOKENS = 3000
# The ledger tag of the judge call: comparison and session ids share their naming
# scheme (the first variant's session often has the very same name), so a bare id
# would count the judge among that session's calls.
LEDGER_PREFIX = "compare:"


def letters(names: list[str]) -> dict[str, str]:
    """Map variant names to the letters the judge sees (A, B, …)."""
    return dict(zip(names, string.ascii_uppercase, strict=False))


def judge_request(
    *,
    compare_id: str,
    topic: str,
    research: list[ResearchNote],
    syntheses: dict[str, str],
    language: Language,
    model: str,
) -> CompletionRequest:
    """Build the judge call over each variant's synthesis (variant name → text)."""
    letter_of = letters(list(syntheses))
    blocks = [
        prompts.COMPARE_JUDGE_SYNTHESIS.format(letter=letter_of[name], text=text.strip())
        for name, text in syntheses.items()
    ]
    task = prompts.COMPARE_JUDGE_TASK.format(
        topic=topic,
        facts=context.facts_text(research) or prompts.COMPARE_JUDGE_NO_FACTS,
        syntheses="\n\n".join(blocks),
        language=LANGUAGE_NAMES[language],
    )
    return CompletionRequest(
        system=prompts.COMPARE_JUDGE_SYSTEM,
        messages=[ChatMessage(role="user", content=task)],
        model=model,
        max_output_tokens=JUDGE_MAX_TOKENS,
        tag=CallTag(session=f"{LEDGER_PREFIX}{compare_id}", purpose=JUDGE_PURPOSE),
    )


async def judge(backend: Backend, request: CompletionRequest) -> str:
    """Run the judge call and return its Markdown (raises BackendError)."""
    result = await backend.complete(request)
    return result.text.strip()
