"""End-of-session memory: private notes, shared-fact proposals and consolidation.

`MemoryWriter` builds the prompts and checks the replies; the engine supplies
the model call (routing, fallback, ledger) and decides what to do with a
failure. Bad model output never corrupts a memory file: unusable notes are
skipped, and a consolidation that comes back empty or still over the cap is
discarded, leaving the file as it was.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from privy_council.backends.base import CompletionResult
from privy_council.config import MemoryConfig
from privy_council.engine import context, prompts
from privy_council.memory.store import MemoryStore, bullet, bullet_lines, size_bytes
from privy_council.models.persona import Persona
from privy_council.models.session import Message
from privy_council.util.language import Language

MAX_NOTES = 3
NOTE_MAX_CHARS = 300
NOTES_MAX_OUTPUT = 400
SYNTHESIS_MAX_CHARS = 8000
MAX_SHARED_FACTS = 5
SHARED_MAX_OUTPUT = 800
# Consolidation aims well under the cap, so a few sessions fit before the next pass.
CONSOLIDATE_TARGET = 0.7
# A quote shorter than this matches too easily to prove the user said anything.
MIN_QUOTE_CHARS = 4

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
# Typographic quotes (U+2018/2019, U+201C/201D, guillemets) match their ASCII forms.
_QUOTES = str.maketrans(
    {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"'}
)
_EDGE_PUNCT = " \t\n\"'.,;:!?…()[]"


class MemoryOutputError(Exception):
    """A memory reply was unusable (not JSON, wrong shape)."""


class ProposedFact(BaseModel):
    """One shared fact the moderator proposes, with the user's words that state it."""

    fact: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class SharedProposal(BaseModel):
    """The moderator's structured reply for shared memory."""

    facts: list[ProposedFact] = Field(default_factory=list)


def shared_schema() -> dict[str, Any]:
    """Return the JSON schema for shared-fact proposals."""
    fact = {
        "type": "object",
        "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["fact", "quote"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": fact, "maxItems": MAX_SHARED_FACTS}},
        "required": ["facts"],
        "additionalProperties": False,
    }


def parse_notes(text: str) -> list[str]:
    """Extract up to MAX_NOTES notes; a reply without bullets counts as one note."""
    stripped = text.strip()
    if not stripped or stripped.upper().rstrip(".") == prompts.NOTES_NONE:
        return []
    notes = bullet_lines(stripped)
    if not notes:
        notes = [next(line for line in stripped.splitlines() if line.strip())]
    return [_shorten(n) for n in notes if n.strip()][:MAX_NOTES]


def _shorten(note: str) -> str:
    """Cut an overlong note at a word boundary, marking the cut."""
    note = note.strip()
    if len(note) <= NOTE_MAX_CHARS:
        return note
    return note[: NOTE_MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def parse_shared(result: CompletionResult) -> list[ProposedFact]:
    """Return the proposed facts from a structured reply, or raise MemoryOutputError."""
    raw: Any = result.structured
    if raw is None:
        try:
            raw = json.loads(_FENCE.sub("", result.text.strip()))
        except json.JSONDecodeError as exc:
            raise MemoryOutputError(f"not valid JSON ({exc.msg})") from exc
    try:
        return SharedProposal.model_validate(raw).facts
    except ValidationError as exc:
        raise MemoryOutputError(f"wrong shape ({exc.error_count()} errors)") from exc


def normalize(text: str) -> str:
    """Normalize text for quote matching: case, quote marks, spacing, edge punctuation."""
    return " ".join(text.translate(_QUOTES).casefold().split()).strip(_EDGE_PUNCT)


def quoted_facts(
    facts: Sequence[ProposedFact], sources: Sequence[str], known: str
) -> tuple[list[str], int]:
    """Keep facts whose quote is in what the user wrote and that aren't on file yet.

    Returns the kept facts and how many were dropped for a quote the user never
    wrote. This is the deterministic guard behind "only what the user stated".
    """
    said = [normalize(s) for s in sources]
    on_file = {normalize(line.partition("): ")[2] or line) for line in bullet_lines(known)}
    kept: list[str] = []
    unquoted = 0
    for item in facts:
        quote = normalize(item.quote)
        if len(quote) < MIN_QUOTE_CHARS or not any(quote in s for s in said):
            unquoted += 1
            continue
        fact = " ".join(item.fact.split())
        if normalize(fact) not in on_file and fact not in kept:
            kept.append(fact)
    return kept[:MAX_SHARED_FACTS], unquoted


def consolidation_target(cap_bytes: int) -> int:
    """Return the size a consolidation should aim for."""
    return int(cap_bytes * CONSOLIDATE_TARGET)


def parse_consolidated(text: str, cap_bytes: int) -> str | None:
    """Return the consolidated bullets, or None if empty or still over the cap."""
    lines = [f"- {line}" for line in bullet_lines(text)]
    merged = "\n".join(lines)
    if not lines or size_bytes(merged) > cap_bytes:
        return None
    return merged


def parse_selection(arg: str, count: int) -> list[int] | None:
    """Parse a `/keep` argument into 0-based indices: `1 3`, `1,3`, `all` or `none`.

    Returns None when the argument is invalid (unknown word, out-of-range number).
    """
    words = arg.replace(",", " ").lower().split()
    if words == ["all"]:
        return list(range(count))
    if words == ["none"]:
        return []
    if not words or not all(w.isdigit() and 1 <= int(w) <= count for w in words):
        return None
    return sorted({int(w) - 1 for w in words})


class Complete(Protocol):
    """The engine's model call for a persona (routing, fallback and ledger included)."""

    async def __call__(
        self,
        persona: Persona,
        prompt: context.Prompt,
        purpose: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_output: int | None = None,
    ) -> CompletionResult:
        """Run one call and return its result (raises BackendError subclasses)."""
        ...


def _by_owner(message: Message) -> bool:
    """Return whether the session owner wrote this message (not a guest in a shared chat)."""
    return message.speaker == "user" and message.author is None


@dataclass(frozen=True)
class SessionDigest:
    """What the close step distills memory from."""

    council: str
    topic: str
    day: date
    synthesis: str
    messages: Sequence[Message]
    state_text: str | None
    nick_of: Callable[[str], str]
    lang: Language = "en"

    def user_lines(self) -> list[str]:
        """Return everything the owner wrote: the question and their messages."""
        return [self.topic, *(m.text for m in self.messages if _by_owner(m))]


class MemoryWriter:
    """Writes private notes and proposes shared facts and merged files."""

    def __init__(
        self,
        store: MemoryStore,
        cfg: MemoryConfig,
        complete: Complete,
        budget: Callable[[Persona], int],
    ) -> None:
        """Wire the memory files, the caps and the engine's call and budget functions."""
        self.store = store
        self.cfg = cfg
        self.complete = complete
        self.budget = budget

    async def write_notes(self, persona: Persona, digest: SessionDigest) -> int:
        """Have a persona distill the session into its private notes; return the count."""
        own = [m for m in digest.messages if m.speaker in (persona.slug, "user")]
        existing = self.store.read_private(persona.slug, self.cfg.notes_council(digest.council))
        prompt = context.build_prompt(
            system=f"{persona.body}\n\n{prompts.NOTES_SYSTEM.format(name=persona.name)}",
            topic=digest.topic,
            task=prompts.task(
                prompts.NOTES_TASK, digest.lang, synthesis=digest.synthesis[:SYNTHESIS_MAX_CHARS]
            ),
            budget_tokens=self.budget(persona),
            memory=(existing, ""),
            memory_share=self.cfg.context_share,
            state_text=digest.state_text,
            history=context.render_history(own, digest.nick_of),
            history_title=prompts.TITLE_OWN_MESSAGES,
        )
        result = await self.complete(persona, prompt, "memory", max_output=NOTES_MAX_OUTPUT)
        notes = parse_notes(result.text)
        self.store.append_private(
            persona.slug, [bullet(digest.day, digest.council, n) for n in notes]
        )
        return len(notes)

    def private_over_cap(self, persona: Persona) -> bool:
        """Return whether a persona's notes exceed the private cap."""
        return size_bytes(self.store.read_private(persona.slug)) > self.cfg.private_cap_bytes

    def shared_over_cap(self) -> bool:
        """Return whether the shared facts exceed the shared cap."""
        return size_bytes(self.store.read_shared()) > self.cfg.shared_cap_bytes

    async def consolidate(self, persona: Persona) -> bool:
        """Merge a persona's notes on its own backend; return False if the result is bad."""
        cap = self.cfg.private_cap_bytes
        task = prompts.CONSOLIDATE_TASK.format(
            target=consolidation_target(cap), notes=self.store.read_private(persona.slug)
        )
        merged = await self._merge(persona, persona.body, task, "consolidate", cap)
        if merged is None:
            return False
        self.store.rewrite_private(persona.slug, merged)
        return True

    async def propose_facts(
        self, moderator: Persona, digest: SessionDigest
    ) -> tuple[list[str], int]:
        """Ask the moderator for new shared facts; return the quoted ones and drop count."""
        known = self.store.read_shared()
        user = [m for m in digest.messages if _by_owner(m)]
        task = prompts.task(
            prompts.SHARED_TASK,
            digest.lang,
            known=known.strip() or prompts.EMPTY,
            limit=MAX_SHARED_FACTS,
        )
        prompt = context.build_prompt(
            system=prompts.SHARED_SYSTEM,
            topic=digest.topic,
            task=task,
            budget_tokens=self.budget(moderator),
            history=context.render_history(user, digest.nick_of),
            history_title=prompts.TITLE_USER_MESSAGES,
        )
        result = await self.complete(
            moderator,
            prompt,
            "shared_memory",
            json_schema=shared_schema(),
            max_output=SHARED_MAX_OUTPUT,
        )
        kept, unquoted = quoted_facts(parse_shared(result), digest.user_lines(), known)
        return [bullet(digest.day, digest.council, f) for f in kept], unquoted

    async def propose_rewrite(self, moderator: Persona) -> str | None:
        """Ask the moderator for a merged shared file (to be approved), or None if bad."""
        cap = self.cfg.shared_cap_bytes
        task = prompts.CONSOLIDATE_SHARED_TASK.format(
            target=consolidation_target(cap), notes=self.store.read_shared()
        )
        return await self._merge(moderator, prompts.SHARED_SYSTEM, task, "consolidate", cap)

    async def _merge(
        self, persona: Persona, system: str, task: str, purpose: str, cap: int
    ) -> str | None:
        """Run a consolidation call and check its output against the cap."""
        prompt = context.Prompt(system=system, user=task)
        # The reply is at most ~cap bytes; tokens ≈ bytes / 4, so half the cap is generous.
        result = await self.complete(persona, prompt, purpose, max_output=max(cap // 2, 500))
        if result.truncated:
            return None
        return parse_consolidated(result.text, cap)
