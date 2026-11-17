"""End-of-session memory: private notes, shared-fact proposals and consolidation.

`MemoryWriter` builds the prompts and checks the replies; the engine supplies
the model call (routing, fallback, ledger) and decides what to do with a
failure. Bad model output never corrupts a memory file: unusable notes are
skipped, and a consolidation that comes back empty or still over the cap is
discarded, leaving the file as it was.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from privy_council.backends.base import CompletionResult
from privy_council.config import MemoryConfig
from privy_council.engine import context, prompts
from privy_council.memory.store import (
    TAGGED_BULLET,
    MemoryStore,
    bullet,
    bullet_lines,
    size_bytes,
    split_council,
)
from privy_council.models.persona import Persona
from privy_council.models.session import Message
from privy_council.util.language import Language


class MergeOutcome(StrEnum):
    """How a persona's over-cap notes were brought back under the cap."""

    MERGED = "merged"
    TRIMMED = "merged, then kept to the newest notes"
    CUT = "kept to the newest notes, unmerged"
    FAILED = "kept as is"


MAX_NOTES = 3
NOTE_MAX_CHARS = 300
NOTES_MAX_OUTPUT = 400
SYNTHESIS_MAX_CHARS = 8000
MAX_SHARED_FACTS = 5
SHARED_MAX_OUTPUT = 800
# Consolidation aims well under the cap, so a few sessions fit before the next pass.
CONSOLIDATE_TARGET = 0.7
# The one retry after an unusable merge aims lower: the first reply ran long.
CONSOLIDATE_RETRY_TARGET = 0.5
# Merges ask for a number of notes, not characters: local models don't count characters
# (Bolt answered "under 2,048 characters" with 4,319 bytes, 2026-11-16) but keep a count.
MERGE_NOTE_WORDS = 30
MERGE_NOTE_BYTES = 225  # a 30-word note with its "- YYYY-MM-DD (council): " prefix
# A quote shorter than this matches too easily to prove the user said anything.
MIN_QUOTE_CHARS = 4
NOTES_CHECK_MAX_OUTPUT = 1500
# A new note whose words are mostly in an existing one only repeats it.
REPEAT_SHARE = 0.8

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
# Typographic quotes (U+2018/2019, U+201C/201D, guillemets) match their ASCII forms.
_QUOTES = str.maketrans(
    {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"'}
)
_EDGE_PUNCT = " \t\n\"'.,;:!?…()[]"
# Words that carry meaning: 4+ letters, or anything with a digit (a changed figure counts).
_WORD = re.compile(r"\w*\d\w*|\w{4,}")

logger = logging.getLogger(__name__)


class MemoryOutputError(Exception):
    """A memory reply was unusable (not JSON, wrong shape)."""


class ProposedFact(BaseModel):
    """One shared fact the moderator proposes, with the user's words that state it."""

    fact: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class SharedProposal(BaseModel):
    """The moderator's structured reply for shared memory."""

    facts: list[ProposedFact] = Field(default_factory=list)


class NoteChange(BaseModel):
    """One member note the moderator corrects; an empty note drops it."""

    n: int = Field(ge=1)
    note: str


class NotesCheck(BaseModel):
    """The moderator's structured reply when checking the members' notes."""

    changes: list[NoteChange] = Field(default_factory=list)


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


def notes_check_schema() -> dict[str, Any]:
    """Return the JSON schema of the notes check reply."""
    return {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}, "note": {"type": "string"}},
                    "required": ["n", "note"],
                },
            }
        },
        "required": ["changes"],
    }


def _structured(result: CompletionResult) -> Any:
    """Return a reply's JSON: the structured output, else the text parsed."""
    if result.structured is not None:
        return result.structured
    try:
        return json.loads(_FENCE.sub("", result.text.strip()))
    except json.JSONDecodeError as exc:
        raise MemoryOutputError(f"not valid JSON ({exc.msg})") from exc


def parse_check(result: CompletionResult) -> list[NoteChange]:
    """Return the note changes from a check reply, or raise MemoryOutputError."""
    try:
        return NotesCheck.model_validate(_structured(result)).changes
    except ValidationError as exc:
        raise MemoryOutputError(f"wrong shape ({exc.error_count()} errors)") from exc


def _words(text: str) -> set[str]:
    """Return the meaningful words of a note, normalized."""
    return set(_WORD.findall(normalize(text)))


def repeats(note: str, other: str) -> bool:
    """Return whether two notes say the same thing.

    Most of the shorter note's words are in the longer one, figures included: a
    note that only updates a number is new, one that adds a tail to an old note
    is not.
    """
    mine, theirs = _words(note), _words(other)
    if not mine or not theirs:
        return False
    shorter, longer = sorted((mine, theirs), key=len)
    if any(w not in longer for w in shorter if any(c.isdigit() for c in w)):
        return False
    return len(shorter & longer) >= REPEAT_SHARE * len(shorter)


def fresh_notes(notes: Sequence[str], existing: str) -> list[str]:
    """Drop notes that repeat an existing note or an earlier one in `notes`.

    Local models copied a lesson from the last session almost word for word,
    despite "don't repeat your existing notes" (career session, 2026-11-07).
    """
    seen = [line.partition("): ")[2] or line for line in bullet_lines(existing)]
    kept: list[str] = []
    for note in notes:
        if not any(repeats(note, old) for old in [*seen, *kept]):
            kept.append(note)
    return kept


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
    try:
        return SharedProposal.model_validate(_structured(result)).facts
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


def consolidation_target(cap_bytes: int, share: float = CONSOLIDATE_TARGET) -> int:
    """Return the size a consolidation should aim for."""
    return int(cap_bytes * share)


def merge_counts(target: int) -> dict[str, int]:
    """Return the prompt's `count` (notes) and `words` (per note) for a size target."""
    return {"count": max(target // MERGE_NOTE_BYTES, 3), "words": MERGE_NOTE_WORDS}


def parse_consolidated(
    text: str, cap_bytes: int, council: str | None = None
) -> tuple[str | None, str]:
    """Return (the consolidated bullets, "") or (None, why the reply is unusable).

    With `council`, every line must keep that council's tag, so a merge can't
    move a note into another council's scope.
    """
    merged = _bullets(text)
    if not merged:
        return None, prompts.MERGE_EMPTY
    if council is not None and any(_council_of(line) != council for line in merged.splitlines()):
        return None, prompts.MERGE_LOST_TAG.format(council=council)
    if size_bytes(merged) > cap_bytes:
        return None, prompts.MERGE_OVER_CAP.format(size=size_bytes(merged), cap=cap_bytes)
    return merged, ""


def _bullets(text: str) -> str:
    """Return a reply's bullet lines, one "- " line each."""
    return "\n".join(f"- {line}" for line in bullet_lines(text))


def keep_newest(notes: str, limit: int) -> str:
    """Return the newest notes that fit in `limit` bytes, in their original order.

    The last resort when no merge fits the cap: the oldest points go first, as a
    merge is told to drop them first anyway. Undated lines count as the oldest.
    """
    lines = [line for line in notes.splitlines() if line.strip()]
    newest_first = sorted(range(len(lines)), key=lambda i: _date_of(lines[i]), reverse=True)
    kept: set[int] = set()
    size = 0
    for i in newest_first:
        line_size = size_bytes(lines[i]) + 1  # its line break
        if size + line_size > limit:
            break
        kept.add(i)
        size += line_size
    return "\n".join(line for i, line in enumerate(lines) if i in kept)


def _date_of(line: str) -> str:
    """Return a bullet's YYYY-MM-DD date, or "" if it has none (sorts as the oldest)."""
    found = re.match(r"- (\d{4}-\d{2}-\d{2})", line)
    return found.group(1) if found else ""


def _council_of(line: str) -> str | None:
    """Return the council tag of a bullet line, or None if it has none."""
    tag = TAGGED_BULLET.match(line)
    return tag.group(1) if tag is not None else None


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


def _apply_changes(
    drafts: dict[str, list[str]], numbered: list[tuple[str, int]], changes: list[NoteChange]
) -> tuple[dict[str, list[str]], int]:
    """Apply the moderator's rewrites to the drafts; return them and how many changed.

    An empty note is ignored, never a drop: a local Chair (Super) emptied most
    notes about members and the debate, which the check must leave alone
    (2026-11-08 bench). A rewrite is the only change applied.
    """
    revised = {slug: list(notes) for slug, notes in drafts.items()}
    fixed = 0
    for change in {c.n: c for c in changes if c.n <= len(numbered)}.values():
        slug, i = numbered[change.n - 1]
        note = _shorten(change.note.strip().removeprefix("- ").strip())
        if not note or note == revised[slug][i]:
            continue
        revised[slug][i] = note
        fixed += 1
    return revised, fixed


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
        return self.save_notes(persona, digest, await self.draft_notes(persona, digest))

    def _existing(self, persona: Persona, council: str) -> str:
        """Return the notes a persona sees when writing new ones."""
        return self.store.read_private(persona.slug, self.cfg.notes_council(council))

    async def draft_notes(self, persona: Persona, digest: SessionDigest) -> list[str]:
        """Have a persona distill the session into notes, without saving them."""
        own = [m for m in digest.messages if m.speaker in (persona.slug, "user")]
        existing = self._existing(persona, digest.council)
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
        return parse_notes(result.text)

    def save_notes(self, persona: Persona, digest: SessionDigest, notes: Sequence[str]) -> int:
        """Append the notes that don't repeat existing ones; return how many were saved."""
        fresh = fresh_notes(notes, self._existing(persona, digest.council))
        self.store.append_private(
            persona.slug, [bullet(digest.day, digest.council, n) for n in fresh]
        )
        return len(fresh)

    async def check_notes(
        self, moderator: Persona, digest: SessionDigest, drafts: dict[str, list[str]]
    ) -> tuple[dict[str, list[str]], int]:
        """Have the moderator check what the drafts say about the user.

        Returns the corrected drafts and how many notes were rewritten.
        Local models credited a member's argument to the user, or dropped the
        condition of an answer (career session, 2026-11-07).
        """
        numbered = [(slug, i) for slug, notes in drafts.items() for i in range(len(notes))]
        if not numbered:
            return drafts, 0
        listing = "\n".join(f"{n}. {drafts[s][i]}" for n, (s, i) in enumerate(numbered, 1))
        asked = [m for m in digest.messages if _by_owner(m) or m.to_user]
        prompt = context.build_prompt(
            system=prompts.NOTES_CHECK_SYSTEM,
            topic=digest.topic,
            task=prompts.task(prompts.NOTES_CHECK_TASK, digest.lang, notes=listing),
            budget_tokens=self.budget(moderator),
            history=context.render_history(asked, digest.nick_of),
            history_title=prompts.TITLE_USER_ANSWERS,
        )
        result = await self.complete(
            moderator,
            prompt,
            "notes_check",
            json_schema=notes_check_schema(),
            max_output=NOTES_CHECK_MAX_OUTPUT,
        )
        return _apply_changes(drafts, numbered, parse_check(result))

    def council_notes(self, persona: Persona, council: str) -> str:
        """Return the persona's notes tagged with `council` (the part the cap applies to)."""
        own, _ = split_council(self.store.read_private(persona.slug), council)
        return "\n".join(own)

    def private_over_cap(self, persona: Persona, council: str) -> bool:
        """Return whether a persona's notes from `council` exceed the private cap.

        The cap is per council: a persona that sits in several councils keeps
        `private_cap_bytes` of notes for each.
        """
        return size_bytes(self.council_notes(persona, council)) > self.cfg.private_cap_bytes

    def shared_over_cap(self) -> bool:
        """Return whether the shared facts exceed the shared cap."""
        return size_bytes(self.store.read_shared()) > self.cfg.shared_cap_bytes

    async def consolidate(self, persona: Persona, council: str) -> MergeOutcome:
        """Merge a persona's notes from `council` on its own backend.

        When neither reply fits, the newest notes that do are kept: of the over-cap
        reply if it was well formed, else of the notes themselves. Notes over the cap
        would otherwise fail every merge again and grow each session (2026-11-16).
        """
        notes = self.council_notes(persona, council)

        def task(target: int) -> str:
            """Return the merge task aiming at about `target` characters."""
            counts = merge_counts(target)
            return prompts.CONSOLIDATE_TASK.format(council=council, notes=notes, **counts)

        cap = self.cfg.private_cap_bytes
        merged, over_cap = await self._merge(persona, persona.body, task, cap, council)
        outcome = MergeOutcome.MERGED
        if merged is None:
            merged = keep_newest(over_cap or notes, consolidation_target(cap))
            outcome = MergeOutcome.TRIMMED if over_cap else MergeOutcome.CUT
            logger.warning("memory of %s kept to its newest notes (%s)", persona.slug, outcome)
        if not merged:
            return MergeOutcome.FAILED
        self.store.rewrite_private_council(persona.slug, council, merged)
        return outcome

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
        notes = self.store.read_shared()

        def task(target: int) -> str:
            """Return the merge task aiming at about `target` characters."""
            return prompts.CONSOLIDATE_SHARED_TASK.format(notes=notes, **merge_counts(target))

        # Never trimmed: the user approves a shared rewrite, and facts about them stay whole.
        merged, _ = await self._merge(
            moderator, prompts.SHARED_SYSTEM, task, self.cfg.shared_cap_bytes
        )
        return merged

    async def _merge(
        self,
        persona: Persona,
        system: str,
        task: Callable[[int], str],
        cap: int,
        council: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Run a consolidation call, retrying once lower if the reply is unusable.

        Returns (the merge, None), or (None, the smaller well-formed reply that was only
        over the cap, if any).
        """
        first = consolidation_target(cap)
        merged, why, over = await self._merge_once(persona, system, task(first), cap, council)
        if merged is not None:
            return merged, None
        logger.warning("memory merge for %s unusable, retrying: %s", persona.slug, why)
        retry = consolidation_target(cap, CONSOLIDATE_RETRY_TARGET)
        counts = merge_counts(retry)
        again = task(retry) + prompts.MERGE_RETRY.format(reason=why, **counts)
        merged, why, over_again = await self._merge_once(persona, system, again, cap, council)
        if merged is not None:
            return merged, None
        logger.warning("memory merge for %s unusable twice: %s", persona.slug, why)
        candidates = [text for text in (over, over_again) if text]
        return None, min(candidates, key=size_bytes) if candidates else None

    async def _merge_once(
        self, persona: Persona, system: str, task: str, cap: int, council: str | None
    ) -> tuple[str | None, str, str | None]:
        """Run one consolidation call.

        Returns (the bullets, "", None) or (None, why they are unusable, the bullets
        if they were well formed and only over the cap).
        """
        prompt = context.Prompt(system=system, user=task)
        # A usable reply is under `cap` bytes (≈ cap / 4 tokens), but Bolt once spent 2,048
        # tokens on a 2,681-byte retry (2026-11-17): room for that; the byte cap still rules.
        result = await self.complete(persona, prompt, "consolidate", max_output=max(cap, 500))
        if result.truncated:
            return None, prompts.MERGE_TRUNCATED, None
        merged, why = parse_consolidated(result.text, cap, council)
        if merged is not None:
            return merged, "", None
        well_formed, _ = parse_consolidated(result.text, sys.maxsize, council)
        return None, why, well_formed
