"""Moderator decisions: strict parsing, one repair retry, round-robin fallback.

Bad model output must never crash a session: `decide` returns None when both
attempts fail, and the engine falls back to `round_robin`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from privy_council.backends.base import (
    BackendError,
    BackendRateLimitError,
    ChatMessage,
    CompletionResult,
)
from privy_council.engine import prompts
from privy_council.models.flaw import Flaw
from privy_council.models.moderator import FLAW_KINDS, ModeratorDecision, NextTurn, ReportedFlaw
from privy_council.models.persona import Persona
from privy_council.models.session import DebateState, Disagreement, PendingTurn
from privy_council.util.language import Language

# Sends a conversation to the moderator in JSON mode; `purpose` tags the call.
ModeratorCall = Callable[[list[ChatMessage], str], Awaitable[CompletionResult]]

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_LEADING_MENTION = re.compile(r"^@([^\W_][\w-]*)")
# Questions the moderator may put to each person per update; more would bury them.
MAX_USER_QUESTIONS = 2
# Share of words two questions to one person have in common for the second to be a repeat.
REPEAT_OVERLAP = 0.8
# Web searches the moderator may request per update (each one is a Claude call).
MAX_RESEARCH_REQUESTS = 2
# New arguments the moderator lists per update; more is a sign it isn't filtering.
MAX_NEW_ARGUMENTS = 5
# User facts kept in the debate state; past this, the state crowds out messages.
MAX_USER_FACTS = 12
# A user fact this long found word for word in the research or the dossier is a copy
# (shorter ones, "40 pity", the user may well have said too).
_MIN_QUOTED_CHARS = 30
# Quote marks the moderator may wrap a flaw's quote in (the floor prompt adds its own).
_QUOTE_MARKS = '"\u201c\u201d\u00ab\u00bb '
# Flaws the moderator reports per update (M11); the engine prompts the first few.
MAX_FLAWS = 4
# Share of a planned sentence's words found in the flaw's quote and note for that
# sentence to restate the flaw (Claude's Chair often did, 2026-10-30).
RESTATED_OVERLAP = 0.6
# Fewer words than this and a sentence ("Check it.") is too short to judge.
_MIN_RESTATED_WORDS = 4
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# Words too common to tell two sentences apart.
_FILLER = frozenset(
    {
        "the",
        "and",
        "that",
        "this",
        "with",
        "your",
        "you",
        "for",
        "not",
        "but",
        "are",
        "was",
        "from",
        "have",
        "has",
        "its",
        "than",
        "then",
        "which",
        "what",
        "about",
        "into",
        "over",
        "also",
        "only",
        "just",
        "say",
        "whether",
        "les",
        "des",
        "une",
        "est",
        "que",
        "qui",
        "pas",
        "pour",
        "dans",
        "ton",
        "tes",
        "avec",
        "sur",
    }
)


def moderator_schema(people: int = 1) -> dict[str, Any]:
    """Return the JSON schema sent to the moderator model, for a chat of `people`.

    Stricter than ModeratorDecision: "turns are required unless stopping" can't be
    expressed in JSON schema, so schema-constrained output (claude --json-schema)
    could legally return no turns. Requiring at least one turn always closes that
    gap; turns are simply ignored when the moderator stops.
    """
    schema = ModeratorDecision.model_json_schema()
    schema["required"] = sorted(
        {
            *schema.get("required", []),
            *(
                "next_turns",
                "stop",
                "consensus",
                "new_arguments",
                "decision_open",
                "position_changes",
                "flaws",
            ),
        }
    )
    # Always a list in constrained output (None only covers replies without the field).
    schema["properties"]["new_arguments"] = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": MAX_NEW_ARGUMENTS,
    }
    schema["properties"]["position_changes"] = {"type": "array", "items": {"type": "string"}}
    schema["properties"]["flaws"]["maxItems"] = MAX_FLAWS
    flaw = schema["$defs"]["ReportedFlaw"]
    flaw["properties"]["kind"] = {"type": "string", "enum": list(FLAW_KINDS)}
    flaw["required"] = ["member", "kind", "quote", "note"]
    schema["properties"]["next_turns"]["minItems"] = 1
    schema["properties"]["questions_for_user"]["maxItems"] = MAX_USER_QUESTIONS * people
    schema["properties"]["research"]["maxItems"] = MAX_RESEARCH_REQUESTS
    state = schema["$defs"]["DebateState"]["properties"]
    state["user_facts"]["maxItems"] = MAX_USER_FACTS
    return schema


class DecisionError(Exception):
    """The moderator's output is unusable; the message goes into the repair prompt."""


def resolve_speaker(name: str, members: Mapping[str, Persona]) -> str | None:
    """Map a slug, nick or @nick to a member slug (case-insensitive), or None."""
    key = name.strip().lstrip("@").lower()
    for slug, persona in members.items():
        if key in (slug, persona.name.lower()):
            return slug
    return None


def json_object(result: CompletionResult) -> dict[str, Any]:
    """Return the reply's JSON object: structured output, else the text parsed as JSON."""
    if result.structured is not None:
        return result.structured
    if result.truncated:
        raise DecisionError("the reply was cut off by the output cap; keep it shorter")
    try:
        value = json.loads(_FENCE.sub("", result.text.strip()))
    except json.JSONDecodeError as exc:
        raise DecisionError(f"not valid JSON ({exc.msg} at char {exc.pos})") from exc
    if not isinstance(value, dict):
        raise DecisionError("expected a JSON object")
    return value


def _normalize_state(state: DebateState, members: Mapping[str, Persona]) -> DebateState:
    """Key positions and sides by member slug, keeping unknown keys as given."""

    def slug(key: str) -> str:
        """Resolve one key, leaving it unchanged if it names no member."""
        return resolve_speaker(key, members) or key

    return DebateState(
        user_facts=_clean(state.user_facts, MAX_USER_FACTS),
        positions={slug(k): v for k, v in state.positions.items()},
        agreements=state.agreements,
        disagreements=[
            Disagreement(topic=d.topic, sides={slug(k): v for k, v in d.sides.items()})
            for d in state.disagreements
        ],
        open_questions=state.open_questions,
    )


def keep_user_facts(previous: DebateState, new: DebateState) -> DebateState:
    """Return the new state, keeping the earlier user facts if it has none at all.

    A changed list is taken as given (the user may have corrected a detail); an
    empty one is the moderator forgetting the field, not the user unsaying everything.
    """
    if new.user_facts or not previous.user_facts:
        return new
    return new.model_copy(update={"user_facts": previous.user_facts})


def drop_quoted_facts(previous: DebateState, new: DebateState, sources: str) -> DebateState:
    """Drop new user facts that copy the facts or the dossier: the web said them, not the user.

    A moderator copying research bullets into user_facts turns them into the user's
    own words (and, with memory on, into shared memory). An earlier fact stays: the
    user may have repeated a figure the research also gives.
    """
    text = _folded(sources)
    old = set(previous.user_facts)
    kept = [
        f
        for f in new.user_facts
        if f in old or len(_folded(f)) < _MIN_QUOTED_CHARS or _folded(f) not in text
    ]
    if len(kept) == len(new.user_facts):
        return new
    return new.model_copy(update={"user_facts": kept})


def _folded(text: str) -> str:
    """Return text lowercased, with punctuation and runs of whitespace folded to a space."""
    return " ".join(re.sub(r"[^\w%€$£]+", " ", text.lower()).split())


def drop_answered(previous: DebateState, new: DebateState, spoke: Collection[str]) -> DebateState:
    """Drop open questions to a person who spoke since, when they come back unchanged.

    A local moderator copies the earlier open_questions word for word, answered or
    not (M16 full check); one it rewrote is taken as still open.
    """
    names = {name.lower() for name in spoke}
    answered = {q for q in previous.open_questions if _addressee(q) in names}
    kept = [q for q in new.open_questions if q not in answered]
    return new.model_copy(update={"open_questions": kept})


def cap_questions(questions: Sequence[str], owner: str, guests: Sequence[str]) -> list[str]:
    """Keep the first MAX_USER_QUESTIONS questions of each person, in order.

    A question goes to the person whose @name starts it, else to the owner.
    """
    guest_names = {g.lower() for g in guests}
    counts: dict[str, int] = {}
    kept: list[str] = []
    for question in questions:
        name = _addressee(question)
        person = name if name in guest_names else owner.lower()
        if counts.get(person, 0) < MAX_USER_QUESTIONS:
            counts[person] = counts.get(person, 0) + 1
            kept.append(question)
    return kept


def drop_asked(questions: Sequence[str], asked: Sequence[str], owner: str) -> list[str]:
    """Drop the questions already asked to the same person, word for word or nearly.

    The prompt lists what was asked, yet a local Chair re-asked a question answered
    an hour before, only with a colon added (2026-10-27 Discord debate).
    """
    return [q for q in questions if not asked_before(q, asked, owner)]


def asked_before(question: str, asked: Sequence[str], owner: str) -> bool:
    """Tell whether `question` repeats one of `asked` to the same person.

    A repeat shares at least REPEAT_OVERLAP of its words (the @name aside).
    """
    person, words = _question_words(question, owner)
    for earlier in asked:
        earlier_person, earlier_words = _question_words(earlier, owner)
        if person != earlier_person or not (words and earlier_words):
            continue
        if len(words & earlier_words) / len(words | earlier_words) >= REPEAT_OVERLAP:
            return True
    return False


def _question_words(question: str, owner: str) -> tuple[str, set[str]]:
    """Return a question's addressee (the owner when it names nobody) and its words."""
    person = _addressee(question) or owner.lower()
    return person, set(_folded(_LEADING_MENTION.sub("", question.strip())).split())


def guest_question(
    questions: Sequence[str],
    open_questions: Sequence[str],
    guests: Sequence[str],
    asked: Sequence[str],
) -> list[str]:
    """Return the first open question to a guest, when `questions` ask no guest.

    Nothing if a guest is already asked, or if every guest question was asked before.
    """
    names = {g.lower() for g in guests}
    if any(_addressee(q) in names for q in questions):
        return []
    for question in open_questions:
        person = _addressee(question)
        if person in names and not asked_before(question, asked, person or ""):
            return [question]
    return []


def _addressee(question: str) -> str | None:
    """Return the lower-cased @name a question starts with, if any."""
    match = _LEADING_MENTION.match(question)
    return match.group(1).lower() if match else None


def _normalize_turns(turns: Sequence[NextTurn], members: Mapping[str, Persona]) -> list[NextTurn]:
    """Resolve speakers to slugs; drop unknown names and repeat speakers.

    One turn per member per round keeps a runaway plan from burning calls.
    """
    out: list[NextTurn] = []
    seen: set[str] = set()
    for turn in turns:
        slug = resolve_speaker(turn.speaker, members)
        if slug is not None and slug not in seen:
            seen.add(slug)
            out.append(NextTurn(speaker=slug, prompt=turn.prompt))
    return out


def _member_slugs(names: Sequence[str] | None, members: Mapping[str, Persona]) -> list[str] | None:
    """Resolve names to member slugs, dropping unknown ones and repeats (None stays None)."""
    if names is None:
        return None
    slugs = [resolve_speaker(name, members) for name in names]
    return list(dict.fromkeys(s for s in slugs if s is not None))


def _flaws(flaws: Sequence[ReportedFlaw], members: Mapping[str, Persona]) -> list[ReportedFlaw]:
    """Resolve flaw members to slugs, dropping unknown names and repeats; keep MAX_FLAWS."""
    out: list[ReportedFlaw] = []
    for flaw in flaws:
        slug = resolve_speaker(flaw.member, members)
        quote = " ".join(flaw.quote.split()).strip(_QUOTE_MARKS)
        if slug is None or not quote:
            continue
        kind = flaw.kind.strip().lower()
        item = ReportedFlaw(
            member=slug,
            kind=kind if kind in FLAW_KINDS else "other",
            quote=quote,
            note=" ".join(flaw.note.split()),
        )
        if all((f.member, f.quote) != (item.member, item.quote) for f in out):
            out.append(item)
    return out[:MAX_FLAWS]


def _clean(items: Sequence[str], limit: int) -> list[str]:
    """Drop blank and repeated items (whitespace folded), and any past `limit`."""
    out: list[str] = []
    for item in items:
        text = " ".join(item.split())
        if text and text not in out:
            out.append(text)
    return out[:limit]


def parse_decision(result: CompletionResult, members: Mapping[str, Persona]) -> ModeratorDecision:
    """Parse and normalize a moderator reply, raising DecisionError if unusable."""
    try:
        decision = ModeratorDecision.model_validate(json_object(result))
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()
        )
        raise DecisionError(f"schema mismatch ({details})") from exc
    turns = _normalize_turns(decision.next_turns, members)
    if not decision.stop and not turns:
        names = ", ".join(t.speaker for t in decision.next_turns)
        raise DecisionError(
            f"next_turns names no council member ({names}); use one of: {', '.join(members)}"
        )
    return decision.model_copy(
        update={
            "debate_state": _normalize_state(decision.debate_state, members),
            "next_turns": turns,
            "questions_for_user": _clean(
                decision.questions_for_user, len(decision.questions_for_user)
            ),
            "research": _clean(decision.research, MAX_RESEARCH_REQUESTS),
            "new_arguments": (
                None
                if decision.new_arguments is None
                else _clean(decision.new_arguments, MAX_NEW_ARGUMENTS)
            ),
            "position_changes": _member_slugs(decision.position_changes, members),
            "flaws": _flaws(decision.flaws, members),
        }
    )


async def decide(
    call: ModeratorCall, request: str, members: Mapping[str, Persona]
) -> tuple[ModeratorDecision | None, list[str]]:
    """Ask for a decision, repairing once; return it (or None) and the problems seen.

    Backend errors other than rate limits count as a failed attempt; rate limits
    propagate so the engine can pause.
    """
    messages = [ChatMessage(role="user", content=request)]
    problems: list[str] = []
    for purpose in ("moderate", "repair"):
        try:
            result = await call(messages, purpose)
        except BackendRateLimitError:
            raise
        except BackendError as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
            continue
        try:
            return parse_decision(result, members), problems
        except DecisionError as exc:
            problems.append(str(exc))
            messages = [
                *messages,
                ChatMessage(role="assistant", content=result.text or "(empty)"),
                ChatMessage(role="user", content=prompts.REPAIR_TASK.format(error=exc)),
            ]
    return None, problems


def round_robin(members: Sequence[str], lang: Language = "en") -> list[PendingTurn]:
    """Fallback plan: every member speaks once, in roster order."""
    prompt = prompts.FALLBACK_TURN_PROMPT[lang]
    return [PendingTurn(speaker=s, prompt=prompt) for s in members]


def with_flaw_turns(
    turns: Sequence[PendingTurn], flaws: Sequence[Flaw], cap: int, lang: Language = "en"
) -> list[PendingTurn]:
    """Put floor turns asking members to check their flaws ahead of the planned ones.

    At most `cap` flaws (2 by default), one per member, in the
    moderator's order. A member who already had a turn gets one turn that does both.
    """
    picked: list[Flaw] = []
    for flaw in flaws:
        if len(picked) >= cap:
            break
        if all(p.member != flaw.member for p in picked):
            picked.append(flaw)
    rest = list(turns)
    front: list[PendingTurn] = []
    for flaw in picked:
        planned = next((t for t in rest if t.speaker == flaw.member), None)
        if planned is not None:
            rest.remove(planned)
        then = _without_flaw(planned.prompt, flaw) if planned is not None else None
        prompt = prompts.flaw_turn_prompt(flaw.excerpt, flaw.note, lang, then)
        front.append(PendingTurn(speaker=flaw.member, prompt=prompt))
    return front + rest


def _without_flaw(planned: str, flaw: Flaw) -> str | None:
    """Return a planned prompt minus its @address and the sentences restating the flaw.

    The flaw prompt already names the member and quotes the error, so "@Rigor, your
    30% gives 11%, not 13%." after it only repeats; None when nothing else is left.
    """
    text = planned.strip()
    address = _LEADING_MENTION.match(text)
    if address and address.group(1).lower() == flaw.member:
        text = text[address.end() :].lstrip(" ,:")
    flaw_words = _content_words(f"{flaw.excerpt} {flaw.note}")
    pieces = _SENTENCE_END.split(text)
    keep = [not _restates(piece, flaw_words) for piece in pieces]
    kept = []
    for i, piece in enumerate(pieces):
        if not keep[i]:
            continue
        if i > 0 and not keep[i - 1]:  # now starts a sentence: "Then say …" → "Say …"
            piece = re.sub(r"^then\b[ ,:]*", "", piece, flags=re.IGNORECASE)
            piece = piece[:1].upper() + piece[1:]
        if piece.endswith(";") and i + 1 < len(pieces) and not keep[i + 1]:
            piece = piece[:-1] + "."
        kept.append(piece)
    rest = " ".join(kept).strip()
    if rest == planned.strip():
        return rest
    return rest[:1].upper() + rest[1:] if rest else None


def _restates(sentence: str, flaw_words: set[str]) -> bool:
    """Whether a sentence mostly repeats the flaw's quote and note."""
    words = _content_words(sentence)
    if len(words) < _MIN_RESTATED_WORDS:
        return False
    return len(words & flaw_words) / len(words) >= RESTATED_OVERLAP


def _content_words(text: str) -> set[str]:
    """Return the words of a text that carry meaning: numbers, and words of 3+ letters."""
    return {w for w in _folded(text).split() if (len(w) > 2 or w[0].isdigit()) and w not in _FILLER}


def to_pending(turns: Sequence[NextTurn]) -> list[PendingTurn]:
    """Convert the moderator's turn plan into pending turns."""
    return [PendingTurn(speaker=t.speaker, prompt=t.prompt) for t in turns]
