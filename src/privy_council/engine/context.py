"""Per-turn context building with token budgeting. Pure functions, no I/O.

Small models never see the full transcript: each turn gets the persona prompt
and engine rules (system), then the question, the dossier (files the user
attached), the moderator's web research (if any), the persona's memory, the
debate state, the last K messages and the task (user). The dossier, research and
memory are each capped at a share of the budget (memory
keeps its newest bullets, private notes first); when the budget is tight, the
oldest messages go first, then the debate state is clipped; the question and
task are never trimmed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from privy_council.config import Settings
from privy_council.engine import prompts
from privy_council.engine.local_research import without_missing
from privy_council.models.persona import BackendKind
from privy_council.models.session import DebateState, DossierFile, Message, ResearchNote
from privy_council.util.tokens import estimate_tokens, tokens_to_chars

# The claude CLI handles long contexts; this only guards against absurd prompts.
CLAUDE_CONTEXT_TOKENS = 100_000
# Never squeeze the input below this, even if num_ctx is misconfigured.
MIN_INPUT_TOKENS = 512

NickOf = Callable[[str], str]


@dataclass(frozen=True)
class Prompt:
    """A built prompt: system text plus a single user message, and what got trimmed."""

    system: str
    user: str
    dropped_messages: int = 0
    state_clipped: bool = False


def input_budget(settings: Settings, kind: BackendKind, model: str, max_output: int) -> int:
    """Return the token budget for the prompt: context window minus the output cap."""
    if kind is BackendKind.OLLAMA:
        window = settings.ollama.models.get(model)
        num_ctx = window.num_ctx if window else 8192
    else:
        num_ctx = CLAUDE_CONTEXT_TOKENS
    return max(num_ctx - max_output, MIN_INPUT_TOKENS)


def sources_text(system: str, user: str) -> str:
    """Return what a member may quote figures from: its prompt minus the task section.

    The task's own numbers ("your top 3", "0-100") would excuse invented figures; a
    floor prompt's numbers are in the recent messages anyway.
    """
    cut = user.rfind(prompts.section(prompts.TITLE_TASK, ""))
    return f"{system}\n{user[:cut] if cut >= 0 else user}"


def given_text(user: str) -> str:
    """Return the fresh facts and dossier sections of a prompt: the figures given as facts."""
    parts = []
    for title in (prompts.TITLE_FACTS, prompts.TITLE_DOSSIER):
        head = prompts.section(title, "")
        start = user.find(head)
        if start >= 0:
            end = user.find("\n\n# ", start + len(head))
            parts.append(user[start : end if end >= 0 else len(user)])
    return "\n".join(parts)


def recent_messages(messages: Sequence[Message], k: int) -> list[Message]:
    """Return the last `k` chat messages, skipping system notices."""
    chat = [m for m in messages if m.speaker != "system"]
    return chat[-k:] if k > 0 else []


def blind_messages(messages: Sequence[Message], moderator: str) -> list[Message]:
    """Return what a member may see in a blind round: the moderator and the user only."""
    return [m for m in messages if m.speaker in (moderator, "user")]


def advocate_messages(
    messages: Sequence[Message], moderator: str, round_: int, advocate: str
) -> list[Message]:
    """Return what a devil's advocate sees in the blind round: also the others' takes."""
    return [
        m
        for m in messages
        if m.speaker in (moderator, "user")
        or (m.round == round_ and m.speaker not in ("system", advocate))
    ]


def _clip(text: str, max_chars: int) -> tuple[str, bool]:
    """Clip text to `max_chars` (keeping the start), marking the cut."""
    if len(text) <= max_chars:
        return text, False
    keep = max(max_chars - len(prompts.TRIMMED_MARK), 0)
    return text[:keep] + prompts.TRIMMED_MARK, True


def _clip_oldest(text: str, max_chars: int) -> str:
    """Clip a memory file to `max_chars` by dropping its oldest (first) lines."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(prompts.MEMORY_TRIMMED_MARK)
    if budget <= 0:
        return ""
    kept, _ = _fit_history(text.splitlines(), budget)
    return prompts.MEMORY_TRIMMED_MARK + "\n".join(kept) if kept else ""


def memory_text(private: str, shared: str, max_chars: int) -> str | None:
    """Render memory within `max_chars`; private notes get priority (brief §10)."""
    overhead = len(prompts.render_memory("x", "x") or "") - 2
    room = max(max_chars - overhead, 0)
    private = _clip_oldest(private, room)
    shared = _clip_oldest(shared, max(room - len(private), 0))
    return prompts.render_memory(private, shared)


def fit_messages(lines: Sequence[str], max_chars: int) -> tuple[list[str], int]:
    """Keep the newest messages that fit in `max_chars`; return them and the dropped count.

    A message longer than half the room is cut first: one huge paste would otherwise
    push out every older line with it.
    """
    longest = max_chars // 2
    return _fit_history([_clip(line, longest)[0] for line in lines], max_chars)


def _fit_history(lines: Sequence[str], max_chars: int) -> tuple[list[str], int]:
    """Keep the newest lines that fit in `max_chars`; return them and the dropped count."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost
    kept.reverse()
    return kept, len(lines) - len(kept)


def _shared_part(
    title: str, text: str | None, budget_tokens: int, share: float, remaining: int
) -> str:
    """Return a titled section capped at `share` of the budget (its start kept), or ""."""
    if text is None:
        return ""
    header = prompts.section(title, "")
    cap = min(int(tokens_to_chars(budget_tokens) * share), remaining) - len(header)
    return header + _clip(text, max(cap, 0))[0]


def build_prompt(
    *,
    system: str,
    topic: str,
    task: str,
    budget_tokens: int,
    state_text: str | None = None,
    history: Sequence[str] = (),
    history_title: str = prompts.TITLE_HISTORY,
    facts: str | None = None,
    facts_share: float = 0.25,
    dossier: str | None = None,
    dossier_share: float = 0.4,
    memory: tuple[str, str] = ("", ""),
    memory_share: float = 0.2,
) -> Prompt:
    """Assemble the user message for one call, trimmed to `budget_tokens`."""
    question = prompts.section(prompts.TITLE_QUESTION, topic)
    task_part = prompts.section(prompts.TITLE_TASK, task)
    # Section separators ("\n\n") are counted generously as 4 per section.
    remaining = tokens_to_chars(budget_tokens) - len(system) - len(question) - len(task_part) - 20
    dossier_part = _shared_part(
        prompts.TITLE_DOSSIER, dossier, budget_tokens, dossier_share, remaining
    )
    remaining -= len(dossier_part)
    facts_part = _shared_part(prompts.TITLE_FACTS, facts, budget_tokens, facts_share, remaining)
    remaining -= len(facts_part)
    memory_part = ""
    if any(m.strip() for m in memory):
        header = prompts.section(prompts.TITLE_MEMORY, "")
        cap = min(int(tokens_to_chars(budget_tokens) * memory_share), remaining) - len(header)
        body = memory_text(*memory, max(cap, 0))
        memory_part = header + body if body else ""
        remaining -= len(memory_part)
    state_part, clipped = "", False
    if state_text is not None:
        header = prompts.section(prompts.TITLE_STATE, "")
        body, clipped = _clip(state_text, max(remaining - len(header), 0))
        state_part = header + body
        remaining -= len(state_part)
    header = prompts.section(history_title, "")
    kept, dropped = fit_messages(history, max(remaining - len(header), 0))
    history_part = header + "\n".join(kept) if kept else ""
    parts = [question, dossier_part, facts_part, memory_part, state_part, history_part, task_part]
    user = "\n\n".join(p for p in parts if p)
    return Prompt(system=system, user=user, dropped_messages=dropped, state_clipped=clipped)


def render_history(messages: Sequence[Message], nick_of: NickOf) -> list[str]:
    """Render messages as chat lines for a prompt."""
    return [prompts.chat_line(m, nick_of) for m in messages]


def state_text(state: DebateState, nick_of: NickOf) -> str | None:
    """Render the debate state for a prompt, or None before the moderator built one."""
    if state == DebateState():
        return None
    return prompts.render_debate_state(state, nick_of)


def facts_text(notes: Sequence[ResearchNote]) -> str | None:
    """Render the session's research for prompts (newest first), or None if there is none."""
    if not notes:
        return None
    entries = [(n.focus, f"{n.ts.astimezone():%Y-%m-%d}", without_missing(n.text)) for n in notes]
    return prompts.render_research(entries)


def dossier_text(files: Sequence[DossierFile]) -> str | None:
    """Render the session's dossier for prompts, or None if nothing is attached."""
    if not files:
        return None
    return prompts.render_dossier([(f.name, f.text, f.summarized, f.stats) for f in files])


def prompt_tokens(prompt: Prompt) -> int:
    """Estimate the total input tokens of a built prompt."""
    return estimate_tokens(prompt.system) + estimate_tokens(prompt.user)
