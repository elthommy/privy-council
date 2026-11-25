"""Replay the moderator's round updates of a saved session with the current prompts.

Checks how the moderator judges a debate (consensus, new arguments, whether the
decision is still open, stop) without running a whole session: for each chosen
round, the engine is rebuilt from the transcript as it stood when that round
ended, and only the moderator's update is sent. Each round costs one Claude call
(two if the reply needs a repair); --dry-run makes none and prints prompt sizes.

    uv run python scripts/replay_moderation.py SESSION_DIR --rounds 3-6 [--root DIR]

Use a --root whose data dir is a scratch dir: calls go into its usage.jsonl, and
without --memory the moderator's memory is empty. --memory DIR replays with the
memory under DIR/memory (a copy of it as it stood then; the replay never writes it).
--runs N sends each round's update N times; the stall count follows the last run.

--pick-read: when an update asks for no page read, also ask a local moderator which
cited page holds a missing figure (the engine's page pick; one more local call, no
page is fetched). Needs a root with the document reader on.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import tempfile
from pathlib import Path

from privy_council.backends.base import ChatMessage, CompletionResult
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.engine import dynamics
from privy_council.engine.budget import Ledger
from privy_council.engine.moderation import decide
from privy_council.engine.protocol import MODERATE_MIN_OUTPUT_TOKENS, DebateEngine
from privy_council.engine.resume import messages_of
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.local_only import apply_local_only
from privy_council.memory.store import MemoryStore
from privy_council.models.moderator import ModeratorDecision
from privy_council.models.session import (
    DebateState,
    Event,
    MessageEvent,
    Phase,
    ResearchAdded,
    SessionMeta,
    SessionState,
    StateUpdated,
)
from privy_council.storage.sessions import SessionStore

_ADDRESS = re.compile(r"^@\S+\s+")
_SUMMON = re.compile(r"Type /summon (\S+) to allow it")


def _rounds(spec: str) -> list[int]:
    """Parse "3-6" or "4" into a list of round numbers."""
    first, _, last = spec.partition("-")
    return list(range(int(first), int(last or first) + 1))


def _asked(events: list[Event]) -> tuple[list[str], list[str]]:
    """Return the questions the moderator put to the user, and the personas it proposed."""
    asked: list[str] = []
    summons: list[str] = []
    for e in events:
        if not (isinstance(e, MessageEvent) and e.message.to_user):
            continue
        for line in e.message.text.splitlines():
            if match := _SUMMON.search(line):
                summons.append(match.group(1))
            elif line.strip():
                asked.append(_ADDRESS.sub("", line.strip()))
    return asked, summons


def _meta_at(meta: SessionMeta, events: list[Event], round_: int, stalled: int) -> SessionMeta:
    """Return the session meta as it stood when `round_` ended, before the update."""
    updates = [i for i, e in enumerate(events) if isinstance(e, StateUpdated)]
    before = events[: updates[round_ - 1]]
    prior = events[updates[round_ - 2]] if round_ >= 2 else None
    asked, summons = _asked(before)
    out = meta.model_copy(deep=True)
    out.state = SessionState.PAUSED
    rt = out.runtime
    rt.phase, rt.round, rt.round_in_progress, rt.pending = Phase.ROUND, round_, False, []
    rt.debate_state = prior.state if isinstance(prior, StateUpdated) else DebateState()
    rt.moderated_upto = len(messages_of(events[: updates[round_ - 2] + 1])) if prior else 0
    rt.stalled_rounds, rt.early_stop, rt.stop_reason = stalled, False, None
    rt.asked_questions, rt.proposed_summons = asked, summons
    rt.estimates = {
        slug: [r for r in records if r.round <= round_] for slug, records in rt.estimates.items()
    }
    # Only the research done by then: later notes would answer what the update asks for.
    rt.research = [e.note for e in before if isinstance(e, ResearchAdded)]
    return out


async def _replay(engine: DebateEngine, round_: int, dry_run: bool) -> ModeratorDecision | None:
    """Send the moderator's update for the engine's round; print and return its decision."""
    last_round = round_ >= engine.council.protocol.max_rounds
    stats = dynamics.member_stats(engine.messages, list(engine.members), round_)
    request = engine._moderate_request(stats, last_round)
    print(f"\n== round {round_}: prompt {len(request.system) + len(request.user)} chars")
    if dry_run:
        return None

    async def call(messages: list[ChatMessage], purpose: str) -> CompletionResult:
        """Send the moderator conversation in JSON mode, as the engine does."""
        result, _ = await engine._complete(
            engine.moderator,
            request,
            purpose,
            messages=messages,
            json_schema=engine._moderator_schema(),
            max_output=MODERATE_MIN_OUTPUT_TOKENS,
        )
        return result

    decision, problems = await decide(call, request.user, engine.members)
    for problem in problems:
        print(f"   problem: {problem}")
    if decision is None:
        return None
    # As the engine does: user facts limited to the user's own, then per-person caps,
    # answered questions, a guest question.
    state = engine._settled_state(decision)
    dropped = [f for f in decision.debate_state.user_facts if f not in state.user_facts]
    print(f"   user_facts dropped by the engine: {dropped or '(none)'}")
    return engine._settle_questions(decision.model_copy(update={"debate_state": state}))


def _report(
    decision: ModeratorDecision, round_: int, stalled: int, stall_limit: int, last_round: bool
) -> None:
    """Print the moderator's judgment and what the engine would do with it."""
    print(f"   consensus={decision.consensus.value} decision_open={decision.decision_open}")
    print(f"   stop={decision.stop} ({decision.stop_reason})")
    print("   user_facts:")
    for fact in decision.debate_state.user_facts or ["(none)"]:
        print(f"     - {fact}")
    state = decision.debate_state
    print("   disagreements:")
    for d in state.disagreements:
        print(f"     - {d.topic}: {d.sides}")
    print("   open_questions:")
    for q in state.open_questions or ["(none)"]:
        print(f"     - {q}")
    print("   questions_for_user:")
    for q in decision.questions_for_user or ["(none)"]:
        print(f"     - {q}")
    print("   research:")
    for focus in decision.research or ["(none)"]:
        print(f"     - {focus}")
    print("   documents:")
    for d in decision.documents:
        print(f"     - {d.url} for: {d.look_for}")
    if not decision.documents:
        print("     - (none)")
    print("   next_turns:")
    for turn in decision.next_turns:
        print(f"     - @{turn.speaker}: {turn.prompt}")
    print("   flaws:")
    for flaw in decision.flaws:
        print(f'     - {flaw.member} ({flaw.kind}) "{flaw.quote}": {flaw.note}')
    if not decision.flaws:
        print("     - (none)")
    print("   new_arguments:")
    for arg in decision.new_arguments or ["(none)"]:
        print(f"     - {arg}")
    print(f"   stalled rounds in a row: {stalled}")
    if last_round:
        print("   => last round: max_rounds ends the debate here anyway")
    elif round_ > 1 and (decision.stop or 0 < stall_limit <= stalled):
        print("   => the engine would stop the debate early here")


def _engine(
    settings: Settings, meta: SessionMeta, events: list[Event], memory_dir: Path | None = None
) -> DebateEngine:
    """Build an engine on the saved session's transcript, with no store.

    Memory is empty unless `memory_dir` is given. A local-only session replays local-only
    too (its seat map, no call to Claude).
    """
    personas, _ = load_personas(settings.personas_dir)
    councils, _ = load_councils(settings.councils_dir, personas)
    if meta.local_only:
        moderators = {c.moderator for c in councils.values()}
        settings, personas = apply_local_only(settings, personas, moderators)
    ledger = Ledger.in_data_dir(settings.data_dir)
    registry = BackendRegistry(settings, ledger, local_only=meta.local_only)
    memory = MemoryStore(memory_dir or Path(tempfile.mkdtemp(prefix="council-replay-")))
    return DebateEngine(
        settings=settings,
        council=councils[meta.council],
        personas=personas,
        registry=registry,
        topic=meta.topic,
        memory=memory,
        restored=(meta, events),
    )


async def main() -> None:
    """Replay the chosen rounds in order, carrying the stall count from one to the next."""
    parser = argparse.ArgumentParser(description="Replay moderator round updates.")
    parser.add_argument("session", type=Path, help="Session directory (data/sessions/...)")
    parser.add_argument("--rounds", default="2-6", help='Rounds to replay, e.g. "3-6"')
    parser.add_argument("--root", type=Path, default=None, help="Directory with config.toml")
    parser.add_argument("--memory", type=Path, default=None, help="Read DIR/memory")
    parser.add_argument("--runs", type=int, default=1, help="Updates sent per round")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts, call nothing")
    parser.add_argument(
        "--pick-read", action="store_true", help="Also run a local moderator's page pick"
    )
    args = parser.parse_args()
    settings = load_settings(args.root)
    store = SessionStore.open(args.session, {})
    meta, events = store.load_meta(), store.events()
    rounds = _rounds(args.rounds)
    first = [e for e in events if isinstance(e, StateUpdated)][rounds[0] - 2 : rounds[0] - 1]
    stalled = first[0].stalled_rounds if first and rounds[0] > 1 else 0
    at, before_round = 0, stalled
    for round_ in [r for r in rounds for _ in range(args.runs)]:
        if round_ != at:
            at, before_round = round_, stalled
        updates = [i for i, e in enumerate(events) if isinstance(e, StateUpdated)]
        engine = _engine(
            settings,
            _meta_at(meta, events, round_, before_round),
            events[: updates[round_ - 1]],
            args.memory,
        )
        decision = await _replay(engine, round_, args.dry_run)
        if decision is None:
            continue
        stats = dynamics.member_stats(engine.messages, list(engine.members), round_)
        progress = round_ == 1 or not dynamics.stalled(
            decision.new_arguments, decision.decision_open, stats, decision.position_changes
        )
        stalled = 0 if progress else before_round + 1
        protocol = engine.council.protocol
        marked = [s.slug for s in stats if s.updated]
        _report(decision, round_, stalled, protocol.stall_rounds, round_ >= protocol.max_rounds)
        print(f"   update markers that count: {marked or '(none)'}")
        print(f"   position_changes (moderator): {decision.position_changes}")
        if args.pick_read and not decision.documents and round_ < protocol.max_rounds:
            picks = await engine._picked_reads()
            print("   page pick (local moderator):")
            for d in picks or []:
                print(f"     - {d.url} for: {d.look_for}")
            if not picks:
                print("     - (none)")


if __name__ == "__main__":
    asyncio.run(main())
