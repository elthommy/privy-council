"""Live M5 acceptance check: a second session uses notes from the first; reset works.

Spends Claude quota (~5 calls: opening, synthesis, Chair and Compounder notes,
shared-fact proposal) and runs local models. Uses the configured data dir, so
the calls land in data/usage.jsonl; memory is wiped at the end unless --no-reset.

    uv run python scripts/live_memory_check.py [--council finance] [--no-reset]

Session 1: opening, one user line (made-up context), an /ask to a local member,
/close, /keep all.
Session 2: /ask a local member what it remembers (no Claude call) and check that
its prompt carried the notes and shared facts. Then reset and ask again in a
fresh session 3 (in session 2 the earlier answer is in the chat history).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from privy_council.backends.base import BackendHealth, CompletionRequest, CompletionResult
from privy_council.backends.claude_cli import ClaudeCliBackend
from privy_council.backends.ollama import OllamaBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings, load_settings
from privy_council.engine.budget import Ledger
from privy_council.engine.protocol import DebateEngine
from privy_council.loaders.councils import load_councils
from privy_council.loaders.personas import load_personas
from privy_council.memory.store import MemoryStore
from privy_council.models.persona import BackendKind
from privy_council.models.session import Event, MemoryProposed, MessageEvent
from privy_council.storage.sessions import SessionStore

TOPIC = "Should I build a 6-month emergency fund before investing in index funds?"
USER_LINE = "For context: I'm a freelancer with irregular income, I rent, and I have no debt."
QUESTION = "What do you remember about my situation and our last discussion?"
# A local member, so its /ask turns cost no Claude quota. Only members who spoke
# write notes, so session 1 asks it a question too (no debate rounds needed).
ASKED = "contrarian"
FIRST_QUESTION = "What's your first reaction, in two sentences?"
FOLLOW_UP_TOPIC = "What should I do with a year-end bonus?"
MEMORY_TITLE = "# Memory from past sessions"


class Recording:
    """Wraps a backend and keeps every request, so prompts can be checked."""

    def __init__(self, inner: ClaudeCliBackend | OllamaBackend) -> None:
        """Wrap `inner`."""
        self.inner = inner
        self.name = inner.name
        self.requests: list[CompletionRequest] = []

    async def complete(self, req: CompletionRequest) -> CompletionResult:
        """Record the request, then delegate."""
        self.requests.append(req)
        return await self.inner.complete(req)

    async def health(self) -> BackendHealth:
        """Delegate."""
        return await self.inner.health()


def show(event: Event) -> None:
    """Print chat lines and proposals as they happen."""
    if isinstance(event, MessageEvent):
        msg = event.message
        text = msg.text if len(msg.text) < 600 else msg.text[:600] + " […]"
        print(f"<{msg.speaker}> {text}", flush=True)
    elif isinstance(event, MemoryProposed):
        print("PROPOSED:", *(event.facts or [event.rewrite or ""]), sep="\n  ", flush=True)


def make_engine(
    settings: Settings, council: str, registry: BackendRegistry, topic: str
) -> DebateEngine:
    """Build a saved session engine on the real backends."""
    personas, _ = load_personas(settings.personas_dir)
    preset = load_councils(settings.councils_dir, personas)[0][council]
    nicks = {slug: p.name for slug, p in personas.items()}
    store = SessionStore.create(settings.data_dir, preset.name, topic, datetime.now(UTC), nicks)
    engine = DebateEngine(
        settings=settings,
        council=preset,
        personas=personas,
        registry=registry,
        topic=topic,
        store=store,
    )
    engine.subscribe(show)
    return engine


def ask_prompt(ollama: Recording) -> str:
    """Return the prompt of the last /ask sent to the asked member."""
    req = [r for r in ollama.requests if r.tag.purpose == "ask" and r.tag.persona == ASKED][-1]
    return req.messages[0].content


def check(label: str, ok: bool) -> bool:
    """Print a check result."""
    print(f"{'PASS' if ok else 'FAIL'}  {label}", flush=True)
    return ok


async def main(council: str, reset: bool) -> int:
    """Run both sessions and the reset; return the exit code."""
    settings = load_settings()
    ledger = Ledger.in_data_dir(settings.data_dir)
    claude = Recording(ClaudeCliBackend(settings.claude, ledger))
    ollama = Recording(OllamaBackend(settings.ollama))
    overrides = {BackendKind.CLAUDE_CLI: claude, BackendKind.OLLAMA: ollama}
    registry = BackendRegistry(settings, ledger, overrides=overrides)
    store = MemoryStore(settings.data_dir)
    results: list[bool] = []
    try:
        print("=== session 1 ===", flush=True)
        first = make_engine(settings, council, registry, TOPIC)
        await first.step()
        first.say(USER_LINE)
        await first.ask(ASKED, FIRST_QUESTION)
        await first.close()
        await first.keep("all")
        print("\n=== memory files ===", flush=True)
        for slug in store.personas_with_notes():
            print(f"[{slug}]\n{store.read_private(slug)}", flush=True)
        print(f"[shared]\n{store.read_shared()}", flush=True)
        notes = store.read_private(ASKED).strip().splitlines()
        if not check(f"{ASKED} has private notes", bool(notes)):
            return 1
        results.append(check("shared facts were saved", bool(store.read_shared().strip())))

        print("\n=== session 2 ===", flush=True)
        second = make_engine(settings, council, registry, FOLLOW_UP_TOPIC)
        await second.ask(ASKED, QUESTION)
        prompt = ask_prompt(ollama)
        results.append(check("session 2 prompt has the memory section", MEMORY_TITLE in prompt))
        results.append(check(f"...with {ASKED}'s notes", all(n[2:] in prompt for n in notes)))
        results.append(check("...and the shared facts", "Facts the user stated" in prompt))

        if reset:
            print("\n=== reset ===", flush=True)
            results.append(check("reset_all removed memory", store.reset_all()))
            print("\n=== session 3 (after reset) ===", flush=True)
            third = make_engine(settings, council, registry, FOLLOW_UP_TOPIC)
            await third.ask(ASKED, QUESTION)
            results.append(check("no memory after reset", MEMORY_TITLE not in ask_prompt(ollama)))
    finally:
        await registry.aclose()
    claude_calls = len(claude.requests)
    print(f"\nClaude calls: {claude_calls}; local calls: {len(ollama.requests)}", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--council", default="finance")
    parser.add_argument("--no-reset", dest="reset", action="store_false")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.council, args.reset)))
