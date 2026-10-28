"""Running a comparison on fake backends: shared research, fixed rounds, stop/resume, judge."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from privy_council.backends.base import BackendError, BackendTimeoutError, CompletionRequest
from privy_council.backends.fake import FakeBackend, Reply
from privy_council.backends.registry import BackendRegistry
from privy_council.compare.runner import CompareRunner, Outcome, run_judge, variant_research
from privy_council.compare.spec import parse_variant
from privy_council.compare.store import CompareState, CompareStore, VariantRun, VariantStatus
from privy_council.config import Settings
from privy_council.engine import prompts
from privy_council.engine.budget import Ledger
from privy_council.engine.research import ResearchMode
from privy_council.models.council import CouncilPreset, ProtocolSettings
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import Phase, ResearchAdded, VoteTaken
from privy_council.storage.sessions import SESSIONS_DIR, SessionStore
from tests.engine.test_protocol import MEMBERS, TOPIC, decision
from tests.engine.test_protocol import personas as personas
from tests.engine.test_research import FACTS, research_result

DECISION = decision([("rigor", "answer @Compounder"), ("compounder", "answer @Rigor")])


def respond(req: CompletionRequest) -> Reply:
    """Answer every call by purpose; the moderator always proposes to stop (ignored)."""
    purpose = req.tag.purpose
    if purpose == "research":
        return research_result()
    if purpose == "opening":
        return "Restating the question."
    if purpose in ("moderate", "repair"):
        return {**DECISION, "stop": True, "stop_reason": "agreed", "research": ["more facts"]}
    if purpose == "vote":
        return {"position": f"{req.tag.persona} keeps its view", "confidence": 70}
    if purpose == "synthesis":
        return f"## Consensus\nKeep 3 months first ({req.tag.session})."
    return f"{req.tag.persona} on {req.model}"


class Rig:
    """A comparison wired to fake backends."""

    def __init__(
        self,
        tmp_path: Path,
        personas: dict[str, Persona],
        variants: list[str],
        *,
        rounds: int = 2,
        research: ResearchMode = ResearchMode.AUTO,
        responder: Callable[[CompletionRequest], Reply] = respond,
    ) -> None:
        """Create the comparison (not run yet)."""
        self.settings = Settings.model_validate({"root": tmp_path, "user": {"name": "User"}})
        self.claude = FakeBackend(responder=responder, name="claude")
        self.ollama = FakeBackend(responder=responder, name="ollama")
        self.registry = BackendRegistry(
            self.settings,
            Ledger.in_data_dir(self.settings.data_dir),
            overrides={BackendKind.CLAUDE_CLI: self.claude, BackendKind.OLLAMA: self.ollama},
        )
        self.preset = CouncilPreset(
            name="finance",
            moderator="chair",
            members=MEMBERS,
            protocol=ProtocolSettings(max_rounds=6),
        )
        self.personas = personas
        self.store = CompareStore.create(
            self.settings.data_dir, "finance", TOPIC, datetime.now(UTC)
        )
        self.store.save(
            CompareState(
                id=self.store.id,
                council="finance",
                topic=TOPIC,
                rounds=rounds,
                research=research.value,
                variants=[VariantRun(variant=parse_variant(v)) for v in variants],
            )
        )
        self.lines: list[str] = []
        self.on_line: Callable[[CompareRunner, str], None] | None = None

    def runner(self) -> CompareRunner:
        """Return a fresh runner (as a new `council compare resume` would build)."""
        runner: CompareRunner

        def progress(line: str) -> None:
            """Collect progress lines, letting a test react to them."""
            self.lines.append(line)
            if self.on_line is not None:
                self.on_line(runner, line)

        runner = CompareRunner(
            settings=self.settings,
            preset=self.preset,
            personas=self.personas,
            registry=self.registry,
            store=self.store,
            progress=progress,
        )
        return runner

    def state(self) -> CompareState:
        """Return the saved compare.json."""
        return self.store.load()

    def session(self, index: int) -> SessionStore:
        """Return a variant's session store."""
        name = self.state().variants[index].session
        assert name is not None
        return SessionStore.open(self.settings.data_dir / SESSIONS_DIR / name)

    def requests(self, session: str | None = None) -> list[CompletionRequest]:
        """Return every request, optionally of one session only."""
        reqs = self.claude.requests + self.ollama.requests
        return [r for r in reqs if session is None or r.tag.session == session]


async def test_every_variant_debates_from_the_same_facts(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rig = Rig(tmp_path, personas, ["base", "qwen:rigor=qwen3.6:27b"])
    assert await rig.runner().run() is Outcome.DONE

    state = rig.state()
    assert [v.status for v in state.variants] == [VariantStatus.DONE] * 2
    assert state.shared_research is not None and FACTS in state.shared_research[0].text
    # One search in all: the first variant's opening. The moderator's own searches
    # ("more facts") are relayed as questions instead, so facts never diverge.
    assert [r.tag.purpose for r in rig.requests()].count("research") == 1
    second = rig.session(1)
    assert any(isinstance(e, ResearchAdded) for e in second.events())
    opening = next(r for r in rig.requests(second.id) if r.tag.purpose == "opening")
    assert "$4,170/oz" in opening.messages[0].content


async def test_variants_run_fixed_rounds_vote_then_synthesize(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rig = Rig(tmp_path, personas, ["base", "qwen:rigor=qwen3.6:27b"], rounds=2)
    await rig.runner().run()

    for index in (0, 1):
        session = rig.session(index)
        meta = session.load_meta()
        assert meta.runtime.phase is Phase.DONE
        assert meta.runtime.round == 2  # the moderator's stop after round 1 is ignored
        purposes = [r.tag.purpose for r in rig.requests(session.id)]
        assert purposes.index("vote") < purposes.index("synthesis")
        assert not {"notes", "facts", "consolidate"} & set(purposes)  # memory off
        assert sum(isinstance(e, VoteTaken) for e in session.events()) == 1
    rigor = [r for r in rig.requests(rig.session(1).id) if r.tag.persona == "rigor"]
    assert rigor and {r.model for r in rigor} == {"qwen3.6:27b"}
    assert rig.session(1).load_meta().models["rigor"] == ["ollama:qwen3.6:27b"]


async def test_stop_lands_on_a_turn_boundary_and_resume_finishes(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    rig = Rig(tmp_path, personas, ["base", "noc:-contrarian"], rounds=2)

    def stop_in_round_two(runner: CompareRunner, line: str) -> None:
        """Ask to stop once the first variant's second round has started."""
        if line == "-- round 2 --" and rig.on_line is not None:
            rig.on_line = None
            runner.request_stop()

    rig.on_line = stop_in_round_two
    assert await rig.runner().run() is Outcome.STOPPED
    state = rig.state()
    assert state.variants[0].status is VariantStatus.RUNNING
    assert state.variants[1].session is None
    meta = rig.session(0).load_meta()
    assert meta.runtime.round == 2 and meta.runtime.round_in_progress

    assert await rig.runner().run() is Outcome.DONE
    assert [v.status for v in rig.state().variants] == [VariantStatus.DONE] * 2
    assert rig.session(0).load_meta().runtime.round == 2
    sessions = list((rig.settings.data_dir / SESSIONS_DIR).iterdir())
    assert len([d for d in sessions if not d.is_symlink()]) == 2


async def test_a_failing_step_is_retried_once_then_the_variant_fails(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    broken = [True]

    def flaky(req: CompletionRequest) -> Reply:
        """Time out on synthesis while `broken`."""
        if req.tag.purpose == "synthesis" and broken[0]:
            return BackendTimeoutError("synthesis timed out")
        return respond(req)

    rig = Rig(tmp_path, personas, ["base", "other:-contrarian"], rounds=1, responder=flaky)
    assert await rig.runner().run() is Outcome.FAILED
    first = rig.state().variants[0]
    assert first.status is VariantStatus.FAILED
    assert first.error is not None and "synthesis timed out" in first.error
    synth = [r for r in rig.requests() if r.tag.purpose == "synthesis"]
    assert len(synth) == 2  # tried, then retried once

    broken[0] = False
    assert await rig.runner().run() is Outcome.DONE
    assert rig.state().variants[0].error is None


async def test_stop_file_asks_for_a_stop(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rig = Rig(tmp_path, personas, ["a", "b:-contrarian"])
    runner = rig.runner()
    rig.store.stop_file.touch()
    await runner._watch_stop_file()
    assert runner.stopping


def test_only_the_first_variant_searches() -> None:
    state = CompareState(id="x", council="c", topic="t", rounds=1, research="required", variants=[])
    assert variant_research(state, 0) is ResearchMode.REQUIRED
    assert variant_research(state, 1) is ResearchMode.ON_DEMAND
    off = state.model_copy(update={"research": "off"})
    assert variant_research(off, 1) is ResearchMode.OFF


async def test_judge_sees_letters_not_models(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rig = Rig(tmp_path, personas, ["base", "qwen:rigor=qwen3.6:27b"], rounds=1)
    await rig.runner().run()
    verdict_text = "## Verdict\nA is clearer."
    rig.claude.script.append(verdict_text)

    verdict = await run_judge(rig.store, rig.settings, rig.registry)

    assert verdict == verdict_text
    req = rig.claude.requests[-1]
    assert req.tag.purpose == "compare_judge" and req.tag.session == f"compare:{rig.store.id}"
    assert req.system == prompts.COMPARE_JUDGE_SYSTEM
    text = req.messages[0].content
    assert '<variant letter="A">' in text and '<variant letter="B">' in text
    assert "$4,170/oz" in text  # the shared facts, to check figures against
    assert "qwen" not in text and "base" not in text
    assert rig.store.load_judge() == verdict_text
    assert rig.state().judged


async def test_judge_needs_two_syntheses(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rig = Rig(tmp_path, personas, ["a", "b:-contrarian"])
    with pytest.raises(BackendError, match="at least two"):
        await run_judge(rig.store, rig.settings, rig.registry)
