"""M19: a local member fixes a flagged draft before it is posted (`repair_turns`)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from privy_council.backends.base import BackendTimeoutError, CompletionRequest
from privy_council.models.flaw import FlawKind
from privy_council.models.persona import Persona
from privy_council.models.session import TextDelta, TurnRevised
from privy_council.tui.commands import flaws_text
from tests.engine.test_protocol import Harness, Reply, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import HangingBackend, Resumed, kill_snapshot, nth_call

DRAFT = "Rigor: build the fund first, three months of expenses."
FIXED = "Build the fund first, three months of expenses."
REPAIR_ON = {"engine": {"repair_turns": "local"}}


def scripted(turns: dict[str, Reply], fix: Reply = FIXED) -> Callable[[CompletionRequest], Reply]:
    """Answer member turns from `turns` (by persona), fix calls with `fix`."""
    base = standard_responder([decision([("rigor", "go")]) for _ in range(4)])

    def respond(req: CompletionRequest) -> Reply:
        """Return the scripted reply for this request."""
        if req.tag.purpose == "fix":
            return fix
        if req.tag.purpose == "turn" and req.tag.persona in turns:
            return turns[req.tag.persona]
        return base(req)

    return respond


async def blind_round(
    tmp_path: Path,
    personas: dict[str, Persona],
    respond: Callable[[CompletionRequest], Reply],
    config: dict[str, Any] | None = REPAIR_ON,
) -> Harness:
    """Run the opening and the blind round."""
    h = Harness(tmp_path, personas, respond, config=config)
    await h.engine.step()
    await h.engine.step()
    return h


def purposes(h: Harness, persona: str) -> list[str]:
    """Return the purposes of a persona's calls, in order."""
    return [r.tag.purpose for r in h.requests() if r.tag.persona == persona]


async def test_a_flagged_local_draft_is_rewritten_and_the_draft_kept(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = await blind_round(tmp_path, personas, scripted({"rigor": DRAFT}))
    assert h.texts("rigor") == [FIXED]
    assert purposes(h, "rigor") == ["turn", "fix"]
    fix = next(r for r in h.ollama.requests if r.tag.purpose == "fix")
    assert [m.role for m in fix.messages] == ["user", "assistant", "user"]
    assert fix.messages[1].content == DRAFT
    assert "don't start with your own name" in fix.messages[2].content
    [revision] = h.engine.meta.runtime.revisions
    assert (revision.member, revision.draft) == ("rigor", DRAFT)
    assert [f.kind for f in revision.flaws] == [FlawKind.SELF_PREFIX]
    assert not [f for f in h.engine.meta.runtime.flaws if f.member == "rigor"]
    events = [e for e in h.events if isinstance(e, TurnRevised)]
    assert [e.revision for e in events] == [revision]
    assert '"kind":"revised"' in (h.store.directory / "transcript.jsonl").read_text()


async def test_a_rewrite_that_fixes_nothing_keeps_the_draft(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = await blind_round(tmp_path, personas, scripted({"rigor": DRAFT}, fix="Rigor: same."))
    assert h.texts("rigor") == [DRAFT]
    assert h.engine.meta.runtime.revisions == []
    assert [f.kind for f in h.engine.meta.runtime.flaws if f.member == "rigor"] == [
        FlawKind.SELF_PREFIX
    ]


async def test_a_failed_fix_call_posts_the_draft(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    respond = scripted({"rigor": DRAFT}, fix=BackendTimeoutError("too slow"))
    h = await blind_round(tmp_path, personas, respond)
    assert h.texts("rigor") == [DRAFT]
    assert h.engine.meta.runtime.revisions == []


async def test_claude_members_and_unrepairable_flags_get_no_fix_call(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    turns: dict[str, Reply] = {
        "compounder": "Compounder: index funds first.",  # Claude: never repaired
        "rigor": "Expect about €123 a year in fees.",  # unsourced: not repairable
    }
    h = await blind_round(tmp_path, personas, scripted(turns))
    assert purposes(h, "compounder") == ["turn"]
    assert purposes(h, "rigor") == ["turn"]
    assert h.texts("compounder") == ["Compounder: index funds first."]
    assert h.texts("rigor") == ["Expect about €123 a year in fees."]


async def test_local_drafts_stream_only_with_repair_off(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    def streamed(h: Harness) -> set[str]:
        """Return the speakers whose text was streamed."""
        return {e.speaker for e in h.events if isinstance(e, TextDelta)}

    on = await blind_round(tmp_path / "on", personas, scripted({}))
    off = await blind_round(tmp_path / "off", personas, scripted({}), config=None)
    assert "rigor" not in streamed(on)
    assert "rigor" in streamed(off)
    assert "fix" not in [r.tag.purpose for r in off.requests()]


async def test_a_crash_between_draft_and_post_reruns_the_turn(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    respond = scripted({"rigor": DRAFT})
    hanging = HangingBackend(nth_call(1, "rigor", "fix"), responder=respond, name="ollama")
    h = Harness(tmp_path, personas, respond, ollama=hanging, config=REPAIR_ON)
    await h.engine.step()
    task = asyncio.create_task(h.engine.step())  # blind round: Rigor's fix call hangs
    await hanging.hung.wait()
    killed = kill_snapshot(h)
    task.cancel()

    r = Resumed(h, killed, personas, respond)
    assert "rigor" in [t.speaker for t in r.engine.meta.runtime.pending]
    assert not [m for m in r.engine.messages if m.speaker == "rigor"]
    await r.engine.step()
    assert [m.text for m in r.engine.messages if m.speaker == "rigor"] == [FIXED]
    assert r.turns().count(("rigor", "fix")) == 1


async def test_flaws_lists_the_fixed_drafts_with_counts(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = await blind_round(tmp_path, personas, scripted({"rigor": DRAFT}))
    rt = h.engine.meta.runtime
    text = flaws_text(rt.flaws, h.engine.nick, rt.revisions)
    assert "-- fixed before posting" in text
    assert "turns fixed: Rigor 1" in text
    assert '[self_prefix]: "Rigor:"' in text
