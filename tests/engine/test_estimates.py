"""M17 estimate lines end to end: task rules, ledger, split, moderator and synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import CompletionRequest
from privy_council.backends.fake import Reply
from privy_council.models.council import ProtocolSettings
from privy_council.models.flaw import FlawKind
from privy_council.models.persona import Persona
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas

SPLIT = "lenders refuse to extend the loan by June"

# What each member writes, by round (made-up numbers).
TURNS: dict[str, list[str]] = {
    # Round 2 moves 12 -> 6 with no update marker.
    "rigor": [
        "Position: unlikely.\n**Estimate:** 12% (7-20%)",
        "Position: lower.\n**Estimate:** 6%",
    ],
    # Round 2's split implies 11%, but the estimate says 30%.
    "compounder": [
        "Position: possible.\n**Estimate:** 20% (10-30%)",
        "**Split:** P(event) 40%; if it happens 20%; if it doesn't 5%\n**Estimate:** 30% (25-35%)",
    ],
    "pragmatist": [
        "No number.",
        "I update: higher.\n**Split:** P(event) 50%; if it happens 30%; "
        "if it doesn't 10%\n**Estimate:** 20% (15-25%)",
    ],
    "contrarian": ["Playing devil's advocate: lenders are the weak link.", "Still the lenders."],
}


def responder(decisions: list[dict[str, Any]]) -> Any:
    """Members answer from TURNS by round; the rest is the standard script."""
    base = standard_responder(list(decisions))
    spoken: dict[str, int] = {}

    def respond(req: CompletionRequest) -> Reply:
        """Answer one request."""
        slug = req.tag.persona or ""
        if req.tag.purpose == "turn" and slug in TURNS:
            n = spoken.get(slug, 0)
            spoken[slug] = n + 1
            return TURNS[slug][min(n, 1)]
        if req.tag.purpose == "synthesis":
            return "## Consensus\n" + req.messages[-1].content
        return base(req)

    return respond


def all_four() -> list[tuple[str, str]]:
    """A round where every member speaks."""
    return [(s, "go on") for s in TURNS]


def harness(tmp_path: Path, personas: dict[str, Persona], **config: Any) -> Harness:
    """A finance-style council with estimate lines, two rounds."""
    first = {**decision(all_four()), "split": SPLIT}
    second = {**decision(stop=True, reason="done"), "split": "ignored: already set"}
    return Harness(
        tmp_path,
        personas,
        responder([first, second]),
        protocol=ProtocolSettings(max_rounds=2, estimate="probability"),
        config={"engine": config} if config else None,
    )


def turn_task(h: Harness, slug: str, index: int) -> str:
    """Return the task section of a member's `index`-th turn prompt."""
    turns = [r for r in h.requests() if r.tag.purpose == "turn" and r.tag.persona == slug]
    return turns[index].messages[-1].content.split("# Your task")[-1]


async def test_estimate_lines_run_through_a_session(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = harness(tmp_path, personas)
    for _ in range(4):  # opening, round 1 + update, round 2 + stop, synthesis
        await h.engine.step()
    rt = h.engine.meta.runtime

    # Tasks: the estimate line, the split once set, none for the devil's advocate.
    assert "**Estimate:** N% (low-high%)" in turn_task(h, "rigor", 0)
    assert SPLIT in turn_task(h, "rigor", 1) and "**Split:**" in turn_task(h, "rigor", 1)
    assert "Don't give your own probability" in turn_task(h, "contrarian", 0)
    assert "**Estimate:**" not in turn_task(h, "contrarian", 0)

    # The ledger and the split (the second split is ignored).
    assert rt.split == SPLIT
    assert [r.estimate.value for r in rt.estimates["rigor"]] == [12, 6]
    assert rt.estimates["compounder"][-1].split is not None
    assert "contrarian" not in rt.estimates and len(rt.estimates["pragmatist"]) == 1
    notices = [m.text for m in h.engine.messages if m.speaker == "system"]
    assert any(f"split the question on: {SPLIT}" in n for n in notices)

    # Flaws: logged for /flaws, but not shown to the moderator by default.
    kinds = {(f.member, f.kind) for f in rt.flaws if f.by == "engine"}
    assert ("rigor", FlawKind.UNMARKED_CHANGE) in kinds
    assert ("compounder", FlawKind.ESTIMATE_INCOHERENT) in kinds
    assert ("pragmatist", FlawKind.MISSING_ESTIMATE) in kinds
    assert not any(
        f.member == "contrarian" and f.kind is FlawKind.MISSING_ESTIMATE for f in rt.flaws
    )
    # The estimate and split numbers are the members' own: never flagged as unsourced.
    assert not any(f.kind is FlawKind.UNSOURCED for f in rt.flaws)
    moderate = [r for r in h.requests() if r.tag.purpose == "moderate"]
    first_update = moderate[0].messages[0].content
    assert "- @Rigor: 12% (7-20%)" in first_update
    assert "devil's advocate: gives no estimate" in first_update
    assert "- split: null, unless the members disagree" in first_update
    assert "split: always null: the split is already set" not in first_update
    assert "Engine flags on estimate lines" not in moderate[1].messages[0].content

    # Synthesis: last estimates, the median, the devil's advocate rule.
    synthesis = next(r for r in h.requests() if r.tag.purpose == "synthesis")
    text = synthesis.messages[-1].content
    assert "- @Rigor (round 2): 6%" in text
    assert "Council estimate (engine): median 20%, members' estimates from 6% to 30%." in text
    assert "@Contrarian argued a devil's advocate role" in text

    # Saved with the session, so a resume keeps it.
    saved = h.store.load_meta().runtime
    assert saved.split == SPLIT and len(saved.estimates["rigor"]) == 2


async def test_no_unmarked_change_after_the_user_spoke(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = harness(tmp_path, personas)
    for _ in range(2):  # opening, round 1 + update
        await h.engine.step()
    h.engine.say("Any default counts, whatever the cause.")  # the question may have changed
    await h.engine.step()  # round 2: Rigor moves 12 -> 6 with no marker
    kinds = {(f.member, f.kind) for f in h.engine.meta.runtime.flaws}
    assert ("rigor", FlawKind.UNMARKED_CHANGE) not in kinds
    assert ("compounder", FlawKind.ESTIMATE_INCOHERENT) in kinds  # other checks still run


async def test_estimate_flaws_reach_the_moderator_when_enabled(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = harness(tmp_path, personas, estimate_flaws_to_moderator=True)
    for _ in range(3):
        await h.engine.step()
    second = [r for r in h.requests() if r.tag.purpose == "moderate"][1].messages[0].content
    assert "Engine flags on estimate lines" in second
    assert "- @Rigor: The estimate moved from 12% to 6%" in second
    assert "40% x 20% + 60% x 5% = 11%, but the estimate says 30%." in second


async def test_councils_without_estimates_are_unchanged(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder([{**decision(stop=True, reason="x"), "split": SPLIT}]),
        protocol=ProtocolSettings(max_rounds=1),
    )
    for _ in range(3):
        await h.engine.step()
    assert "Estimate:" not in turn_task(h, "rigor", 0)
    assert "Fresh facts:" not in turn_task(h, "rigor", 0)  # no research, no tag rule
    moderate = next(r for r in h.requests() if r.tag.purpose == "moderate")
    assert "- split: always null." in moderate.messages[0].content
    rt = h.engine.meta.runtime
    assert rt.split is None and rt.estimates == {}
    assert not any(f.kind is FlawKind.MISSING_ESTIMATE for f in rt.flaws)
