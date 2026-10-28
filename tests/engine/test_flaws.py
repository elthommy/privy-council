"""M11 flaws: the moderator's reports, flaw floor turns, the synthesis, end to end."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privy_council.backends.base import CompletionRequest, CompletionResult
from privy_council.backends.fake import Reply
from privy_council.engine import prompts
from privy_council.engine.moderation import (
    MAX_FLAWS,
    moderator_schema,
    parse_decision,
    with_flaw_turns,
)
from privy_council.loaders.personas import load_personas
from privy_council.models.council import ProtocolSettings
from privy_council.models.flaw import Flaw, FlawKind
from privy_council.models.session import FlawsFound, PendingTurn
from privy_council.tui.commands import flaws_text
from tests.conftest import REPO_ROOT
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas

MEMBERS = {
    s: p
    for s, p in load_personas(REPO_ROOT / "personas")[0].items()
    if s in ("rigor", "compounder", "pragmatist", "contrarian")
}


def flaw(member: str, quote: str = "q", kind: str = "arithmetic") -> dict[str, str]:
    """A flaw report as the moderator writes it."""
    return {"member": member, "kind": kind, "quote": quote, "note": "16 + 16 is 32."}


def reported(member: str, quote: str = "q") -> Flaw:
    """A flaw as the engine stores the moderator's report."""
    return Flaw(member=member, kind=FlawKind.ARITHMETIC, excerpt=quote, note="n", by="moderator")


def parsed(flaws: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Parse a decision carrying `flaws`; return (member, kind, quote) per kept flaw."""
    raw = {**decision([("rigor", "go")]), "flaws": flaws}
    res = CompletionResult(text="", structured=raw, latency_s=0, backend="f", model="m")
    return [(f.member, f.kind, f.quote) for f in parse_decision(res, MEMBERS).flaws]


def test_flaws_are_resolved_cleaned_and_capped() -> None:
    reports = [
        flaw("@Rigor", ' "32  GB max" '),
        flaw("ghost"),
        flaw("rigor", "32 GB max"),  # the same flaw again
        flaw("Compounder", "x", kind="Typo"),
        *(flaw("pragmatist", f"q{i}") for i in range(5)),
    ]
    got = parsed(reports)
    assert got[:2] == [("rigor", "arithmetic", "32 GB max"), ("compounder", "other", "x")]
    assert len(got) == MAX_FLAWS


def test_replies_without_flaws_report_none() -> None:
    assert parsed([]) == []
    res = CompletionResult(
        text="", structured=decision([("rigor", "go")]), latency_s=0, backend="f", model="m"
    )
    assert parse_decision(res, MEMBERS).flaws == []


def test_schema_requires_flaws_with_known_kinds() -> None:
    schema = moderator_schema()
    assert "flaws" in schema["required"]
    item = schema["$defs"]["ReportedFlaw"]
    assert item["properties"]["kind"]["enum"] == [
        "arithmetic",
        "contradiction",
        "logic",
        "unsourced",
        "other",
    ]
    assert item["required"] == ["member", "kind", "quote", "note"]


def test_flaw_turns_go_first_one_per_member_within_the_cap() -> None:
    planned = [
        PendingTurn(speaker="rigor", prompt="answer X"),
        PendingTurn(speaker="contrarian", prompt="answer Y"),
    ]
    flaws = [
        reported("compounder", "a"),
        reported("compounder", "b"),
        reported("rigor", "c"),
        reported("pragmatist", "d"),
    ]
    turns = with_flaw_turns(planned, flaws, 2)
    assert [t.speaker for t in turns] == ["compounder", "rigor", "contrarian"]
    assert (
        turns[0].prompt == 'you wrote "a". n. Check it and correct it if it is wrong; say '
        "plainly whether it stands."
    )
    assert turns[1].prompt.endswith(" Then: answer X")
    assert with_flaw_turns(planned, flaws, 0) == planned


def test_flaw_turn_prompt_in_french() -> None:
    prompt = prompts.flaw_turn_prompt("40 Go", "", "fr")
    assert (
        prompt == "tu as écrit « 40 Go ». Vérifie-le et corrige-le s'il est faux ; dis "
        "clairement s'il tient."
    )


def test_synthesis_errors_only_when_there_are_flaws() -> None:
    assert prompts.synthesis_errors([], str, "en") is None
    text = prompts.synthesis_errors([reported("rigor", "32 GB max")], str.capitalize, "fr")
    assert text is not None
    assert '- round 0, Rigor (arithmetic): "32 GB max" — n' in text
    assert '"## Erreurs relevées" right after "## Désaccords restants"' in text


def test_flaws_text_lists_both_sources() -> None:
    engine_flaw = Flaw(
        member="rigor", kind=FlawKind.UNSOURCED, excerpt="max 32 GB.", note="32 GB", round=2
    )
    text = flaws_text([engine_flaw, reported("rigor", "q")], str.capitalize)
    assert 'r2 Rigor [unsourced, engine]: 32 GB in "max 32 GB."' in text
    assert 'r0 Rigor [arithmetic, moderator]: "q" (n)' in text
    assert flaws_text([], str) == "No flaws caught so far."


def responder_with(decisions: list[dict[str, Any]], member_text: str) -> Any:
    """Standard replies, but members say `member_text` and syntheses echo their task."""
    base = standard_responder(list(decisions))

    def respond(req: CompletionRequest) -> Reply:
        """Answer one request."""
        if req.tag.purpose == "turn":
            return member_text
        return base(req)

    return respond


async def test_engine_flags_turns_and_prompts_the_flaws(
    tmp_path: Path, personas: dict[str, Any]
) -> None:
    first = {**decision([("contrarian", "push back")]), "flaws": [flaw("rigor", "32 GB max")]}
    # Round 2's update flags Rigor's round-1 quote again, after Rigor answered it.
    again = [flaw("compounder", "40 GB"), flaw("rigor", "32 GB max")]
    second = {**decision(stop=True, reason="done"), "flaws": again}
    h = Harness(
        tmp_path,
        personas,
        responder_with([first, second], "The T490 maxes at 32 GB. @nick agrees."),
        config={"engine": {"max_flaw_prompts_per_round": 1}},
        protocol=ProtocolSettings(max_rounds=2),
    )
    await h.engine.step()  # opening
    await h.engine.step()  # blind round + update
    found = [e for e in h.events if isinstance(e, FlawsFound)]
    kinds = {f.kind for e in found for f in e.flaws}
    assert {FlawKind.UNSOURCED, FlawKind.NICK_LITERAL, FlawKind.ARITHMETIC} <= kinds
    moderate = next(r for r in h.claude.requests if r.tag.purpose == "moderate")
    assert '- @Rigor: "32 GB" in "The T490 maxes at 32 GB."' in moderate.messages[0].content
    assert "The engine asks the members to check the first 1 next round" in (
        moderate.messages[0].content
    )
    pending = h.engine.meta.runtime.pending
    assert [t.speaker for t in pending] == ["rigor", "contrarian"]
    assert pending[0].prompt.startswith('you wrote "32 GB max". 16 + 16 is 32.')
    # The text is never changed.
    assert h.texts("rigor") == ["The T490 maxes at 32 GB. @nick agrees."]
    await h.engine.step()  # round 2 + stop
    await h.engine.step()  # synthesis
    synthesis = next(r for r in h.claude.requests if r.tag.purpose == "synthesis")
    # Rigor answered the challenge in round 2, so only Compounder's error is left.
    errors = synthesis.messages[0].content.split("Errors you caught")[1]
    assert 'Compounder (arithmetic): "40 GB"' in errors and "32 GB max" not in errors
    assert "@Rigor" not in h.engine.meta.runtime.flaws[0].excerpt


async def test_an_unsourced_flag_on_a_figure_the_member_was_given_is_dropped(
    tmp_path: Path, personas: dict[str, Any]
) -> None:
    flags = [
        flaw("rigor", "You have 25,000 EUR saved", "unsourced"),  # the user said it
        flaw("rigor", "a 4% rate", "unsourced"),  # nobody did
    ]
    first = {**decision([("rigor", "go")]), "flaws": flags}
    text = "You have 25,000 EUR saved and a 4% rate."
    h = Harness(tmp_path, personas, responder_with([first, first], text))
    await h.engine.step()  # opening
    h.engine.say("I have 25,000 EUR saved.")
    await h.engine.step()  # blind round + update
    by_moderator = [f.excerpt for f in h.engine.meta.runtime.flaws if f.by == "moderator"]
    assert by_moderator == ["a 4% rate"]


async def test_an_arithmetic_flag_whose_note_says_it_holds_is_dropped(
    tmp_path: Path, personas: dict[str, Any]
) -> None:
    def note(quote: str, text: str) -> dict[str, str]:
        """An arithmetic flag with its note."""
        return {"member": "rigor", "kind": "arithmetic", "quote": quote, "note": text}

    flags = [
        note("25,000 EUR at 4% is 1,000 EUR", "25,000 x 4% = 1,000, which is correct."),
        # Says "correct", also "wrong", and the quote's 7,300 EUR is nobody's figure: kept.
        note("so 7,300 EUR a year", "1,000 is correct, but 7,300 is wrong."),
        note("a 4% rate", "4% of 25,000 is 1,000, not 1,200."),
    ]
    first = {**decision([("rigor", "go")]), "flaws": flags}
    text = "You have 25,000 EUR at 4% is 1,000 EUR, so 7,300 EUR a year, at a 4% rate."
    h = Harness(tmp_path, personas, responder_with([first, first], text))
    await h.engine.step()  # opening
    h.engine.say("I have 25,000 EUR saved at 4%.")
    await h.engine.step()  # blind round + update
    by_moderator = [f.excerpt for f in h.engine.meta.runtime.flaws if f.by == "moderator"]
    assert by_moderator == ["so 7,300 EUR a year", "a 4% rate"]


async def test_no_errors_heading_without_moderator_flaws(
    tmp_path: Path, personas: dict[str, Any]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        responder_with([decision(stop=True, reason="x")], "Fine."),
        protocol=ProtocolSettings(max_rounds=1),
    )
    for _ in range(3):
        await h.engine.step()
    synthesis = next(r for r in h.claude.requests if r.tag.purpose == "synthesis")
    assert "Errors caught" not in synthesis.messages[0].content


async def test_cross_task_names_a_real_member(tmp_path: Path, personas: dict[str, Any]) -> None:
    h = Harness(tmp_path, personas, responder_with([decision([("rigor", "go")])] * 2, "Fine."))
    for _ in range(3):
        await h.engine.step()
    turn = next(
        r
        for r in h.requests()
        if r.tag.purpose == "turn" and "asks you: go" in r.messages[0].content
    )
    assert "Address members by name, e.g. @Compounder." in turn.messages[0].content
    assert "@nick" not in turn.messages[0].content
