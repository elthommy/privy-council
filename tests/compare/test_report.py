"""Comparison metrics, estimate and report (built from saved files, no model calls)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from privy_council.compare import estimate
from privy_council.compare.metrics import run_seconds, seat_times
from privy_council.compare.render import (
    build_compare_report,
    duration,
    render_html,
    render_markdown,
    write_reports,
)
from privy_council.compare.spec import apply_variant, parse_variant
from privy_council.engine.budget import Ledger, UsageEntry
from privy_council.engine.research import ResearchMode
from privy_council.export.report import Report, Seat
from privy_council.models.council import CouncilPreset
from privy_council.models.persona import Persona
from privy_council.models.session import (
    Event,
    Message,
    MessageEvent,
    SessionMeta,
    TurnStarted,
    VoteTaken,
)
from privy_council.models.vote import Vote
from tests.compare.test_runner import Rig
from tests.engine.test_protocol import personas as personas

T0 = datetime(2026, 10, 23, 20, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    """Return T0 plus `seconds`."""
    return T0 + timedelta(seconds=seconds)


def message(speaker: str, seconds: float, to_user: bool = False) -> MessageEvent:
    """A chat message event at T0 + seconds."""
    msg = Message(speaker=speaker, text="x", round=1, to_user=to_user, ts=at(seconds))
    return MessageEvent(message=msg, ts=at(seconds))


def report_for(slugs: list[str]) -> Report:
    """A bare report seating `slugs` under the moderator chair."""
    meta = SessionMeta(id="s", council="c", topic="t", moderator="chair", members=slugs)
    seats = [
        Seat(slug=s, nick=s.capitalize(), role="", color="#fff", models=[], recorded=False)
        for s in ["chair", *slugs]
    ]
    return Report(
        meta=meta, user_name="U", seats=seats, synthesis=None, events=[], flaws=[], final_vote=None
    )


def test_seat_times_pair_starts_with_messages() -> None:
    events: list[Event] = [
        TurnStarted(speaker="a", round=1, ts=at(0)),
        TurnStarted(speaker="b", round=1, ts=at(0)),  # blind round: both start at once
        message("a", 10),
        message("b", 30),
        TurnStarted(speaker="chair", round=1, ts=at(31)),
        message("chair", 40, to_user=True),
        TurnStarted(speaker="a", round=2, ts=at(50)),
        message("a", 70),
        TurnStarted(speaker="b", round=2, ts=at(80)),  # a vote turn: no message follows
        VoteTaken(round=2, votes=[Vote(speaker="b", position="p", confidence=50)], ts=at(85)),
        message("b", 300),  # unrelated later message, not timed
    ]
    times = {t.seat.slug: t for t in seat_times(events, report_for(["a", "b"]))}
    assert set(times) == {"a", "b"}  # the moderator isn't listed
    assert (times["a"].turns, times["a"].median_s, times["a"].max_s) == (2, 15, 20)
    assert (times["b"].turns, times["b"].max_s) == (1, 30)


def test_run_seconds_leave_out_idle_gaps() -> None:
    events: list[Event] = [message("a", 0), message("a", 60), message("a", 60 + 3 * 3600)]
    events.append(message("a", 60 + 3 * 3600 + 30))
    assert run_seconds(events) == 90


def test_duration_format() -> None:
    assert duration(45) == "45 s"
    assert duration(200) == "3 min 20 s"
    assert duration(3900) == "1 h 05 min"


def test_estimate_counts_claude_calls_and_time(personas: dict[str, Persona]) -> None:
    preset = CouncilPreset(
        name="finance", moderator="chair", members=["rigor", "compounder", "pragmatist"]
    )
    council, seated = apply_variant(preset, personas, parse_variant("v"), 3)
    claude_members = sum(seated[s].meta.backend.value == "claude_cli" for s in council.members)
    est = estimate.estimate_variant(
        council,
        seated,
        rounds=3,
        research=ResearchMode.AUTO,
        history={estimate.seat(seated["rigor"]): 300.0},
        moderator_s=40.0,
    )
    # chair: research, opening, 3 updates, synthesis; Claude members: 3 turns + vote
    assert est.claude_calls == 1 + 5 + claude_members * 4
    assert est.seconds > 300 * 4  # rigor's history counts for its 4 turns
    quiet = estimate.estimate_variant(
        council, seated, rounds=3, research=ResearchMode.OFF, history={}, moderator_s=40.0
    )
    assert quiet.claude_calls == est.claude_calls - 1
    assert estimate.total_calls([est, quiet], judge=True) == est.claude_calls * 2


async def test_report_compares_the_variants(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rig = Rig(tmp_path, personas, ["base", "qwen:rigor=qwen3.6:27b"], rounds=1)
    await rig.runner().run()
    first = rig.session(0).id
    ledger = Ledger.in_data_dir(rig.settings.data_dir)
    ledger.record(UsageEntry(session=first, persona="chair", purpose="opening", model="sonnet"))
    # The comparison often shares the first session's name; its judge call isn't that session's.
    assert rig.store.id == first
    judge_tag = f"compare:{rig.store.id}"
    ledger.record(UsageEntry(session=judge_tag, purpose="compare_judge", model="sonnet"))
    rig.store.save_judge("## Verdict\nA <b>wins</b>.")

    report = build_compare_report(
        rig.store,
        rig.settings.data_dir,
        personas,
        "User",
        Ledger.in_data_dir(rig.settings.data_dir),
    )

    base, qwen = report.variants
    assert (base.claude_calls, qwen.claude_calls) == (1, 0)
    assert base.rounds == 1 and base.confidence() == (70, 70, 70)
    assert [n for n, _ in base.flaw_counts()] == ["Rigor", "Compounder", "Pragmatist", "Contrarian"]
    md = render_markdown(report)
    assert "# Comparison: Should I max my PEA" in md
    assert "| base | as is | done | 1 | 1 |" in md
    assert "| qwen | rigor=qwen3.6:27b | done | 1 | 0 |" in md
    assert "| 70% |" in md
    assert "A = base, B = qwen" in md
    assert "\n### Verdict\n" in md  # the judge's headings nest under "## Judge"
    assert "`ollama:qwen3.6:27b`" not in md and "ollama:qwen3.6:27b" in md
    assert "\n##### Consensus\n" in md  # a synthesis's "#" title lands under "### Synthesis"
    html = render_html(report)
    assert "A &lt;b&gt;wins&lt;/b&gt;" in html  # model text never becomes markup
    assert "<script" not in html
    paths = write_reports(report, rig.store.directory)
    assert [p.name for p in paths] == ["report.md", "report.html"]


async def test_report_of_an_unstarted_variant(tmp_path: Path, personas: dict[str, Persona]) -> None:
    rig = Rig(tmp_path, personas, ["base", "other:-contrarian"])
    report = build_compare_report(rig.store, rig.settings.data_dir, personas, "User", None)
    md = render_markdown(report)
    assert "| base | as is | pending | 0 | 0 | — | — | — | — |" in md
    assert "*Not run yet (pending).*" in md
    assert "Not run yet (pending)." in render_html(report)
