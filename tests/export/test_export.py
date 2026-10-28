"""Session export (M13): Markdown and HTML reports built from saved files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from privy_council.export import ExportError, ExportFormat, export_session
from privy_council.export.report import build_report
from privy_council.models.flaw import Flaw, FlawKind
from privy_council.models.persona import Persona
from privy_council.models.session import (
    DossierAdded,
    DossierFile,
    Event,
    Message,
    MessageEvent,
    Phase,
    ResearchAdded,
    ResearchNote,
    RoundStarted,
    SessionMeta,
    SessionState,
    SessionStateChanged,
    VoteTaken,
)
from privy_council.models.vote import Vote
from privy_council.storage.sessions import SessionStore
from privy_council.util.language import Language
from tests.engine.test_protocol import personas as personas

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
SYNTHESIS = "## Consensus\n\n- Build the emergency fund first.\n"


def msg(speaker: str, text: str, rnd: int = 1, **flags: bool) -> MessageEvent:
    """A chat message event."""
    return MessageEvent(message=Message(speaker=speaker, text=text, round=rnd, ts=NOW, **flags))


def events() -> list[Event]:
    """A short closed debate touching every kind of event a report shows or hides."""
    note = ResearchNote(focus="PEA ceiling", text="- Ceiling: €150,000.", searches=["pea plafond"])
    return [
        msg("system", "Question set; opening the debate", 0),
        ResearchAdded(note=note),
        msg("chair", "**The question:** PEA or emergency fund?", 0),
        RoundStarted(round=1, label="Round 1 — blind takes"),
        msg("rigor", "Fund first. ## Not a heading\n\n## Mine\n\nKeep 6 months.", fallback=True),
        msg("compounder", "PEA first <script>alert(1)</script>", truncated=True),
        msg("skeptic", "I was kicked later."),
        msg("chair", "How stable is your job?", to_user=True),
        msg("user", "Very stable."),
        DossierAdded(file=DossierFile(name="budget.csv", text="secret rows", tokens=1200)),
        VoteTaken(
            round=1,
            votes=[
                Vote(speaker="rigor", position="Fund | then PEA", confidence=80),
                Vote(speaker="compounder", error="timed out"),
            ],
        ),
        SessionStateChanged(state=SessionState.SYNTHESIZING),
        msg("chair", SYNTHESIS),
        SessionStateChanged(state=SessionState.CLOSED),
    ]


def save_session(data_dir: Path, lang: Language = "en", synthesis: bool = True) -> Path:
    """Write a closed session through the storage layer; return its directory."""
    store = SessionStore.create(data_dir, "finance", "PEA or fund?", NOW)
    meta = SessionMeta(
        id=store.id,
        council="finance",
        topic="PEA or fund?",
        moderator="chair",
        members=["rigor", "compounder"],
        created_at=NOW,
        state=SessionState.CLOSED,
        lang=lang,
        models={"chair": ["claude_cli:sonnet"], "rigor": ["ollama:qwen3.6:27b", "ollama:super"]},
    )
    meta.runtime.phase = Phase.DONE
    meta.runtime.round = 1
    meta.runtime.flaws = [
        Flaw(member="rigor", kind=FlawKind.UNSOURCED, excerpt="6 months at 4%", note="4%", round=1),
        Flaw(member="rigor", kind=FlawKind.SELF_PREFIX, excerpt="Rigor:", round=1),  # formatting
        Flaw(
            member="compounder",
            kind=FlawKind.LOGIC,
            excerpt="PEA first",
            note="ignores the job risk",
            round=1,
            by="moderator",
        ),
    ]
    store.save_meta(meta)
    for event in events():
        store.append(event)
    if synthesis:
        store.save_synthesis(SYNTHESIS)
    return store.directory


def export(directory: Path, personas: dict[str, Persona], fmt: ExportFormat, **kw: object) -> str:
    """Export one format to the default place and return the text."""
    [path] = export_session(
        directory,
        personas,
        user_name="Alex",
        exports_dir=directory.parent.parent / "exports",
        fmt=fmt,
        **kw,  # type: ignore[arg-type]
    )
    assert path == directory.parent.parent / "exports" / f"{directory.name}.{fmt.value}"
    return path.read_text(encoding="utf-8")


def test_report_seats_flaws_and_events(tmp_path: Path, personas: dict[str, Persona]) -> None:
    report = build_report(save_session(tmp_path), personas, "Alex")
    assert [s.slug for s in report.seats] == ["chair", "rigor", "compounder", "skeptic"]
    rigor = report.seat("rigor")
    assert rigor is not None and rigor.recorded
    assert rigor.models == ["ollama:qwen3.6:27b", "ollama:super"]
    compounder = report.seat("compounder")
    assert compounder is not None and not compounder.recorded  # from today's persona file
    assert [f.kind for f in report.flaws] == [FlawKind.UNSOURCED, FlawKind.LOGIC]
    texts = [e.message.text for e in report.events if isinstance(e, MessageEvent)]
    assert SYNTHESIS not in texts  # shown once, at the top
    assert report.final_vote is not None and report.final_vote.round == 1


def test_markdown(tmp_path: Path, personas: dict[str, Persona]) -> None:
    text = export(save_session(tmp_path), personas, ExportFormat.MD)
    assert text.startswith("# PEA or fund?\n")
    order = ["## Members", "## Synthesis", "### Consensus", "## Final vote", "## Errors caught"]
    order += ["## Transcript", "### Opening", "### Round 1 — blind takes", "#### Vote after"]
    positions = [text.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert "`ollama:qwen3.6:27b → ollama:super`" in text
    assert "| Compounder | advisor |" in text and "\\* from today's persona file" in text
    assert "| Rigor | 80% | Fund \\| then PEA |" in text
    assert "| Compounder | — | no vote: timed out |" in text
    assert '- r1 [unsourced, engine] 4% in "6 months at 4%"' in text
    assert "self_prefix" not in text  # formatting flaws stay in /flaws
    assert "\\<Rigor\\>** *[local fallback]*" in text
    assert "*[cut off]*" in text and "*[question for you]*" in text
    assert "\\<Alex\\>** Very stable." in text
    assert "###### Mine" in text  # a member's heading nests under the round
    assert "*Attached budget.csv (~1,200 tokens)*" in text
    assert "secret rows" not in text  # dossier contents stay out
    assert "> *Claude web search · Searches: pea plafond*" in text
    assert text.count("Build the emergency fund first.") == 1


def test_html_is_self_contained_and_escaped(tmp_path: Path, personas: dict[str, Persona]) -> None:
    text = export(save_session(tmp_path), personas, ExportFormat.HTML)
    assert text.startswith("<!doctype html>")
    assert "<script" not in text  # neither ours nor the model's
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "http" not in text.split("<main>")[0]  # no CDN, no external stylesheet
    assert "<h3>Consensus</h3>" in text
    assert f'style="--c:{personas["rigor"].meta.color}"' in text
    assert "<summary>Research: PEA ceiling</summary>" in text
    assert "[local fallback]" in text and "<td>Fund | then PEA</td>" in text


def test_french_labels(tmp_path: Path, personas: dict[str, Persona]) -> None:
    text = export(save_session(tmp_path, lang="fr"), personas, ExportFormat.MD)
    assert "- **Langue :** français" in text  # French spacing before the colon
    assert "## Synthèse" in text and "## Transcription" in text and "### Ouverture" in text
    assert "| Compounder | conseiller |" in text


def test_without_flaws_and_unfinished(tmp_path: Path, personas: dict[str, Persona]) -> None:
    text = export(save_session(tmp_path, synthesis=False), personas, ExportFormat.MD, flaws=False)
    assert "## Errors caught" not in text
    assert "*No synthesis yet: the debate isn't finished.*" in text


def test_all_formats_and_out(tmp_path: Path, personas: dict[str, Persona]) -> None:
    directory = save_session(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    paths = export_session(
        directory, personas, user_name="A", exports_dir=tmp_path, fmt=ExportFormat.ALL, out=out
    )
    assert [p.name for p in paths] == [f"{directory.name}.md", f"{directory.name}.html"]
    file = tmp_path / "report.html"
    [path] = export_session(
        directory, personas, user_name="A", exports_dir=tmp_path, fmt=ExportFormat.HTML, out=file
    )
    assert path == file and file.read_text().startswith("<!doctype html>")
    with pytest.raises(ExportError, match="must be a directory"):
        export_session(
            directory, personas, user_name="A", exports_dir=tmp_path, fmt=ExportFormat.ALL, out=file
        )


def test_unreadable_session(tmp_path: Path, personas: dict[str, Persona]) -> None:
    (tmp_path / "broken").mkdir()
    with pytest.raises(ExportError, match="can't read session broken"):
        build_report(tmp_path / "broken", personas, "A")
