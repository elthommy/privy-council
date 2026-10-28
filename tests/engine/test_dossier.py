"""The dossier: `/attach`, verbatim small files, summarized large ones."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from privy_council.backends.base import BackendOutputError, CompletionRequest
from privy_council.backends.fake import Reply
from privy_council.engine import context, prompts
from privy_council.engine.dossier import DossierError, attach_path, csv_head, read_attachable
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import DossierAdded, ErrorEvent, TextDelta
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed
from tests.engine.test_roster import notices
from tests.engine.test_topic import topicless

SUMMARY = "Monthly spend is 2,400 EUR; rent 1,100 EUR."
# A small threshold keeps the test files small.
SMALL = {"dossier": {"summarize_over_tokens": 50}}


def summarizing(decisions: list[dict[str, Any] | str]) -> Any:
    """Answer dossier summaries with SUMMARY; everything else as usual."""
    standard = standard_responder(decisions)

    def respond(req: CompletionRequest) -> Reply:
        """Return the scripted reply for this request."""
        return SUMMARY if req.tag.purpose == "dossier" else standard(req)

    return respond


def write(tmp_path: Path, name: str, text: str) -> Path:
    """Write a file to attach, outside the session directory."""
    path = tmp_path / "files" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def added(h: Harness) -> list[DossierAdded]:
    """Return the DossierAdded events so far."""
    return [e for e in h.events if isinstance(e, DossierAdded)]


def test_attach_path_strips_quotes_and_expands_home() -> None:
    assert attach_path(" 'my file.csv' ") == Path("my file.csv")
    assert attach_path('"a.md"') == Path("a.md")
    assert attach_path("~/a.txt") == Path.home() / "a.txt"
    assert attach_path("  ") is None


def test_read_attachable_refusals(tmp_path: Path) -> None:
    with pytest.raises(DossierError, match="No file at"):
        read_attachable(tmp_path / "nope.txt", 100)
    with pytest.raises(DossierError, match=r"only \.txt, \.json, \.md, \.csv files"):
        read_attachable(write(tmp_path, "offer.pdf", "x"), 100)
    with pytest.raises(DossierError, match="the limit is 100"):
        read_attachable(write(tmp_path, "big.txt", "x" * 101), 100)
    with pytest.raises(DossierError, match="it is empty"):
        read_attachable(write(tmp_path, "empty.md", "  \n"), 100)
    binary = tmp_path / "files" / "bin.csv"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DossierError, match="isn't UTF-8 text"):
        read_attachable(binary, 100)
    assert read_attachable(write(tmp_path, "UP.JSON", '{"a": 1}'), 100) == '{"a": 1}'


def test_csv_head_keeps_the_header_and_first_rows() -> None:
    assert csv_head("a,b\n1,2\n3,4\n5,6\n", 2) == "a,b\n1,2\n3,4"
    assert csv_head("a,b\n1,2\n", 0) is None


async def test_small_file_goes_verbatim_into_every_prompt(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    source = write(tmp_path, "constraints.md", "- no crypto\n- horizon 20 years\n")
    await h.engine.attach(str(source))
    [event] = added(h)
    assert event.file.name == "constraints.md" and not event.file.summarized
    assert event.file.text == "- no crypto\n- horizon 20 years"
    assert notices(h)[-1].startswith("Attached constraints.md (~")
    copy = h.store.directory / "dossier" / "constraints.md"
    assert copy.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert not h.requests()  # no model call for a small file

    await h.engine.step()  # opening
    await h.engine.step()  # blind round + moderator update
    for req in h.requests():
        assert prompts.TITLE_DOSSIER in req.messages[0].content, req.tag.purpose
        assert "File constraints.md:\n- no crypto" in req.messages[0].content
    md = h.store.transcript_md.read_text(encoding="utf-8")
    assert "*Attached constraints.md (~" in md and "included verbatim)*" in md


async def test_large_csv_is_summarized_once_and_keeps_its_first_rows(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(
        tmp_path,
        personas,
        summarizing([decision([("rigor", "go")])]),
        config={"dossier": {"summarize_over_tokens": 50, "csv_head_rows": 2}},
    )
    rows = "\n".join(f"2026-0{i % 9 + 1}-01,item {i},{i * 10}" for i in range(60))
    await h.engine.attach(str(write(tmp_path, "budget.csv", f"date,label,eur\n{rows}\n")))
    [call] = h.requests()
    assert call.tag.purpose == "dossier" and call.tag.persona == "chair"
    assert call.system.startswith(prompts.DOSSIER_SUMMARY_SYSTEM)
    assert "item 59" in call.messages[0].content  # the whole file was read
    assert h.engine.meta.topic in call.messages[0].content
    assert not any(isinstance(e, TextDelta) for e in h.events)  # not streamed

    [event] = added(h)
    assert event.file.summarized
    assert event.file.text.startswith(SUMMARY)
    assert event.file.text.endswith("date,label,eur\n2026-01-01,item 0,0\n2026-02-01,item 1,10")
    summary_md = h.store.directory / "dossier" / "budget.csv.summary.md"
    assert summary_md.read_text(encoding="utf-8") == SUMMARY + "\n"
    assert "everyone sees Chair's summary" in notices(h)[-1]

    await h.engine.step()
    opening = next(r for r in h.requests() if r.tag.purpose == "opening")
    assert f"File budget.csv (too long to include; the moderator's summary):\n{SUMMARY}" in (
        opening.messages[0].content
    )
    assert "item 59" not in opening.messages[0].content


async def test_failed_summary_attaches_nothing(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    standard = standard_responder([])

    def respond(req: CompletionRequest) -> Reply:
        """Fail the summary."""
        return BackendOutputError("boom") if req.tag.purpose == "dossier" else standard(req)

    h = Harness(tmp_path, personas, respond, config=SMALL)
    await h.engine.attach(str(write(tmp_path, "notes.txt", "word " * 200)))
    assert isinstance(h.events[-1], ErrorEvent)
    assert (
        h.events[-1].text
        == "Couldn't summarize notes.txt (boom); nothing attached. /attach to retry"
    )
    assert not (h.store.directory / "dossier" / "notes.txt").exists()
    assert h.engine.meta.runtime.dossier == [] and not h.engine.busy


async def test_too_long_to_summarize_whole_is_clipped(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    # On Ollama at the default 8k window, the Chair reads ~16k characters at once.
    chair = personas["chair"].model_copy(
        update={
            "meta": personas["chair"].meta.model_copy(
                update={"backend": BackendKind.OLLAMA, "model": "qwen3.6:27b"}
            )
        }
    )
    h = Harness(tmp_path, {**personas, "chair": chair}, summarizing([]), config=SMALL)
    await h.engine.attach(str(write(tmp_path, "long.txt", "start " + "x" * 40_000 + " FINAL")))
    [call] = h.requests()
    assert "start" in call.messages[0].content and "FINAL" not in call.messages[0].content
    assert "cut at the end" in call.messages[0].content
    assert any("only its first" in n and "% is summarized" in n for n in notices(h))


async def test_same_name_twice_and_reserved_research_name(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    await h.engine.attach(str(write(tmp_path, "a.txt", "one")))
    await h.engine.attach(str(write(tmp_path, "a.txt", "two")))
    await h.engine.attach(str(write(tmp_path, "research.md", "mine")))
    assert [e.file.name for e in added(h)] == ["a.txt", "a-2.txt", "research-2.md"]
    dossier_text = context.dossier_text(h.engine.meta.runtime.dossier) or ""
    assert dossier_text.index("File a.txt:\none") < dossier_text.index("File a-2.txt:\ntwo")


async def test_attach_refusals(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision(stop=True, reason="done")]))
    await h.engine.attach("")
    await h.engine.attach(str(tmp_path / "missing.csv"))
    assert notices(h) == [
        "usage: /attach <path> (.txt, .json, .md, .csv files)",
        f"No file at {tmp_path / 'missing.csv'}",
    ]
    await h.engine.step()
    await h.engine.step()
    await h.engine.close()
    await h.engine.attach(str(write(tmp_path, "late.txt", "too late")))
    assert notices(h)[-1] == "The session is closed"


async def test_attach_before_the_question(tmp_path: Path, personas: dict[str, Persona]) -> None:
    engine, store, _fake = topicless(tmp_path, personas)
    await engine.attach(str(write(tmp_path, "offer.md", "Salary 60k")))
    assert [f.name for f in engine.meta.runtime.dossier] == ["offer.md"]
    assert (store.directory / "dossier" / "offer.md").is_file()


async def test_dossier_survives_a_resume(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.attach(str(write(tmp_path, "c.md", "no crypto")))
    await h.engine.step()
    r = Resumed(h, h.store.directory, personas, standard_responder([decision([("rigor", "go")])]))
    assert [e for e in r.engine.history if isinstance(e, DossierAdded)] == added(h)
    await r.engine.step()
    assert all("File c.md:\nno crypto" in q.messages[0].content for q in r.ollama.requests)


def test_dossier_is_capped_at_its_share() -> None:
    prompt = context.build_prompt(
        system="s",
        topic="t",
        task="do it",
        budget_tokens=1000,
        dossier="d" * 10_000,
        dossier_share=0.4,
    )
    assert prompt.user.count("d") < 1700  # 0.4 x ~4000 chars
    assert prompts.TRIMMED_MARK in prompt.user


async def test_trimming_is_logged_at_debug(
    tmp_path: Path, personas: dict[str, Persona], caplog: pytest.LogCaptureFixture
) -> None:
    config = {"ollama": {"models": {"nemotron-3.5-super:latest": {"num_ctx": 2000}}}}
    h = Harness(
        tmp_path, personas, standard_responder([decision([("rigor", "go")])]), config=config
    )
    await h.engine.step()
    for i in range(6):
        h.engine.say(f"long interjection {i} " + "y" * 600)
    with caplog.at_level(logging.DEBUG, logger="privy_council.engine.core"):
        await h.engine.step()
    assert any(
        "trimmed rigor/turn (nemotron-3.5-super:latest, round 1, ollama): " in r.getMessage()
        for r in caplog.records
    )


async def test_web_research_never_sees_the_dossier(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    from tests.engine.test_research import responder

    h = Harness(tmp_path, personas, responder([]))
    await h.engine.attach(str(write(tmp_path, "payslip.txt", "Salary 4,200 EUR")))
    await h.engine.step()
    await h.engine.research("rates")
    [search] = [r for r in h.requests() if r.tag.purpose == "research"]
    assert "Salary" not in search.messages[0].content
    assert prompts.TITLE_DOSSIER not in search.messages[0].content


BUDGET = """\
id,date,category,label,amount
1,2026-01-03,rent,flat,1100.00
2,2026-01-09,food,market,84.50
3,2026-02-03,rent,flat,1100.00
4,2026-02-11,food,market,"1,015.25"
"""


def test_csv_stats_totals_per_month_and_per_category() -> None:
    from privy_council.engine.dossier import csv_stats

    stats = csv_stats(BUDGET) or ""
    assert stats.splitlines() == [
        prompts.CSV_STATS_TITLE,
        "- 4 rows; columns: id, date, category, label, amount",
        "- amount: 4 values, sum 3,299.75, min 84.50, max 1,100.00",
        "- date: 2026-01-03 to 2026-02-11",
        "- sum of amount per month (date): 2026-01 1,184.50; 2026-02 2,115.25",
        "- sum of amount per category: food 1,099.75 (rows: 2); rent 2,200.00 (rows: 2)",
        "- sum of amount per label: flat 2,200.00 (rows: 2); market 1,099.75 (rows: 2)",
    ]  # the id column is never summed


def test_csv_stats_semicolon_files_use_comma_decimals() -> None:
    from privy_council.engine.dossier import csv_stats

    stats = csv_stats("Datum;Betrag\n2026-01-03;1.100,00\n2026-01-09;84,50\n") or ""
    assert "- Betrag: 2 values, sum 1,184.50, min 84.50, max 1,100.00" in stats


def test_csv_stats_never_guesses_ambiguous_or_mixed_columns() -> None:
    from privy_council.engine.dossier import csv_stats

    stats = csv_stats("k,v,w\na,1.5,7\nb,oops,\nc,2,3\n") or ""
    assert "- v:" not in stats  # one cell isn't a number: the column is text
    assert "- w: 2 values, sum 10, min 3, max 7" in stats  # blanks are skipped
    assert csv_stats("only,a,header\n") is None


def test_csv_stats_are_capped() -> None:
    from privy_council.engine.dossier import STATS_MAX_CHARS, csv_stats

    rows = "\n".join(f"2026-{m:02d}-01,c{m % 15},{m}" for m in range(1, 13))
    many = "date,cat,a\n" + "\n".join(rows for _ in range(30))
    wide = (
        "date,"
        + ",".join(f"col{i}" for i in range(40))
        + "\n"
        + "\n".join(f"2026-{m:02d}-01," + ",".join(str(i) for i in range(40)) for m in range(1, 13))
    )
    for text in (many, wide):
        stats = csv_stats(text) or ""
        assert len(stats) <= STATS_MAX_CHARS
    assert (csv_stats(wide) or "").endswith(prompts.CSV_STATS_CUT)


async def test_small_csv_gets_its_totals_in_prompts(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    await h.engine.attach(str(write(tmp_path, "budget.csv", BUDGET)))
    [event] = added(h)
    assert event.file.stats is not None and not event.file.summarized
    await h.engine.step()
    opening = h.requests()[0].messages[0].content
    assert "File budget.csv:\nid,date" in opening
    assert "2026-01 1,184.50; 2026-02 2,115.25" in opening
    assert "per month" in h.store.transcript_md.read_text(encoding="utf-8")


async def test_large_csv_summary_is_given_the_exact_figures(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, summarizing([]), config=SMALL)
    rows = "\n".join(f"{i},2026-01-{i % 28 + 1:02d},rent,flat,10" for i in range(5, 60))
    await h.engine.attach(str(write(tmp_path, "big.csv", BUDGET + rows + "\n")))
    [call] = h.requests()
    assert "quote them rather than recomputing" in call.messages[0].content
    assert "- amount: 59 values, sum 3,849.75" in call.messages[0].content
    [event] = added(h)
    assert event.file.summarized and event.file.stats and "sum 3,849.75" in event.file.stats
    text = context.dossier_text(h.engine.meta.runtime.dossier) or ""
    assert text.index(SUMMARY) < text.index(prompts.CSV_STATS_TITLE)
