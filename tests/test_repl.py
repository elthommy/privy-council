from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from privy_council import cli
from privy_council.backends.fake import FakeBackend
from privy_council.backends.registry import BackendRegistry
from privy_council.config import Settings
from privy_council.engine.budget import Ledger
from privy_council.models.persona import BackendKind, Persona
from privy_council.models.session import SessionState
from privy_council.repl import run_repl
from tests.engine.test_memory import USER_LINE, memory_responder
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_topic import topicless
from tests.engine.test_user_questions import QUESTION, asking


def feeder(lines: list[str]) -> Iterator[str]:
    """Yield input lines, then raise EOFError like a closed stdin."""
    yield from lines
    raise EOFError


def run(h: Harness, lines: list[str]) -> list[str]:
    """Drive the REPL with scripted input and return printed lines."""
    out: list[str] = []
    feed = feeder(lines)
    run_repl(h.engine, read_line=lambda _: next(feed), out=lambda line, _c: out.append(line))
    return out


def test_repl_session(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [
        decision([("rigor", "go")]),
        decision(stop=True, reason="agreed"),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    out = run(
        h, ["/step", "plain words", "/state", "/budget", "/bogus", "/help", "/step", "/close"]
    )
    text = "\n".join(out)
    assert out[0].startswith("#finance — Should I max my PEA")
    assert "<Chair> Restating" in text  # the opening ran at start
    assert "-- Round 1 — blind takes --" in text
    assert "Rigor is thinking…" in text
    assert "<User> plain words" in text
    assert "Positions:" in text and "Rigor: fund first" in text
    assert "session: 0/40 Claude calls" in text
    assert "unknown command /bogus" in text
    assert "/step" in text and "/close" in text
    assert "* Debate stopped early: agreed" in text
    assert "Consensus: none · decision: open · stalled rounds: 0" in text
    assert "<Chair> ## Consensus" in text
    assert h.state() is SessionState.CLOSED


def test_repl_eof_quits_and_keeps_session(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    out = run(h, [])
    assert any("Saved session" in line for line in out)
    assert h.store.load_meta().state is SessionState.PAUSED


def test_repl_clips_long_turns(tmp_path: Path, personas: dict[str, Persona]) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "x")])]
    base = standard_responder(decisions)
    h = Harness(
        tmp_path, personas, lambda req: "z" * 5000 if req.tag.purpose != "moderate" else base(req)
    )
    out = run(h, ["/step"])
    chair = next(line for line in out if "<Chair>" in line)
    rigor = next(line for line in out if "<Rigor>" in line)
    assert not chair.endswith(" […]")  # the moderator's opening is shown whole
    assert rigor.endswith(" […]") and len(rigor) < 2100
    rigor_msg = next(m for m in h.engine.messages if m.speaker == "rigor")
    assert len(rigor_msg.text) == 5000  # the transcript keeps everything


def test_cli_new_runs_repl_with_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import REPO_ROOT

    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )
    fake = FakeBackend(responder=standard_responder([]))

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose backends are both the fake."""
        overrides = {BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)

    monkeypatch.setattr(cli, "BackendRegistry", registry)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(
        cli.app,
        ["--root", str(tmp_path), "new", "finance", "Rent or buy?", "--no-tui"],
        input="/close\n",
    )
    assert result.exit_code == 0, result.output
    assert "<Chair> ## Consensus" in result.output
    base = tmp_path / "data" / "sessions"
    sessions = [d for d in base.iterdir() if not d.is_symlink()]
    assert len(sessions) == 1 and sessions[0].name.endswith("_finance_rent-or-buy")
    assert (base / "latest").resolve() == sessions[0].resolve()
    assert f"Live log: {sessions[0] / 'transcript.md'}" in result.output
    assert (sessions[0] / "synthesis.md").is_file()
    assert (tmp_path / "data" / "logs" / "council.log").exists()


def test_cli_new_unknown_council() -> None:
    from tests.conftest import REPO_ROOT

    result = CliRunner().invoke(cli.app, ["--root", str(REPO_ROOT), "new", "nope", "x"])
    assert result.exit_code == 1
    assert "unknown or invalid council" in result.output


def _cli_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Write a config pointing at the repo personas and install a fake registry."""
    from tests.conftest import REPO_ROOT
    from tests.engine.test_research import responder

    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )
    fake = FakeBackend(responder=responder([]))

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose backends are both the fake."""
        overrides = {BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)

    monkeypatch.setattr(cli, "BackendRegistry", registry)
    return fake


@pytest.mark.parametrize(
    ("flag", "line", "searched"),
    [("--research", "/quit\n", True), ("--no-research", "/research gold\n/quit\n", False)],
)
def test_cli_research_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, line: str, searched: bool
) -> None:
    fake = _cli_setup(tmp_path, monkeypatch)
    # career: research is optional there (finance requires it)
    args = ["--root", str(tmp_path), "new", "career", "Gold?", "--no-tui", flag]
    result = CliRunner().invoke(cli.app, args, input=line)
    assert result.exit_code == 0, result.output
    assert any(r.tag.purpose == "research" for r in fake.requests) is searched
    if searched:
        assert "-- research (Claude web search): background for the question" in result.output
    else:
        assert "Research is off for this session" in result.output


def test_repl_stays_open_for_keep_after_close(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    out = run(h, [USER_LINE, "/close", "/keep 9", "/keep all", "/never-read"])
    text = "\n".join(out)
    assert "-- shared memory: proposed facts (from your own words) --" in text
    assert "usage: /keep 1 3 | all | none" in text
    assert "Saved 2 fact(s) to shared memory" in text
    assert "/never-read" not in text and "unknown command" not in text  # exited after /keep
    assert not any("Saved session" in line for line in out)


def test_repl_quit_with_pending_facts_warns(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, memory_responder())
    out = run(h, [USER_LINE, "/close", "/forget all", "/forget all", "/quit"])
    text = "\n".join(out)
    assert "Forgot all memory" in text
    assert "Unreviewed shared facts were not saved." in text


def test_repl_asks_for_the_question_first(tmp_path: Path, personas: dict[str, Persona]) -> None:
    engine, _, _ = topicless(tmp_path, personas)
    out: list[str] = []
    prompts_seen: list[str] = []
    feed = feeder(["", "Rent or buy?", "/quit"])

    def read(prompt: str) -> str:
        """Record the prompt shown, then feed the next line."""
        prompts_seen.append(prompt)
        return next(feed)

    run_repl(engine, read_line=read, out=lambda line, _c: out.append(line))
    assert prompts_seen[:2] == ["Question: ", "Question: "]
    assert any(line.startswith("#finance — Rent or buy?") for line in out)
    assert any("<Chair> Restating" in line for line in out)


def test_repl_eof_before_the_question(tmp_path: Path, personas: dict[str, Persona]) -> None:
    engine, _, fake = topicless(tmp_path, personas)
    out: list[str] = []
    feed = feeder([])
    run_repl(engine, read_line=lambda _: next(feed), out=lambda line, _c: out.append(line))
    assert out == ["No question given; nothing to debate."]
    assert fake.requests == []


def test_repl_highlights_chair_questions(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([asking([("rigor", "go")], QUESTION)]))
    text = "\n".join(run(h, ["/run 3"]))
    assert f"▶ <Chair> question for you:\n@User {QUESTION}" in text
    assert "/run paused for Chair's question" in text
    assert "↳ answer in plain text, then /step (or /run N) to continue" in text


def test_cli_new_without_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import REPO_ROOT

    (tmp_path / "config.toml").write_text(
        f"[paths]\npersonas = '{REPO_ROOT / 'personas'}'\n"
        f"councils = '{REPO_ROOT / 'councils'}'\ndata = '{tmp_path / 'data'}'\n"
    )
    fake = FakeBackend(responder=standard_responder([]))

    def registry(settings: Settings, ledger: Ledger, local_only: bool = False) -> BackendRegistry:
        """A registry whose backends are both the fake."""
        overrides = {BackendKind.CLAUDE_CLI: fake, BackendKind.OLLAMA: fake}
        return BackendRegistry(settings, ledger, overrides=overrides, local_only=local_only)

    monkeypatch.setattr(cli, "BackendRegistry", registry)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base = tmp_path / "data" / "sessions"
    args = ["--root", str(tmp_path), "new", "finance", "--no-tui"]

    started = CliRunner().invoke(cli.app, args, input="Rent or buy?\n/quit\n")
    assert started.exit_code == 0, started.output
    sessions = [d for d in base.iterdir() if not d.is_symlink()]
    assert [d.name.endswith("_finance_rent-or-buy") for d in sessions] == [True]

    abandoned = CliRunner().invoke(cli.app, args, input="")
    assert abandoned.exit_code == 0, abandoned.output
    assert len([d for d in base.iterdir() if not d.is_symlink()]) == 1  # no empty session left
    assert not (base / "latest").exists() and not (base / "latest").is_symlink()


def test_repl_summon_and_kick(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision(stop=True, reason="agreed")]))
    out = run(h, ["/summon optimist", "/kick @Rigor", "/help"])
    assert "* Optimist joined the channel and speaks next round" in out
    assert "* Rigor left the channel" in out
    assert any(line.startswith("/summon ") for line in out)
    assert list(h.engine.members) == ["compounder", "pragmatist", "contrarian", "optimist"]


def test_repl_vote_and_summary(tmp_path: Path, personas: dict[str, Persona]) -> None:
    from tests.engine.test_review import SUMMARY, reviewing

    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])]))
    text = "\n".join(run(h, ["/step", "/vote", "/summary"]))
    assert "-- vote after round 1 --" in text
    assert "  Rigor        70%  rigor: fund first" in text
    assert f"<Chair> {SUMMARY}" in text


def test_cli_attach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _cli_setup(tmp_path, monkeypatch)
    bad = CliRunner().invoke(
        cli.app,
        ["--root", str(tmp_path), "new", "finance", "Gold?", "--no-tui", "--attach", "x.pdf"],
    )
    assert bad.exit_code == 1 and "No file at x.pdf" in bad.output
    assert not (tmp_path / "data" / "sessions").exists()  # failed before creating a session

    (tmp_path / "goals.md").write_text("Retire at 55.")
    args = ["--root", str(tmp_path), "new", "finance", "Gold?", "--no-tui"]
    result = CliRunner().invoke(
        cli.app, [*args, "--attach", str(tmp_path / "goals.md")], input="/quit\n"
    )
    assert result.exit_code == 0, result.output
    assert "-- dossier: goals.md (~" in result.output
    research, opening = fake.requests[:2]  # finance searches first
    assert research.tag.purpose == "research" and opening.tag.purpose == "opening"
    assert "Retire at 55." not in research.messages[0].content  # files never reach searches
    assert "File goals.md:\nRetire at 55." in opening.messages[0].content
    [session] = [d for d in (tmp_path / "data" / "sessions").iterdir() if not d.is_symlink()]
    assert (session / "dossier" / "goals.md").is_file()


def test_repl_attach(tmp_path: Path, personas: dict[str, Persona]) -> None:
    (tmp_path / "c.txt").write_text("no crypto")
    h = Harness(tmp_path, personas, standard_responder([]))
    text = "\n".join(run(h, [f"/attach {tmp_path / 'c.txt'}", "/attach"]))
    assert "-- dossier: c.txt (~3 tokens, verbatim) --" in text
    assert "* usage: /attach <path>" in text


def test_repl_as_speaks_for_a_guest(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    out = run(h, ["/as @Ana I own in Paris", "/as Ana", "/quit"])
    text = "\n".join(out)
    assert "<Ana> I own in Paris" in text
    assert "usage: /as <name> <text>" in text
    assert h.engine.guests == ["Ana"]
