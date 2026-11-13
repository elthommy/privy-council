"""scripts/flaw_bench.py: a stopped run resumes, running only the missing answers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import REPO_ROOT


def bench() -> Any:
    """Import the script (scripts/ is not a package; it imports replay_turn)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "flaw_bench", REPO_ROOT / "scripts/flaw_bench.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["flaw_bench"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))


def turn_record(spec: str) -> dict[str, Any]:
    """Return a saved turn as a run file stores it (made-up content)."""
    return {
        "type": "turn",
        "session": "data/sessions/s1",
        "spec": spec,
        "system": "You are the Skeptic.",
        "user": "Is a 5 % raise worth a move?",
        "temperature": 0.7,
        "max_output": 400,
        "nick": "Skeptic",
        "known": ["Skeptic", "Optimist"],
        "devil_line": None,
        "original": "It depends on the rent.",
    }


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    """Load the script with a fake model call; return it and the list of calls made."""
    module = bench()
    calls: list[str] = []

    def fake_run(model: str, think: Any, request: Any, num_ctx: int) -> dict[str, Any]:
        """Answer without a model; record which call was made."""
        calls.append(model)
        return {"message": {"content": f"answer from {model}"}, "wall_s": 0.1}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "load_settings", lambda root: None)
    monkeypatch.setattr(module, "model_think", lambda settings, raw: (raw, False))
    prompts = tmp_path / "prompts.jsonl"
    lines = [turn_record("skeptic:1"), turn_record("skeptic:2")]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    return module, calls


def args(tmp_path: Path, fresh: bool = False) -> argparse.Namespace:
    """Return the bench's arguments for two models on the saved prompts."""
    return argparse.Namespace(
        root=None,
        prompts_from=tmp_path / "prompts.jsonl",
        manifest=None,
        out=tmp_path / "out.jsonl",
        fresh=fresh,
        original=False,
        models=["m1", "m2"],
        runs=1,
        num_ctx=4096,
        verbose=False,
    )


def records(path: Path, kind: str) -> list[dict[str, Any]]:
    """Return a run file's records of one type."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["type"] == kind]


def test_resume_runs_only_missing_answers(
    setup: tuple[Any, list[str]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module, calls = setup
    module._bench(args(tmp_path))
    out = tmp_path / "out.jsonl"
    # Simulate a Ctrl-C after m1's answers: drop m2's.
    kept = [r for r in map(json.loads, out.read_text().splitlines()) if r.get("model") != "m2"]
    out.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    calls.clear()

    module._bench(args(tmp_path))

    assert calls == ["m2", "m2"]
    assert len(records(out, "turn")) == 2  # prompts not saved twice
    assert sorted((r["model"], r["spec"]) for r in records(out, "answer")) == [
        ("m1", "skeptic:1"),
        ("m1", "skeptic:2"),
        ("m2", "skeptic:1"),
        ("m2", "skeptic:2"),
    ]
    printed = capsys.readouterr().out
    assert "2 answer(s) already done" in printed
    table = printed.split("answers saved")[0].splitlines()
    assert [row.split()[:2] for row in table if row[:2] in ("m1", "m2")] == [
        ["m1", "2"],
        ["m2", "2"],
    ]


def test_complete_run_makes_no_call(setup: tuple[Any, list[str]], tmp_path: Path) -> None:
    module, calls = setup
    module._bench(args(tmp_path))
    calls.clear()
    module._bench(args(tmp_path))
    assert calls == []
    assert len(records(tmp_path / "out.jsonl", "answer")) == 4


def test_fresh_starts_over(setup: tuple[Any, list[str]], tmp_path: Path) -> None:
    module, calls = setup
    module._bench(args(tmp_path))
    calls.clear()
    module._bench(args(tmp_path, fresh=True))
    assert calls == ["m1", "m1", "m2", "m2"]
    assert len(records(tmp_path / "out.jsonl", "answer")) == 4
