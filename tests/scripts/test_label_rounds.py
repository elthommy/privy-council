"""scripts/label_rounds.py: a round is rebuilt with only what the members could see."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from tests.conftest import REPO_ROOT


def labeller() -> Any:
    """Import the script (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "label_rounds", REPO_ROOT / "scripts/label_rounds.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["label_rounds"] = module  # @dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


def msg(speaker: str, text: str, round_: int, to_user: bool = False) -> dict[str, Any]:
    """Return a message event as the transcript stores it."""
    m = {"speaker": speaker, "text": text, "round": round_, "to_user": to_user}
    return {"kind": "message", "message": m}


def flaws(round_: int, *kinds: str) -> dict[str, Any]:
    """Return a flaws event with one flaw per kind."""
    items = [
        {"member": "rigor", "kind": k, "excerpt": "x", "note": "", "round": round_, "by": "engine"}
        for k in kinds
    ]
    return {"kind": "flaws", "round": round_, "flaws": items}


def session(tmp_path: Path) -> Path:
    """Write a made-up three-round session and return its folder."""
    events = [
        {"kind": "research", "note": {"text": "fact A"}},
        msg("user", "I have 3 cats", 0),
        {"kind": "round_started", "round": 1},
        msg("rigor", "first take", 1),
        msg("chair", "@Rigor why?", 1, to_user=False),
        {"kind": "round_started", "round": 2},
        msg("rigor", "costs 12 euros", 2),
        msg("chair", "Thomas, how many?", 2, to_user=True),
        msg("user", "4 cats now", 2),
        flaws(2, "unsourced", "self_mention", "logic"),
        {"kind": "round_started", "round": 3},
        {"kind": "research", "note": {"text": "fact B, later"}},
        msg("rigor", "third", 3),
        msg("user", "too late", 3),
    ]
    folder = tmp_path / "2026-11-01_x_y"
    folder.mkdir()
    (folder / "session.json").write_text(json.dumps({"topic": "Cats?"}))
    (folder / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return folder


def test_round_keeps_only_what_members_saw(tmp_path: Path) -> None:
    lr = labeller()
    r = lr.load_round(session(tmp_path), 2)
    assert r.research == ["fact A"]
    assert r.user_lines == ["I have 3 cats", "4 cats now"]
    assert r.earlier == [(1, "rigor", "first take")]
    assert r.turns == [("rigor", "costs 12 euros")]
    assert [f["kind"] for f in r.flags] == ["unsourced", "logic"]
    prompt = lr.build_prompt(r)
    assert "fact B" not in prompt and "too late" not in prompt
    assert '[flag 1] by engine, on rigor, kind logic: "x"' in prompt


def test_missed_quotes_are_checked_against_their_turn(tmp_path: Path) -> None:
    lr = labeller()
    r = lr.load_round(session(tmp_path), 2)
    missed = [
        {"turn": 0, "kind": "unsourced", "quote": "costs  12\neuros", "note": ""},
        {"turn": 0, "kind": "logic", "quote": "costs 13 euros", "note": ""},
        {"turn": 5, "kind": "logic", "quote": "x", "note": ""},
    ]
    lr.check_quotes(r, missed)
    assert [m["quote_found"] for m in missed] == [True, False, False]
    assert [m["speaker"] for m in missed] == ["rigor", "rigor", "?"]


def test_check_sheet_has_one_answer_line_per_label(tmp_path: Path) -> None:
    lr = labeller()
    folder = session(tmp_path)
    flag = {"member": "rigor", "kind": "unsourced", "excerpt": "12 euros", "by": "engine"}
    row = {
        "key": f"{folder.name}:2",
        "flags_in": [flag],
        "flags": [{"id": 0, "verdict": "false", "reason": "estimate"}],
        "missed": [{"turn": 0, "speaker": "rigor", "kind": "logic", "quote": "costs", "note": "n"}],
    }
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(row))
    sheet = lr.check_sheet(labels, tmp_path)
    assert sheet.count("- you:") == 2
    assert "## E." not in sheet
    assert "## A. Flags Fable called FALSE" in sheet and "## B." not in sheet
    assert "- **Context**: costs 12 euros" in sheet


def test_novelty_reads_pilot_and_new_rows() -> None:
    lr = labeller()
    assert [lr.novelty(t) for t in ({"new": True}, {"new": False})] == ["new", "repeat"]
    row = {
        "key": "s:2",
        "latency_s": 1.0,
        "flags": [],
        "missed": [],
        "turns": [{"novelty": "partly_new"}, {"novelty": "repeat"}, {"novelty": "partly_new"}],
    }
    assert "turns 3 (partly_new 2, repeat 1)" in lr.brief(row)


def test_all_takes_flagless_rounds_but_pick_does_not(tmp_path: Path) -> None:
    lr = labeller()
    folder = session(tmp_path)
    rounds = [lr.load_round(folder, n) for n in (1, 2, 3)]
    assert [r.round for r in lr.pick(rounds, 9)] == [2]
    assert sorted(r.round for r in lr.pick(rounds, 9, flagged=False)) == [2, 3]
    assert lr.council(rounds[0]) == "x"


def test_check_sheet_sample_keeps_every_section(tmp_path: Path) -> None:
    lr = labeller()
    folder = session(tmp_path)
    flag = {"member": "rigor", "kind": "unsourced", "excerpt": "12 euros", "by": "engine"}
    row = {
        "key": f"{folder.name}:2",
        "flags_in": [flag] * 6,
        "flags": [{"id": i, "verdict": "false", "reason": "r"} for i in range(6)],
        "turns": [{"id": 0, "novelty": "repeat", "what_new": "repeats r1"}],
        "missed": [],
    }
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(row))
    sheet = lr.check_sheet(labels, tmp_path, sample=3)
    assert sheet.count("- you:") == 4  # 3 of 6 flags, and the one turn (at least 1)
    assert "- **Fable**: repeat: repeats r1" in sheet
