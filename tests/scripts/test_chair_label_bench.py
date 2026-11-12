"""scripts/chair_label_bench.py: a Chair's flaws are matched to the labelled errors."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from tests.conftest import REPO_ROOT


def bench() -> Any:
    """Import the script (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "chair_label_bench", REPO_ROOT / "scripts/chair_label_bench.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["chair_label_bench"] = module  # @dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


ROW = {
    "flags_in": [
        {"member": "rigor", "excerpt": "a runway of 12 months", "by": "moderator"},
        {"member": "rigor", "excerpt": "about 4.2 %", "by": "engine"},
        {"member": "skeptic", "excerpt": "the 30 % tax", "by": "engine"},
    ],
    "flags": [
        {"id": 0, "verdict": "real"},
        {"id": 1, "verdict": "real"},
        {"id": 2, "verdict": "false"},
    ],
    "missed": [
        {"speaker": "optimist", "quote": "so 3 x 40 is 150", "quote_found": True},
        {"speaker": "optimist", "quote": "not in the turn", "quote_found": False},
    ],
}


def test_same_passage_needs_most_words_of_the_shorter_quote() -> None:
    b = bench()
    assert b.same_passage("3 x 40 is 150", "…so 3 x 40 is 150, which means")
    assert not b.same_passage("3 x 40 is 150", "the yield is 4.2 %")


def test_targets_leave_out_real_engine_flags_and_unfound_quotes() -> None:
    b = bench()
    goals = b.targets(b.known_passages(ROW))
    assert [(k.member, k.source) for k in goals] == [("rigor", "moderator"), ("optimist", "missed")]


def test_score_flaws_sorts_caught_false_and_unlabelled() -> None:
    b = bench()
    flaws = [
        {"member": "optimist", "quote": "3 x 40 is 150", "kind": "arithmetic", "note": ""},
        {"member": "optimist", "quote": "3 x 40 is 150", "kind": "arithmetic", "note": ""},  # twice
        {"member": "skeptic", "quote": "the 30 % tax", "kind": "unsourced", "note": ""},
        {"member": "rigor", "quote": "about 4.2 %", "kind": "unsourced", "note": ""},  # engine's
        {"member": "rigor", "quote": "a runway of 12 months", "kind": "logic", "note": ""},
        {"member": "skeptic", "quote": "something else entirely", "kind": "logic", "note": ""},
        {"member": "rigor", "quote": "3 x 40 is 150", "kind": "logic", "note": ""},  # wrong member
    ]
    s = b.score_flaws(flaws, b.known_passages(ROW))
    assert (s["targets"], s["caught"], s["false_alarms"]) == (2, 2, 1)
    assert [f["quote"] for f in s["unlabelled"]] == ["something else entirely", "3 x 40 is 150"]
