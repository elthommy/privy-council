"""Tests for scripts/load_gate.py (no Ollama, no journal)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import load_gate

from privy_council.config import OllamaModelConfig

TURN = {"temperature": 0.7, "max_output": 1500, "system": "s", "user": "u", "spec": "macro:2"}
GOOD = load_gate.Run(
    version="0.40.16",
    model="m",
    index=0,
    starts=1,
    oom_lines=0,
    min_free_mib=1500,
    load_s=8.0,
    total_s=30.0,
    tok_s=20.0,
    tokens=400,
    done_reason="stop",
    error="",
    spec="macro:2",
    text="An answer.",
)


def test_a_hidden_retry_is_counted():
    log = (
        'msg="starting llama-server" cmd="x"\n'
        "allocating 1653.35 MiB on device 0: cudaMalloc failed: out of memory\n"
        'msg="starting llama-server" cmd="x --no-mmproj-offload"\n'
    )
    assert load_gate.journal_counts(log) == (2, 1)


def test_passes_only_on_one_start_and_a_whole_answer():
    assert GOOD.passed
    bad = [{"starts": 2}, {"oom_lines": 1}, {"done_reason": "length"}, {"text": " "}]
    assert not any(replace(GOOD, **change).passed for change in bad)


def test_nodraft_sends_the_option_under_the_plain_name():
    config = OllamaModelConfig(think=False)
    body = load_gate.payload("bolt:latest+nodraft", config, TURN)
    assert body["model"] == "bolt:latest"
    assert body["options"]["draft_num_predict"] == 0
    assert "draft_num_predict" not in load_gate.payload("bolt:latest", config, TURN)["options"]


def test_summary_has_one_line_per_model():
    runs = [GOOD, replace(GOOD, index=1, starts=2)]
    assert "pass 1/2" in load_gate.summary(runs)
