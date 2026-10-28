from __future__ import annotations

from pathlib import Path

from privy_council.config import Settings
from privy_council.engine import context, prompts
from privy_council.models.persona import BackendKind
from privy_council.models.session import DebateState, Message


def test_input_budget_uses_num_ctx_for_ollama(tmp_path: Path) -> None:
    settings = Settings.model_validate(
        {"root": tmp_path, "ollama": {"models": {"small": {"num_ctx": 4096}}}}
    )
    assert context.input_budget(settings, BackendKind.OLLAMA, "small", 1000) == 3096
    assert context.input_budget(settings, BackendKind.OLLAMA, "unknown", 1000) == 7192
    assert context.input_budget(settings, BackendKind.OLLAMA, "small", 9000) == (
        context.MIN_INPUT_TOKENS
    )
    claude = context.input_budget(settings, BackendKind.CLAUDE_CLI, "sonnet", 4000)
    assert claude == context.CLAUDE_CONTEXT_TOKENS - 4000


def test_build_prompt_keeps_everything_when_it_fits() -> None:
    prompt = context.build_prompt(
        system="sys",
        topic="Q?",
        task="Answer.",
        budget_tokens=10_000,
        state_text="Positions: none",
        history=["<A> one", "<B> two"],
    )
    assert prompt.dropped_messages == 0 and not prompt.state_clipped
    user = prompt.user
    assert user.index("# Question") < user.index("# Debate state") < user.index("# Recent")
    assert user.endswith("# Your task\nAnswer.")


def test_build_prompt_drops_oldest_messages_first() -> None:
    history = [f"<A> message number {i} " + "x" * 80 for i in range(20)]
    prompt = context.build_prompt(
        system="s" * 200, topic="Q?", task="T.", budget_tokens=200, history=history
    )
    assert prompt.dropped_messages > 0
    assert "message number 19" in prompt.user
    assert "message number 0 " not in prompt.user
    assert len(prompt.system) + len(prompt.user) <= 200 * 4


def test_build_prompt_clips_state_but_never_question_or_task() -> None:
    prompt = context.build_prompt(
        system="s",
        topic="The question",
        task="The task",
        budget_tokens=60,
        state_text="y" * 1000,
        history=["<A> hi"],
    )
    assert prompt.state_clipped and prompt.dropped_messages == 1
    assert "[…trimmed]" in prompt.user
    assert "The question" in prompt.user and "The task" in prompt.user


def test_message_filters() -> None:
    msgs = [
        Message(speaker="chair", text="open"),
        Message(speaker="user", text="note"),
        Message(speaker="rigor", text="take"),
        Message(speaker="system", text="notice"),
    ]
    assert [m.text for m in context.blind_messages(msgs, "chair")] == ["open", "note"]
    assert [m.text for m in context.recent_messages(msgs, 2)] == ["note", "take"]
    assert context.recent_messages(msgs, 0) == []


def test_state_text_none_until_built() -> None:
    assert context.state_text(DebateState(), str.upper) is None
    text = context.state_text(DebateState(positions={"rigor": "wait"}), str.upper)
    assert text is not None and "RIGOR: wait" in text


def test_user_facts_come_first_in_the_state() -> None:
    # First, so they survive when a tight budget clips the state from the end.
    state = DebateState(user_facts=["shaves every 4 days"], positions={"rigor": "wait"})
    text = context.state_text(state, str.upper)
    assert text is not None
    assert text.startswith(f"{prompts.STATE_USER_FACTS}\n- shaves every 4 days\n")


def test_floor_line_does_not_double_the_nick() -> None:
    from privy_council.engine.prompts import floor_line

    assert floor_line("Rigor", "@Rigor, @rigor: quantify it") == "@Rigor: quantify it"
    assert floor_line("Rigor", "respond to @Compounder") == "@Rigor: respond to @Compounder"
    assert floor_line("Plato", "Respond to @Kant") == "@Plato: Respond to @Kant"


def test_facts_come_after_the_question_and_respect_their_share() -> None:
    prompt = context.build_prompt(
        system="s",
        topic="Q?",
        task="T.",
        budget_tokens=1000,
        state_text="state",
        history=["<A> hi"],
        facts="f" * 10_000,
        facts_share=0.25,
    )
    user = prompt.user
    assert user.index("# Question") < user.index("# Fresh facts") < user.index("# Debate state")
    facts_section = user[user.index("# Fresh facts") : user.index("# Debate state")]
    assert len(facts_section) <= 1000 * 4 * 0.25 + 2
    assert "<A> hi" in user and user.endswith("T.")


def test_facts_text_newest_first() -> None:
    from datetime import UTC, datetime

    from privy_council.models.session import ResearchNote

    assert context.facts_text([]) is None
    old = ResearchNote(focus="old", text="a", ts=datetime(2026, 10, 1, tzinfo=UTC))
    new = ResearchNote(focus="new", text="b", ts=datetime(2026, 10, 2, tzinfo=UTC))
    text = context.facts_text([old, new])
    assert text is not None and text.index("new (searched 2026-10-02)") < text.index("old")


def test_memory_section_sits_after_facts_and_before_state() -> None:
    prompt = context.build_prompt(
        system="sys",
        topic="Q?",
        task="Answer.",
        budget_tokens=10_000,
        facts="fact",
        memory=("- my note", "- user fact"),
        state_text="Positions: none",
    )
    user = prompt.user
    assert user.index("# Fresh facts") < user.index("# Memory") < user.index("# Debate state")
    assert "Your private notes:\n- my note" in user
    assert "Facts the user stated in past sessions:\n- user fact" in user


def test_no_memory_no_section() -> None:
    prompt = context.build_prompt(
        system="s", topic="Q?", task="T", budget_tokens=1000, memory=("  ", "")
    )
    assert "# Memory" not in prompt.user


def test_memory_keeps_newest_bullets_and_private_first() -> None:
    private = "\n".join(f"- private {i:02d} " + "p" * 40 for i in range(30))
    shared = "\n".join(f"- shared {i:02d} " + "s" * 40 for i in range(30))
    text = context.memory_text(private, shared, 600)
    assert text is not None and len(text) <= 600
    assert "private 29" in text and "private 00" not in text
    assert "[…older notes trimmed]" in text
    assert "shared" not in text  # private notes took the whole room
    roomy = context.memory_text(private[:200], shared, 1200)
    assert roomy is not None and "shared 29" in roomy and "private 00" in roomy


def test_memory_respects_its_share_of_the_budget() -> None:
    notes = "\n".join(f"- note {i} " + "n" * 60 for i in range(200))
    prompt = context.build_prompt(
        system="s", topic="Q?", task="T", budget_tokens=2000, memory=(notes, ""), memory_share=0.2
    )
    memory = prompt.user.split("# Memory", 1)[1].split("# Your task")[0]
    assert len(memory) <= 0.2 * 2000 * 4
    assert "note 199" in memory


def test_one_huge_message_is_cut_not_allowed_to_push_out_the_rest() -> None:
    lines = ["<Ana> hi", "<Ben> hello", "<Ana> " + "x" * 10_000]
    kept, dropped = context.fit_messages(lines, 1000)
    assert dropped == 0 and kept[:2] == ["<Ana> hi", "<Ben> hello"]
    assert len(kept[2]) <= 500 and kept[2].endswith(prompts.TRIMMED_MARK)
