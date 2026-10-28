"""TUI smoke tests with Textual's pilot and fake backends."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual.events import Paste
from textual.geometry import Offset
from textual.pilot import Pilot
from textual.selection import Selection
from textual.widgets import Static

from privy_council.backends.base import CompletionRequest, CompletionResult, TextSink
from privy_council.backends.fake import FakeBackend
from privy_council.engine.budget import UsageEntry
from privy_council.models.persona import Persona
from privy_council.models.session import SessionState
from privy_council.tui.app import CouncilApp
from privy_council.tui.widgets import ChatInput, ChatLog, NickList, StatePanel, TopBar
from tests.engine.test_memory import USER_LINE, memory_responder
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas
from tests.engine.test_resume import Resumed
from tests.engine.test_topic import topicless
from tests.engine.test_user_questions import QUESTION, asking

SIZE = (160, 50)


def chat_text(app: CouncilApp) -> str:
    """Return the chat log as plain text."""
    return "\n".join(line.text for line in app.query_one(ChatLog).lines)


async def submit(pilot: Pilot[None], line: str) -> None:
    """Type a line into the input and press Enter, then let workers finish."""
    app = pilot.app
    assert isinstance(app, CouncilApp)
    app.query_one(ChatInput).value = line
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_opening_step_state_pane_and_help(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "respond to Compounder")])]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "<Chair> Restating" in chat_text(app)
        assert f"Live log: {h.store.transcript_md}" in chat_text(app).replace("\n", "")
        assert h.state() is SessionState.PAUSED

        await submit(pilot, "/step")
        text = chat_text(app)
        assert "-- Round 1 — blind takes --" in text
        assert "<Rigor> rigor says something" in text
        assert "round 1/6" in str(app.query_one(TopBar).render())
        nicks = app.query_one(NickList)
        assert set(nicks.status.values()) == {"idle"}
        assert "Rigor: fund first" in str(app.query_one("#state-body").render())
        assert "Consensus: none · decision: open · stalled rounds: 0" in str(
            app.query_one("#state-body").render()
        )

        panel = app.query_one(StatePanel)
        assert not panel.has_class("visible")
        await pilot.press("f2")
        assert panel.has_class("visible")

        await submit(pilot, "/help")
        await submit(pilot, "/bogus")
        await submit(pilot, "I forgot: variable income")
        text = chat_text(app)
        assert "/run" in text and "Esc pause" in text
        assert "unknown command /bogus" in text
        assert "<User> I forgot: variable income" in text


async def test_ask_and_close(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/ask @Rigor worst case?")
        await submit(pilot, "/ask")
        assert "<User> @Rigor worst case?" in chat_text(app)
        assert any(r.tag.purpose == "ask" for r in h.ollama.requests)
        assert "usage: /ask @nick <question>" in chat_text(app)
        await submit(pilot, "/as Ben I rent")
        await submit(pilot, "/as Ben")
        assert "<Ben> I rent" in chat_text(app)
        assert "usage: /as <name> <text>" in chat_text(app)
        await submit(pilot, "/close")
        assert h.state() is SessionState.CLOSED
        assert "Consensus" in chat_text(app)  # the synthesis, rendered as Markdown


async def test_run_needs_confirmation_over_cap(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "x")]) for _ in range(2)]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    for _ in range(39):
        h.ledger.record(UsageEntry(session=h.store.id, model="sonnet"))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/run 2")
        assert "Type /run 2 again to confirm" in chat_text(app)
        assert h.engine.meta.runtime.round == 0
        await submit(pilot, "/run 2")
        assert h.engine.meta.runtime.round == 2
        assert "Claude 39/40" in str(app.query_one(TopBar).render())


async def test_more_raises_the_round_counter(tmp_path: Path, personas: dict[str, Persona]) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/more x")
        assert "usage: /more N (N ≥ 1)" in chat_text(app)
        await submit(pilot, "/more 2")
        assert "round 0/8" in str(app.query_one(TopBar).render())


class GatedBackend(FakeBackend):
    """A fake whose calls for `gated` personas wait until the gate opens."""

    def __init__(self, gated: set[str], **kwargs: Any) -> None:
        """Set up the gate."""
        super().__init__(**kwargs)
        self.gated = gated
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Block gated personas' turns on the gate."""
        if req.tag.persona in self.gated and req.tag.purpose == "turn":
            self.waiting.set()
            await self.gate.wait()
        return await super().complete(req, on_text)


async def test_escape_pauses_a_run_at_the_turn_boundary(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    responder = standard_responder([decision([("rigor", "x")])])
    gated = GatedBackend({"rigor"}, responder=responder, name="ollama")
    h = Harness(tmp_path, personas, responder, ollama=gated)
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        app.query_one(ChatInput).value = "/run 3"
        await pilot.press("enter")
        await asyncio.wait_for(gated.waiting.wait(), 5)
        assert "thinking…" in app.query_one(NickList).status.values()
        assert "running" in str(app.query_one(TopBar).render())
        await pilot.press("escape")
        assert "Pausing after the current turn…" in chat_text(app)
        gated.gate.set()  # Rigor's in-flight call completes; it is never cut
        await app.workers.wait_for_complete()
        await pilot.pause()
        rt = h.engine.meta.runtime
        assert h.state() is SessionState.PAUSED and rt.round_in_progress
        assert [t.speaker for t in rt.pending] == ["pragmatist", "contrarian"]
        assert "<Rigor>" in chat_text(app)
        assert "Paused. /step or /run N to continue" in chat_text(app)
        assert "paused" in str(app.query_one(TopBar).render())


async def test_chat_rewraps_when_state_pane_toggles(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, lambda req: "word " * 40)
    app = CouncilApp(h.engine)
    async with app.run_test(size=(110, 30)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        chat = app.query_one(ChatLog)
        await pilot.press("f2")
        await pilot.pause()
        width = chat.scrollable_content_region.width
        assert width < 60
        assert max(len(line.text.rstrip()) for line in chat.lines) <= width


async def test_multiline_paste_is_sent_as_one_message(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        field = app.query_one(ChatInput)
        field.post_message(Paste("To @Rigor: 6 months saved.\r\nTo @Compounder: no match.\n\n"))
        await pilot.pause()
        assert field.value == "To @Rigor: 6 months saved. ⏎ To @Compounder: no match."
        await pilot.press("enter")
        await pilot.pause()
        assert h.texts("user") == ["To @Rigor: 6 months saved.\nTo @Compounder: no match."]
        field.post_message(Paste("single line"))
        await pilot.pause()
        assert field.value == "single line"


async def test_research_command_shows_queries_and_facts(
    tmp_path: Path,
    personas: dict[str, Persona],
) -> None:
    from tests.engine.test_research import responder

    h = Harness(tmp_path, personas, responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/research gold price today")
        text = chat_text(app)
        assert "* Chair searched the web: gold price" in text
        assert "-- research (Claude web search): gold price today --" in text
        assert "$4,170/oz" in text
        assert app.query_one(NickList).status["chair"] == "idle"  # not stuck on thinking…


async def test_close_keep_forget_and_quit_guard(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    respond = memory_responder()
    h = Harness(tmp_path, personas, respond)
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/step")
        await submit(pilot, USER_LINE)
        await submit(pilot, "/close")
        text = chat_text(app)
        assert "shared memory: proposed facts" in text
        assert "1. " in text and "rents a flat in Lyon" in text
        assert "Rigor saved 1 memory note(s)" in text

        await submit(pilot, "/quit")
        assert "/quit again to leave without saving them" in chat_text(app)
        await submit(pilot, "/keep 1")
        assert "Saved 1 fact(s) to shared memory" in chat_text(app)

        await submit(pilot, "/forget shared")
        assert "Type /forget shared again to confirm" in chat_text(app)
        await submit(pilot, "/forget shared")
        assert "Forgot the shared facts about you" in chat_text(app)


async def test_question_typed_in_the_tui_starts_the_session(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    engine, store, _ = topicless(tmp_path, personas)
    app = CouncilApp(engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Type your question to start" in chat_text(app)
        assert "(type your question)" in str(app.query_one(TopBar).render())
        await submit(pilot, "/step")
        assert "No question yet" in chat_text(app)
        await submit(pilot, "Rent or buy?")
        assert "<Chair> Restating" in chat_text(app)
        assert "Rent or buy?" in str(app.query_one(TopBar).render())
        assert f"Live log: {store.transcript_md}" in chat_text(app).replace("\n", "")


async def test_chair_questions_are_highlighted(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [
        asking([("rigor", "go")], QUESTION),
        decision([("rigor", "go")]),
    ]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/run 3")
        assert h.engine.meta.runtime.round == 1
        text = chat_text(app)
        assert "▶ <Chair> question for you:" in text
        assert f"@User {QUESTION}" in text
        assert "/run paused for Chair's question" in text
        assert "↳ answer in plain text, then /step" in text
        assert "answer the Chair here" in app.query_one(ChatInput).placeholder
        await submit(pilot, "2 months")
        await submit(pilot, "/step")
        assert "/help for commands" in app.query_one(ChatInput).placeholder


def unread_line(app: CouncilApp) -> Static:
    """Return the "new messages" line under the chat."""
    return app.query_one("#unread", Static)


async def fill_chat(pilot: Pilot[None], count: int, prefix: str) -> None:
    """Write `count` numbered notices and let the log settle."""
    app = pilot.app
    assert isinstance(app, CouncilApp)
    for i in range(count):
        app.chat.notice(f"#{i} {prefix}")
    await pilot.pause()
    await pilot.pause()


async def test_chat_stays_put_while_scrolled_up(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=(110, 30)) as pilot:
        await app.workers.wait_for_complete()
        await fill_chat(pilot, 80, "old")
        chat = app.chat
        assert chat.is_vertical_scroll_end and chat.following

        await pilot.press("pageup")
        await pilot.pause()
        top = chat.scroll_y
        assert not chat.following

        await fill_chat(pilot, 3, "new")
        assert chat.scroll_y == top
        assert unread_line(app).has_class("visible")
        assert "↓ 3 new messages (End to jump)" in str(unread_line(app).render())

        await pilot.press("end")
        await pilot.pause()
        await pilot.pause()
        assert chat.is_vertical_scroll_end and chat.following
        assert not unread_line(app).has_class("visible")

        # Scrolling back down by hand also follows again.
        await pilot.press("pageup")
        await fill_chat(pilot, 1, "more")
        assert "↓ 1 new message (End" in str(unread_line(app).render())
        chat.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert chat.following and not unread_line(app).has_class("visible")
        await fill_chat(pilot, 2, "tail")
        assert chat.is_vertical_scroll_end


async def test_end_moves_the_cursor_when_following_and_sending_jumps_down(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=(110, 30)) as pilot:
        await app.workers.wait_for_complete()
        await fill_chat(pilot, 80, "old")
        box = app.query_one(ChatInput)
        box.value = "hello"
        box.cursor_position = 0
        await pilot.press("end")
        assert box.cursor_position == len("hello")

        await pilot.press("pageup")
        await pilot.pause()
        assert not app.chat.following
        await submit(pilot, "/state")
        await pilot.pause()
        assert app.chat.following and app.chat.is_vertical_scroll_end


async def test_resize_keeps_the_view_while_scrolled_up(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=(110, 30)) as pilot:
        await app.workers.wait_for_complete()
        await fill_chat(pilot, 60, "word " * 25 + "line")  # long: wraps differently later
        chat = app.chat
        top = next(i for i, line in enumerate(chat.lines) if line.text.startswith("#30 "))
        chat.scroll_to(y=top, animate=False, immediate=True)
        await pilot.pause()
        assert not chat.following
        height = len(chat.lines)

        await pilot.press("f2")  # the state pane narrows the chat: a re-wrap
        await pilot.pause()
        await pilot.pause()
        assert len(chat.lines) > height  # the re-wrap did change line positions
        assert chat.lines[round(chat.scroll_y)].text.startswith("#30 ")
        assert not chat.following


async def test_resumed_session_replays_history_without_stepping(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "go")])]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    await h.engine.step()
    await h.engine.step()
    r = Resumed(h, h.store.directory, personas, standard_responder([]))
    app = CouncilApp(r.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = chat_text(app)
        assert "<Chair> Restating" in text
        assert "<Rigor> rigor says something" in text
        assert "* Resumed after round 1. /step runs the next one" in text
        assert "keep 3 months of expenses" in str(app.query_one("#state-body", Static).render())
        assert r.turns() == []  # nothing ran on its own
        assert r.engine.state is SessionState.PAUSED


async def test_summon_and_kick_update_the_nick_list(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        nicks = app.query_one(NickList)
        await submit(pilot, "/summon skeptic")
        assert "Skeptic" in [p.name for p in nicks.personas]
        assert nicks.status["skeptic"] == "idle"
        assert "* Skeptic joined the channel and speaks next round" in chat_text(app)
        await submit(pilot, "/kick rigor")
        assert "rigor" not in nicks.status
        assert "* Rigor left the channel" in chat_text(app)


async def test_vote_table_and_summary(tmp_path: Path, personas: dict[str, Persona]) -> None:
    from tests.engine.test_review import SUMMARY, reviewing

    h = Harness(tmp_path, personas, reviewing([decision([("rigor", "go")])]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await submit(pilot, "/step")
        await submit(pilot, "/vote")
        text = chat_text(app)
        assert "-- vote after round 1 --" in text
        assert "Compounder" in text and "70%" in text and "compounder: fund first" in text
        assert set(app.query_one(NickList).status.values()) == {"idle"}
        await submit(pilot, "/summary")
        assert SUMMARY.splitlines()[-1] in chat_text(app)


async def test_attach_shows_the_file_in_the_chat(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    (tmp_path / "c.md").write_text("no crypto")
    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine, attach=[tmp_path / "c.md"])
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = chat_text(app)
        assert "-- dossier: c.md (~" in text
        assert text.index("-- dossier: c.md") < text.index(
            "<Chair> Restating"
        )  # before the opening
        await submit(pilot, "/attach nope.csv")
        assert "* No file at nope.csv" in chat_text(app)


async def test_long_question_keeps_the_round_counter_on_the_bar(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    engine, _, _ = topicless(tmp_path, personas)
    app = CouncilApp(engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await submit(pilot, "Should I rent or buy? " * 20 + "\nSecond line")
        bar = app.query_one(TopBar)
        text = str(bar.render())
        assert "round 0/6" in text and "…" in text
        assert "\n" not in text and len(text) <= bar.content_size.width


async def test_chat_selection_copies_text_without_row_padding(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    decisions: list[dict[str, Any] | str] = [decision([("rigor", "respond to Compounder")])]
    h = Harness(tmp_path, personas, standard_responder(decisions))
    app = CouncilApp(h.engine)
    async with app.run_test(size=SIZE) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        chat = app.query_one(ChatLog)
        app.screen.selections = {chat: Selection(None, None)}
        await pilot.pause()
        copied = app.screen.get_selected_text()
        assert copied is not None and "<Chair> Restating" in copied
        assert all(line == line.rstrip() for line in copied.splitlines())

        # Rows carry offsets for mouse selection, and the selected span is highlighted.
        app.screen.selections = {chat: Selection(Offset(0, 0), Offset(5, 0))}
        strip = chat.render_line(0)
        assert strip.text == chat.lines[0].text.ljust(strip.cell_length)
        style = app.screen.get_component_rich_style("screen--selection")
        first = next(iter(strip))
        assert first.style is not None and first.style.meta["offset"] == (0, 0)
        assert first.style.bgcolor == style.bgcolor
