"""Typing effect: chat turns stream `TextDelta`s; JSON calls and the transcript don't."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from rich.console import Console

from privy_council.backends.base import (
    BackendRateLimitError,
    CompletionRequest,
    CompletionResult,
    TextSink,
)
from privy_council.backends.fake import FakeBackend
from privy_council.models.persona import Persona
from privy_council.models.session import MessageEvent, TextDelta
from tests.engine.test_protocol import Harness, decision, standard_responder
from tests.engine.test_protocol import personas as personas


def shown(area: Any) -> str:
    """Return the typing area's content as plain text."""
    console = Console(width=118, record=True, file=io.StringIO())
    console.print(area.content)
    return console.export_text()


def typed(h: Harness, speaker: str) -> list[str]:
    """Return each streamed reply of `speaker`, split where a chunk restarts."""
    replies: list[str] = []
    for e in h.events:
        if isinstance(e, TextDelta) and e.speaker == speaker:
            if e.restart:
                replies.append("")
            replies[-1] += e.text
    return replies


async def test_chat_turns_stream_and_json_calls_dont(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    h = Harness(tmp_path, personas, standard_responder([decision([("rigor", "go")])]))
    await h.engine.step()  # opening
    await h.engine.step()  # blind round + moderator update (JSON)
    assert typed(h, "chair") == [h.texts("chair")[0]]  # the opening only
    for slug in ["rigor", "compounder", "pragmatist", "contrarian"]:
        assert typed(h, slug) == h.texts(slug)
    await h.engine.ask("rigor", "Why?")
    assert typed(h, "rigor")[-1] == h.texts("rigor")[-1]
    on_disk = (h.store.directory / "transcript.jsonl").read_text()
    assert '"kind":"delta"' not in on_disk


class LimitedMidReply(FakeBackend):
    """A Claude fake that types part of the reply, then hits the usage limit."""

    async def complete(
        self, req: CompletionRequest, on_text: TextSink | None = None
    ) -> CompletionResult:
        """Stream a fragment and fail with a usage limit."""
        if on_text is not None:
            on_text("I think the PEA")
        raise BackendRateLimitError("limit")


async def test_a_fallback_retry_restarts_the_typing(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    responder = standard_responder([])
    h = Harness(tmp_path, personas, responder, fallback=True, claude=LimitedMidReply(name="claude"))
    await h.engine.step()  # the Chair's opening: Claude fails mid-reply, local retries
    opening = next(
        e.message.text
        for e in h.events
        if isinstance(e, MessageEvent) and e.message.speaker == "chair"
    )
    assert typed(h, "chair") == ["I think the PEA", opening]


async def test_tui_typing_area(tmp_path: Path, personas: dict[str, Persona]) -> None:
    from privy_council.tui.app import CouncilApp
    from privy_council.tui.widgets import TypingArea

    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.workers.wait_for_complete()
        area = app.query_one(TypingArea)
        assert not area.has_class("visible")
        events: list[Any] = [
            TextDelta(speaker="compounder", text="Max the ", restart=True),
            TextDelta(speaker="rigor", text="Fund first", restart=True),
            TextDelta(speaker="compounder", text="PEA now."),
        ]
        for event in events:
            app.on_engine_event(event)
        await pilot.pause()
        text = shown(area)
        assert area.has_class("visible")
        assert "<Compounder> Max the PEA now." in text
        assert "Rigor is typing…" in text
        assert area.buffers["compounder"] == "Max the PEA now."
        h.engine.say("hi")  # a message from someone else leaves the typing alone
        assert set(area.buffers) == {"compounder", "rigor"}
        h.engine._post_result(  # Compounder finishes: its reply moves to the chat
            h.engine.members["compounder"],
            CompletionResult(text="Max the PEA now.", latency_s=0, backend="f", model="m"),
            False,
        )
        await pilot.pause()
        assert list(area.buffers) == ["rigor"]
        assert "<Rigor> Fund first" in shown(area)
        app.on_engine_event(TextDelta(speaker="rigor", text="Retry", restart=True))
        assert area.buffers["rigor"] == "Retry"


async def test_typing_area_grows_to_half_the_chat_column(
    tmp_path: Path, personas: dict[str, Persona]
) -> None:
    from privy_council.tui.app import CouncilApp
    from privy_council.tui.widgets import TypingArea

    h = Harness(tmp_path, personas, standard_responder([]))
    app = CouncilApp(h.engine)
    async with app.run_test(size=(120, 60)) as pilot:
        await app.workers.wait_for_complete()
        area = app.query_one(TypingArea)
        long_reply = "\n".join(f"line {i}" for i in range(100))
        app.on_engine_event(TextDelta(speaker="rigor", text=long_reply, restart=True))
        await pilot.pause()
        column = app.query_one("#chat-col").size.height
        text = shown(area)
        assert "line 99" in text  # the end is shown
        assert len(text.splitlines()) == column // 2 - 1  # half the column, minus the border
        assert area.outer_size.height == column // 2  # with its border
