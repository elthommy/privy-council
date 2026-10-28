"""Chat names (usernames → @names) and rendering events as posts."""

from __future__ import annotations

import pytest

from privy_council.chat.frontend import council_choices
from privy_council.chat.names import NameBook, clean_name
from privy_council.chat.render import (
    GENERIC_ERROR,
    HIDDEN_PATH,
    Post,
    chat_notice,
    hide_paths,
    render_event,
    split_text,
)
from privy_council.engine import notices
from privy_council.models.persona import Role
from privy_council.models.session import ErrorEvent, Message, MessageEvent, RoundStarted


@pytest.mark.parametrize(
    ("username", "name"),
    [
        ("ana.martin", "ana-martin"),
        ("Ben_42", "ben_42"),
        ("__x", "x"),
        ("...", "guest"),
        ("é", "guest"),
    ],
)
def test_clean_name(username: str, name: str) -> None:
    assert clean_name(username) == name


def test_names_are_frozen_and_suffixed_on_collisions() -> None:
    book = NameBook(["User", "Rigor", "rigor"])
    assert book.name_for(1, "user") == "user-2"  # not the owner's name
    assert book.name_for(2, "rigor") == "rigor-2"  # not a persona
    assert book.name_for(3, "ana.martin") == "ana-martin"
    assert book.name_for(3, "ana.renamed") == "ana-martin"  # frozen for the session
    assert book.name_for(4, "x", override="Ben") == "ben"
    assert book.user_of("Ana-Martin") == 3 and book.user_of("nobody") is None
    assert NameBook([], {5: "ana"}).name_for(6, "ana") == "ana-2"  # names from a resume count


def test_split_text_prefers_headings_then_paragraphs() -> None:
    text = "## A\n" + "a" * 30 + "\n## B\n" + "b" * 30
    assert split_text(text, 40) == ["## A\n" + "a" * 30, "## B\n" + "b" * 30]
    paras = "\n\n".join(["x" * 20] * 3)
    assert split_text(paras, 45) == ["x" * 20 + "\n\n" + "x" * 20, "x" * 20]
    words = " ".join(["word"] * 60)
    parts = split_text(words, 120)
    assert len(parts) > 1 and all(len(p) <= 120 for p in parts) and " ".join(parts) == words
    assert split_text("short", 50) == ["short"] and split_text("  ", 50) == []


def test_hide_paths() -> None:
    text = "Live log: /home/thomas/x/data/sessions/s/transcript.md, see ~/code/a.md and 1/2"
    assert hide_paths(text) == f"Live log: {HIDDEN_PATH}, see {HIDDEN_PATH} and 1/2"


def nick(slug: str) -> str:
    """Display names of the test personas."""
    return slug.capitalize()


def test_render_posts_turns_as_personas_and_hides_people_and_errors() -> None:
    turn = MessageEvent(message=Message(speaker="rigor", text="My take."))
    assert render_event(turn, nick, "Chair", 2000) == [Post("**Rigor**\nMy take.", persona="Rigor")]
    person = MessageEvent(message=Message(speaker="user", text="hi", author="ana"))
    assert render_event(person, nick, "Chair", 2000) == []  # already in the channel
    question = MessageEvent(message=Message(speaker="chair", text="@ana how much?", to_user=True))
    assert render_event(question, nick, "Chair", 2000)[0].question
    error = ErrorEvent(text="Ollama error 500 at /home/thomas/x")
    assert render_event(error, nick, "Chair", 2000) == [Post(GENERIC_ERROR)]
    assert render_event(RoundStarted(round=2), nick, "Chair", 2000) == [Post("**— Round 2 —**")]


CHAT = {"run", "pause", "close", "ask"}


@pytest.mark.parametrize(
    ("notice", "posted"),
    [
        (notices.NOTICE_NEXT.format(speakers="Fan, Meta"), "Next round: Fan, Meta."),
        (notices.NOTICE_LANGUAGE.format(name="English", other="fr"), "Debate language: English."),
        (notices.NOTICE_PAUSED, "Paused."),
        (notices.NOTICE_SUMMARY_DONE, "Interim summary; the debate continues."),
        (notices.NOTICE_BUSY, notices.NOTICE_BUSY),  # /pause is a chat command
        ("Debate stopped early: loop. /close, or /step", "Debate stopped early: loop. /close."),
    ],
)
def test_notices_lose_the_repl_hints_the_chat_cannot_follow(notice: str, posted: str) -> None:
    assert chat_notice(notice, CHAT) == posted


def test_render_rewords_notices_for_the_chat() -> None:
    text = notices.NOTICE_NEXT.format(speakers="Fan")
    event = MessageEvent(message=Message(speaker="system", text=text))
    assert render_event(event, nick, "Chair", 2000, commands=CHAT) == [Post("Next round: Fan.")]


def test_council_choices_are_sorted_labelled_and_capped() -> None:
    councils = {"salon": "Philosophers", "finance": "Personal finance", "x": "", "long": "y" * 200}
    choices = council_choices(councils)
    assert [value for _, value in choices] == ["finance", "long", "salon", "x"]
    assert choices[0] == ("finance: Personal finance", "finance")
    assert choices[3] == ("x", "x")
    assert len(choices[1][0]) == 100 and choices[1][0].endswith("…")
    many = {f"c{i:02}": "d" for i in range(30)}
    assert len(council_choices(many)) == 25


def test_turns_are_headed_with_name_and_role_and_continued() -> None:
    roles = {"saver": Role.RISK_OFFICER, "contrarian": Role.DEVIL_ADVOCATE, "chair": Role.MODERATOR}
    turn = MessageEvent(message=Message(speaker="saver", text="Wait.\n\n" + "x" * 150))
    posts = render_event(turn, nick, "Chair", 160, roles.get)
    assert posts[0].text == "**Saver** · *risk officer*\nWait."
    assert posts[1].text.startswith("*(Saver, continued)*\nxxx")
    assert all(len(p.text) <= 160 and p.persona == "Saver" for p in posts)
    devil = MessageEvent(message=Message(speaker="contrarian", text="No."))
    assert render_event(devil, nick, "Chair", 2000, roles.get)[0].text.startswith(
        "**Contrarian** · *devil's advocate*"
    )
    question = MessageEvent(message=Message(speaker="chair", text="@ana how much?", to_user=True))
    assert render_event(question, nick, "Chair", 2000, roles.get)[0].text == (
        "**Chair** · *question*\n@ana how much?"
    )
