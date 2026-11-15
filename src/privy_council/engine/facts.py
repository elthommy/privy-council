"""Fact tags (M18): number the research bullets and find the members' citations.

Pure functions, no I/O. Each bullet of the session's research notes gets a tag such
as [F7], numbered oldest note first so a tag never changes when a note is added.
Members put the tag after a fact they state; the engine then checks the claim
against that one bullet instead of guessing where a number came from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from privy_council.engine import prompts

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# "[F7]", "[F7, F9]", "[F7; F9]", and a tier after the ids ("[F7, primary]", M20).
TAG = re.compile(r"\[(F\d+(?:\s*[,;/]\s*F\d+)*)(?:\s*,\s*[a-z]+)?\]")
_ID = re.compile(r"F\d+")
# A fact's tag at the start of its line, after the bullet, as `tag_block` writes it.
_TAGGED_LINE = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?\[(F\d+)\] ?", re.MULTILINE)
# Where a sentence ends: after . ! or ? (and any tags right after it) before a space,
# or at a line break, so a list item is a sentence of its own. "5.5%" has no space.
_SENTENCE_END = re.compile(r"[.!?](?:\s*" + TAG.pattern + r")*(?=\s)|\n")


@dataclass(frozen=True)
class Fact:
    """One research bullet with its tag."""

    id: str  # "F7"
    text: str  # the bullet, continuation lines included
    note_index: int  # which note it came from (0 = oldest)


def blocks(text: str) -> list[str]:
    """Split a summary into bullets, each with its continuation lines."""
    out: list[str] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if BULLET.match(line) or not out:
            out.append(line)
        else:
            out[-1] += "\n" + line
    return out


def numbered(notes: Sequence[str]) -> list[Fact]:
    """Number every bullet of the notes (oldest first): F1, F2, ..."""
    out: list[Fact] = []
    for index, note in enumerate(notes):
        for block in blocks(note):
            out.append(Fact(id=f"F{len(out) + 1}", text=block, note_index=index))
    return out


def tag_block(fact: Fact) -> str:
    """Return the bullet with its tag after the bullet marker: "- [F7] ..."."""
    tag = prompts.FACT_TAG.format(id=fact.id)
    bullet = BULLET.match(fact.text)
    if bullet is None:
        return f"{tag} {fact.text}"
    return f"{fact.text[: bullet.end()]}{tag} {fact.text[bullet.end() :]}"


def tagged_notes(notes: Sequence[str]) -> list[str]:
    """Return each note's text with its bullets tagged, in the notes' order."""
    out = ["" for _ in notes]
    for fact in numbered(notes):
        joined = out[fact.note_index]
        out[fact.note_index] = f"{joined}\n{tag_block(fact)}" if joined else tag_block(fact)
    return out


def facts_section(prompt: str) -> str:
    """Return a prompt's fresh-facts section, or "" when it has none."""
    head = prompts.section(prompts.TITLE_FACTS, "")
    start = prompt.find(head)
    if start < 0:
        return ""
    end = prompt.find("\n\n# ", start + len(head))
    return prompt[start : end if end >= 0 else len(prompt)]


def in_prompt(section: str) -> dict[str, str]:
    """Return the facts a prompt's facts section shows, by tag (trimmed ones are absent).

    A fact runs from its tag to the next tagged line or the next blank line.
    """
    found = list(_TAGGED_LINE.finditer(section))
    out: dict[str, str] = {}
    for i, match in enumerate(found):
        stop = found[i + 1].start() if i + 1 < len(found) else len(section)
        body = section[match.end() : stop]
        out[match.group(1)] = body.split("\n\n", 1)[0].strip()
    return out


def tags(sentence: str) -> list[str]:
    """Return the fact ids a sentence cites, in order, without repeats."""
    ids: list[str] = []
    for match in TAG.finditer(sentence):
        ids += [i for i in _ID.findall(match.group(1)) if i not in ids]
    return ids


def sentences(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) spans of a turn's sentences and list items."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def tagged_sentences(text: str) -> list[tuple[int, int, list[str]]]:
    """Return the sentences that cite facts, with the ids each one cites."""
    out: list[tuple[int, int, list[str]]] = []
    for start, end in sentences(text):
        ids = tags(text[start:end])
        if ids:
            out.append((start, end, ids))
    return out
