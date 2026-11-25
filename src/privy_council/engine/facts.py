"""Fact tags (M18): number the research bullets and find the members' citations.

Pure functions, no I/O. Each bullet of the session's research notes gets a tag such
as [F7], numbered oldest note first so a tag never changes when a note is added.
Members put the tag after a fact they state; the engine then checks the claim
against that one bullet instead of guessing where a number came from.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from privy_council.engine import prompts

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# "[F7]", "[F7, F9]", "[F7; F9]", a span "[F7-F9]", and a tier after the ids
# ("[F7, primary]", M20).
TAG = re.compile(r"\[(F\d+(?:\s*[,;/\-\u2013]\s*F\d+)*)(?:\s*,\s*[a-z]+)?\]")
_ID = re.compile(r"F\d+")
# A span of ids, "F12-F15" or "F12\u2013F15" (bench 2026-11-15); a longer one is a typo.
_SPAN = re.compile(r"F(\d+)\s*[\-\u2013]\s*F(\d+)")
_MAX_SPAN = 10
# A fact's tag at the start of its line, after the bullet, as `tag_block` writes it.
_TAGGED_LINE = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?\[(F\d+)(?:, [a-z]+)?\] ?", re.MULTILINE)
# "[F3]-[F6]": the moderator's way of writing a span.
_TAG_SPAN = re.compile(r"(?<=\d)(?:, [a-z]+)?\]\s*[\-\u2013]\s*\[(?=F\d)")
# A Markdown link; not a bare tag, which no "(" follows.
_LINK = re.compile(r"\[([^\[\]]+)\]\((https?://[^\s)]+)\)")
URL = re.compile(r"https?://[^\s)\]>\"']+")
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


def tier(text: str, primary_domains: Collection[str]) -> Literal["primary", "secondary"]:
    """Return "primary" when every URL of a fact is on an official domain (M20).

    A fact with no URL is secondary. A domain entry matches its host and subdomains:
    ".gov" or "gov" matches "www.sec.gov", "insee.fr" matches "www.insee.fr".
    """
    hosts = [(urlsplit(u).hostname or "").lower() for u in URL.findall(text)]
    suffixes = [d.lower().strip().lstrip(".") for d in primary_domains if d.strip()]
    if hosts and all(any(_on_domain(h, s) for s in suffixes) for h in hosts):
        return "primary"
    return "secondary"


def on_domains(url: str, domains: Collection[str]) -> bool:
    """Tell whether a URL's host is one of the domains or a subdomain of one."""
    host = (urlsplit(url).hostname or "").lower()
    return any(_on_domain(host, d.lower().strip().lstrip(".")) for d in domains if d.strip())


def _on_domain(host: str, suffix: str) -> bool:
    """Tell whether a host is the domain or one of its subdomains."""
    return host == suffix or host.endswith("." + suffix)


def tag_block(fact: Fact, label: str = "") -> str:
    """Return the bullet with its tag after the bullet marker: "- [F7] ...".

    With a `label` (the primary tier's word), the tag reads "[F7, primary]".
    """
    tag = (
        prompts.FACT_TAG_TIER.format(id=fact.id, tier=label)
        if label
        else (prompts.FACT_TAG.format(id=fact.id))
    )
    bullet = BULLET.match(fact.text)
    if bullet is None:
        return f"{tag} {fact.text}"
    return f"{fact.text[: bullet.end()]}{tag} {fact.text[bullet.end() :]}"


def tagged_notes(
    notes: Sequence[str], primary_domains: Collection[str] = (), primary_label: str = ""
) -> list[str]:
    """Return each note's text with its bullets tagged, in the notes' order.

    With `primary_domains`, a primary fact's tag carries `primary_label`; others don't,
    since most facts are secondary (6 % of the cited URLs were official, 2026-11-15).
    """
    out = ["" for _ in notes]
    for fact in numbered(notes):
        primary = primary_domains and tier(fact.text, primary_domains) == "primary"
        block = tag_block(fact, primary_label if primary else "")
        joined = out[fact.note_index]
        out[fact.note_index] = f"{joined}\n{block}" if joined else block
    return out


def without_urls(text: str) -> str:
    """Return facts with each URL cut to its site: "[$4bn](https://x.org/a)" is "$4bn (x.org)".

    For a small model's prompt: the links were a third of it, so the facts the
    moderator asked about were trimmed away (2026-11-25).
    """
    text = _LINK.sub(lambda m: f"{m[1]} ({_site(m[2])})", text)
    return URL.sub(lambda m: _site(m[0]), text)


def _site(url: str) -> str:
    """Return a URL's host without "www."."""
    return (urlsplit(url).hostname or "").removeprefix("www.")


def cited_ids(text: str) -> list[str]:
    """Return the fact ids a floor line names, a span of two tags ("[F3]-[F6]") included."""
    return tags(_TAG_SPAN.sub("-", text))


def split_named(text: str, ids: Collection[str]) -> tuple[str, str]:
    """Split a tagged note into the facts with these ids and the others."""
    if not ids:
        return "", text
    top: list[str] = []
    others: list[str] = []
    for block in blocks(text):
        tag = _TAGGED_LINE.match(block)
        (top if tag and tag[1] in ids else others).append(block)
    return "\n".join(top), "\n".join(others)


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
        inside = _SPAN.sub(_spelled_span, match.group(1))
        ids += [i for i in _ID.findall(inside) if i not in ids]
    return ids


def _spelled_span(span: re.Match[str]) -> str:
    """Write a span of ids out in full: "F3-F5" gives "F3 F4 F5"."""
    lo, hi = int(span[1]), int(span[2])
    if not 0 <= hi - lo <= _MAX_SPAN:
        return f"F{lo} F{hi}"
    return " ".join(f"F{n}" for n in range(lo, hi + 1))


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
