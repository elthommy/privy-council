"""Session files under `data/sessions/<yyyy-mm-dd>_<council>_<topic-slug>/`.

`transcript.jsonl` is the source of truth: every event is appended as it
happens, so a crash loses at most the in-flight turn. The other files are
conveniences rewritten at boundaries.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from privy_council.engine.notices import RESEARCH_SOURCE_LABELS
from privy_council.engine.prompts import USER_NICK
from privy_council.models.session import (
    BudgetWarning,
    DebateState,
    DossierAdded,
    ErrorEvent,
    Event,
    MemoryProposed,
    MessageEvent,
    ResearchAdded,
    ResearchNote,
    RoundStarted,
    SessionMeta,
    VoteTaken,
)

SESSIONS_DIR = "sessions"
TRANSCRIPT_JSONL = "transcript.jsonl"
TRANSCRIPT_MD = "transcript.md"
SESSION_JSON = "session.json"
DEBATE_STATE_JSON = "debate_state.json"
SYNTHESIS_MD = "synthesis.md"
DOSSIER_DIR = "dossier"
RESEARCH_MD = "research.md"
RESEARCH_RAW_JSONL = "research_raw.jsonl"  # local search hits, for checking a note by hand
SUMMARY_SUFFIX = ".summary.md"  # the moderator's summary of a large attached file
# Symlink to the newest session, so `tail -f data/sessions/latest/transcript.md` works.
LATEST_LINK = "latest"
TOPIC_SLUG_MAX = 40

EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def topic_slug(topic: str) -> str:
    """Make a short filesystem-safe slug from a topic, cut at a word boundary."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "session"
    if len(slug) > TOPIC_SLUG_MAX:
        slug = slug[:TOPIC_SLUG_MAX].rsplit("-", 1)[0] or slug[:TOPIC_SLUG_MAX]
    return slug


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so readers never see half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _research_markdown(note: ResearchNote, level: str) -> str:
    """Render one research note as a Markdown section."""
    searched = "; ".join(note.searches) or "no web search"
    when = f"{note.ts.astimezone():%Y-%m-%d %H:%M}"
    source = RESEARCH_SOURCE_LABELS[note.source]
    read = f" · read: {'; '.join(note.fetched)}" if note.fetched else ""
    meta = f"*{when} · {source} · searched: {searched}{read}*"
    return f"{level} Research: {note.focus}\n\n{meta}\n\n{note.text}\n\n"


def _cell(text: str) -> str:
    """Make text safe for one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _session_name(now: datetime, council: str, topic: str) -> str:
    """Return a session directory name: `<date>_<council>_<topic-slug>`."""
    return f"{now.astimezone():%Y-%m-%d}_{council}_{topic_slug(topic)}"


def _free_directory(base: Path, name: str) -> Path:
    """Return `base/name`, or `base/name-N` if taken."""
    directory, n = base / name, 1
    while directory.exists():
        n += 1
        directory = base / f"{name}-{n}"
    return directory


def _point_latest(sessions_dir: Path, name: str) -> None:
    """Point the `latest` symlink at a session directory (replaced atomically)."""
    tmp = sessions_dir / f".{LATEST_LINK}.tmp"
    tmp.unlink(missing_ok=True)
    try:
        tmp.symlink_to(name, target_is_directory=True)
        os.replace(tmp, sessions_dir / LATEST_LINK)
    except OSError:
        tmp.unlink(missing_ok=True)  # a convenience only; never block a session on it


class SessionStore:
    """Reads and writes one session's directory."""

    def __init__(self, directory: Path, nicks: Mapping[str, str] | None = None) -> None:
        """Use an existing session directory; `nicks` maps slugs for transcript.md."""
        self.directory = directory
        self.nicks = dict(nicks or {})
        self.created = datetime.now().astimezone()

    @property
    def id(self) -> str:
        """Return the session id (the directory name)."""
        return self.directory.name

    @classmethod
    def create(
        cls,
        data_dir: Path,
        council: str,
        topic: str,
        now: datetime,
        nicks: Mapping[str, str] | None = None,
    ) -> SessionStore:
        """Create a fresh, uniquely named session directory."""
        base = data_dir / SESSIONS_DIR
        directory = _free_directory(base, _session_name(now, council, topic))
        directory.mkdir(parents=True, mode=0o700)
        store = cls(directory, nicks)
        store.created = now
        store._write_header(council, topic)
        _point_latest(base, directory.name)
        return store

    @classmethod
    def open(cls, directory: Path, nicks: Mapping[str, str] | None = None) -> SessionStore:
        """Reopen a saved session (for resume) and point `latest` at it."""
        store = cls(directory, nicks)
        store.created = store.load_meta().created_at
        _point_latest(directory.parent, directory.name)
        return store

    def retopic(self, council: str, topic: str) -> None:
        """Rename a fresh session's directory after its question (set after start).

        Only valid before the first event: transcript.md is just the header then.
        """
        base = self.directory.parent
        target = _free_directory(base, _session_name(self.created, council, topic))
        os.replace(self.directory, target)
        self.directory = target
        self._write_header(council, topic)
        _point_latest(base, target.name)

    def discard(self) -> None:
        """Delete this session's directory, and the `latest` link if it pointed here."""
        shutil.rmtree(self.directory, ignore_errors=True)
        latest = self.directory.parent / LATEST_LINK
        if latest.is_symlink() and not latest.exists():
            latest.unlink()

    def _write_header(self, council: str, topic: str) -> None:
        """(Re)write transcript.md with its header only."""
        when = f"{self.created.astimezone():%Y-%m-%d %H:%M}"
        header = f"# #{council} — {topic}\n\n*Started {when}*\n\n"
        (self.directory / TRANSCRIPT_MD).write_text(header, encoding="utf-8")

    @property
    def transcript_md(self) -> Path:
        """Return the live, human-readable transcript path."""
        return self.directory / TRANSCRIPT_MD

    def append(self, event: Event) -> None:
        """Append one event to transcript.jsonl (and chat lines to transcript.md)."""
        line = EVENT_ADAPTER.dump_json(event).decode() + "\n"
        with (self.directory / TRANSCRIPT_JSONL).open("a", encoding="utf-8") as fh:
            fh.write(line)
        md = self._markdown_line(event)
        if md is not None:
            with (self.directory / TRANSCRIPT_MD).open("a", encoding="utf-8") as fh:
                fh.write(md)

    def _markdown_line(self, event: Event) -> str | None:
        """Render a chat or round event for transcript.md; None for other events."""
        if isinstance(event, RoundStarted):
            return f"\n## {event.label or f'Round {event.round}'}\n\n"
        if isinstance(event, ResearchAdded):
            return _research_markdown(event.note, level="###")
        if isinstance(event, ErrorEvent):
            return f"> **Error:** {event.text}\n\n"
        if isinstance(event, BudgetWarning):
            return f"> **Budget:** {event.text}\n\n"
        if isinstance(event, MemoryProposed):
            items = event.facts or ([event.rewrite] if event.rewrite else [])
            kind = "facts" if event.facts else "merged file"
            body = "".join(f"> {line}\n" for item in items for line in item.splitlines())
            return f"> **Shared memory, proposed {kind}:**\n{body}\n"
        if isinstance(event, DossierAdded):
            how = "summarized" if event.file.summarized else "included verbatim"
            line = f"*Attached {event.file.name} (~{event.file.tokens:,} tokens, {how})*\n\n"
            shown = [event.file.text] if event.file.summarized else []
            shown += [event.file.stats] if event.file.stats else []
            if shown:
                rows = "\n\n".join(shown).splitlines()
                line += "".join(f"> {row}\n" for row in rows) + "\n"
            return line
        if isinstance(event, VoteTaken):
            return self._vote_markdown(event)
        if not isinstance(event, MessageEvent):
            return None
        msg = event.message
        if msg.speaker == "system":
            return f"*{msg.text}*\n\n"
        default = USER_NICK if msg.speaker == "user" else msg.speaker
        nick = msg.author or self.nicks.get(msg.speaker, default)
        flags = " *[fallback]*" if msg.fallback else ""
        flags += " *[cut off]*" if msg.truncated else ""
        flags += " *[question for you]*" if msg.to_user else ""
        return f"**[{msg.ts.astimezone():%H:%M}] <{nick}>**{flags} {msg.text}\n\n"

    def _vote_markdown(self, event: VoteTaken) -> str:
        """Render a vote as a Markdown table."""
        lines = [
            f"### Vote after round {event.round}\n",
            "| Member | Confidence | Position |",
            "|---|---|---|",
        ]
        for v in event.votes:
            nick = self.nicks.get(v.speaker, v.speaker)
            if v.error is not None:
                lines.append(f"| {nick} | — | *no vote: {_cell(v.error)}* |")
            else:
                lines.append(f"| {nick} | {v.confidence}% | {_cell(v.position)} |")
        return "\n".join(lines) + "\n\n"

    def events(self) -> list[Event]:
        """Read all events back from transcript.jsonl."""
        path = self.directory / TRANSCRIPT_JSONL
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [EVENT_ADAPTER.validate_json(line) for line in lines if line.strip()]

    def save_meta(self, meta: SessionMeta) -> None:
        """Rewrite session.json."""
        _write_atomic(self.directory / SESSION_JSON, meta.model_dump_json(indent=2))

    def load_meta(self) -> SessionMeta:
        """Read session.json."""
        text = (self.directory / SESSION_JSON).read_text(encoding="utf-8")
        return SessionMeta.model_validate_json(text)

    def save_debate_state(self, state: DebateState) -> None:
        """Rewrite debate_state.json."""
        _write_atomic(self.directory / DEBATE_STATE_JSON, state.model_dump_json(indent=2))

    def save_research(self, notes: list[ResearchNote]) -> None:
        """Rewrite dossier/research.md with all research so far."""
        dossier = self.directory / DOSSIER_DIR
        dossier.mkdir(exist_ok=True)
        text = "# Web research\n\n" + "".join(_research_markdown(n, "##") for n in notes)
        _write_atomic(dossier / RESEARCH_MD, text)

    def append_research_raw(self, record: str) -> None:
        """Append one local search's queries, hits and dropped bullets (a JSON line)."""
        with (self.directory / RESEARCH_RAW_JSONL).open("a", encoding="utf-8") as fh:
            fh.write(record.rstrip("\n") + "\n")

    def add_dossier_file(self, source: Path) -> str:
        """Copy an attached file into dossier/ under a free name, and return that name."""
        dossier = self.directory / DOSSIER_DIR
        dossier.mkdir(exist_ok=True)
        name, n = source.name, 1
        while name == RESEARCH_MD or (dossier / name).exists():
            n += 1
            name = f"{source.stem}-{n}{source.suffix}"
        shutil.copyfile(source, dossier / name)
        return name

    def save_dossier_summary(self, name: str, text: str) -> None:
        """Write the moderator's summary of an attached file next to its copy."""
        _write_atomic(
            self.directory / DOSSIER_DIR / f"{name}{SUMMARY_SUFFIX}", text.rstrip() + "\n"
        )

    def remove_dossier_file(self, name: str) -> None:
        """Delete an attached file's copy (an attach that failed)."""
        (self.directory / DOSSIER_DIR / name).unlink(missing_ok=True)

    def save_synthesis(self, text: str) -> None:
        """Write synthesis.md."""
        _write_atomic(self.directory / SYNTHESIS_MD, text.rstrip() + "\n")


class SessionLookupError(Exception):
    """No session, or more than one, matches what the user typed."""


@dataclass(frozen=True)
class SessionInfo:
    """One saved session, for `council sessions` (meta is None if session.json is unreadable)."""

    directory: Path
    meta: SessionMeta | None
    updated: datetime  # last write to session.json or the transcript

    @property
    def id(self) -> str:
        """Return the session id (the directory name)."""
        return self.directory.name


def _updated(directory: Path) -> datetime:
    """Return when a session last changed (newest of its main files)."""
    paths = [directory / SESSION_JSON, directory / TRANSCRIPT_JSONL, directory]
    stamp = max(p.stat().st_mtime for p in paths if p.exists())
    return datetime.fromtimestamp(stamp, UTC)


def list_sessions(data_dir: Path) -> list[SessionInfo]:
    """Return every saved session, most recently updated first."""
    base = data_dir / SESSIONS_DIR
    if not base.is_dir():
        return []
    infos = []
    for directory in base.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue  # `latest` and stray files
        try:
            meta: SessionMeta | None = SessionStore(directory).load_meta()
        except (OSError, ValidationError):
            meta = None
        infos.append(SessionInfo(directory, meta, _updated(directory)))
    return sorted(infos, key=lambda i: i.updated, reverse=True)


def find_session(data_dir: Path, ref: str | None) -> Path:
    """Resolve a session id, a unique part of one, or None/"latest" for the newest."""
    infos = list_sessions(data_dir)
    if not infos:
        raise SessionLookupError(f"No saved sessions in {data_dir / SESSIONS_DIR}")
    if ref is None or ref == LATEST_LINK:
        return infos[0].directory
    exact = [i for i in infos if i.id == ref]
    matches = exact or [i for i in infos if ref in i.id]
    if len(matches) == 1:
        return matches[0].directory
    if not matches:
        raise SessionLookupError(f"No session matches {ref!r} (council sessions list)")
    names = ", ".join(i.id for i in matches[:5])
    raise SessionLookupError(f"{ref!r} matches {len(matches)} sessions: {names}")


def remove_session(directory: Path) -> None:
    """Delete one session directory, and the `latest` link if it pointed there."""
    SessionStore(directory).discard()
