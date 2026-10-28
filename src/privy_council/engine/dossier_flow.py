"""The dossier: files the user attaches (`/attach`, `council new --attach`).

Small files go into every prompt verbatim. Larger ones are summarized once by
the moderator at attach time. The summary is saved next to the copy in dossier/
and kept in session.json, so a resume reuses it.
"""

from __future__ import annotations

from pathlib import Path

from privy_council.backends.base import BackendError
from privy_council.engine import context, prompts
from privy_council.engine.core import EngineCore
from privy_council.engine.dossier import (
    ALLOWED_TEXT,
    DossierError,
    attach_path,
    csv_head,
    csv_stats,
    read_attachable,
)
from privy_council.engine.notices import (
    NOTICE_ATTACH_FAILED,
    NOTICE_ATTACH_USAGE,
    NOTICE_ATTACHED,
    NOTICE_ATTACHED_SUMMARIZED,
    NOTICE_BUSY,
    NOTICE_SESSION_CLOSED,
    NOTICE_SUMMARIZE_CLIPPED,
    NOTICE_SUMMARIZING,
)
from privy_council.models.persona import BackendKind
from privy_council.models.session import (
    BudgetWarning,
    DossierAdded,
    DossierFile,
    ErrorEvent,
    Phase,
    TurnStarted,
)
from privy_council.util.tokens import estimate_tokens, tokens_to_chars

# Room kept for the question and the task around a file being summarized.
_SUMMARY_OVERHEAD_CHARS = 2000
# Always send at least this much of the file, even to a misconfigured tiny window.
_MIN_SUMMARY_CHARS = 2000


class DossierFlow(EngineCore):
    """Attach files; everyone sees them (or their summary) in every later prompt."""

    async def attach(self, arg: str) -> None:
        """Add a file to the dossier, summarizing it first when it is large."""
        path = attach_path(arg)
        if path is None:
            self._notice(NOTICE_ATTACH_USAGE.format(allowed=ALLOWED_TEXT))
            return
        if self.phase is Phase.DONE:
            self._notice(NOTICE_SESSION_CLOSED)
            return
        if self.busy:
            self._notice(NOTICE_BUSY)
            return
        try:
            content = read_attachable(path, self.settings.dossier.max_file_bytes)
        except DossierError as exc:
            self._notice(str(exc))
            return
        self.busy = True
        try:
            file = await self._dossier_file(path, content)
        finally:
            self.busy = False
        if file is None:
            return
        self.meta.runtime.dossier.append(file)
        self._save()
        self._emit(DossierAdded(file=file))
        notice = NOTICE_ATTACHED_SUMMARIZED if file.summarized else NOTICE_ATTACHED
        self._notice(notice.format(name=file.name, tokens=file.tokens, nick=self.moderator.name))

    async def _dossier_file(self, path: Path, content: str) -> DossierFile | None:
        """Copy the file and return its dossier entry; None if summarizing failed."""
        tokens = estimate_tokens(content)
        name = self._copy_name(path)
        is_csv = path.suffix.lower() == ".csv"
        stats = csv_stats(content) if is_csv else None
        if tokens <= self.settings.dossier.summarize_over_tokens:
            return DossierFile(name=name, text=content.strip(), tokens=tokens, stats=stats)
        self._notice(NOTICE_SUMMARIZING.format(nick=self.moderator.name, name=name, tokens=tokens))
        try:
            summary = await self._summarize_file(name, content, stats)
        except BackendError as exc:
            if self.store:
                self.store.remove_dossier_file(name)
            self._emit(ErrorEvent(text=NOTICE_ATTACH_FAILED.format(name=name, error=exc)))
            return None
        if self.store:
            self.store.save_dossier_summary(name, summary)
        head = csv_head(content, self.settings.dossier.csv_head_rows)
        if is_csv and head:
            summary += "\n\n" + prompts.DOSSIER_CSV_HEAD.format(head=head)
        return DossierFile(name=name, text=summary, tokens=tokens, summarized=True, stats=stats)

    def _copy_name(self, path: Path) -> str:
        """Copy the file into the session (when there is one); return its dossier name."""
        if self.store:
            return self.store.add_dossier_file(path)
        taken = {f.name for f in self.meta.runtime.dossier}
        name, n = path.name, 1
        while name in taken:
            n += 1
            name = f"{path.stem}-{n}{path.suffix}"
        return name

    async def _summarize_file(self, name: str, content: str, stats: str | None) -> str:
        """Have the moderator summarize a large file once (one call, not streamed).

        `stats`: a CSV's exact figures, given so the summary quotes rather than guesses.
        """
        if (
            self.registry.route(self.moderator, self.claude_exhausted).kind
            is BackendKind.CLAUDE_CLI
        ):
            for text in self.budget_warnings(1):
                self._emit(BudgetWarning(text=text))
        room = tokens_to_chars(self._budget(self.moderator)) - _SUMMARY_OVERHEAD_CHARS
        room -= len(prompts.DOSSIER_SUMMARY_SYSTEM) + len(self.meta.topic)
        room = max(room, _MIN_SUMMARY_CHARS)
        clipped = len(content) > room
        if clipped:
            percent = max(int(100 * room / len(content)), 1)
            self._notice(NOTICE_SUMMARIZE_CLIPPED.format(name=name, percent=percent))
            content = content[:room]
        task = prompts.DOSSIER_SUMMARY_TASK.format(
            name=name,
            clipped=prompts.DOSSIER_SUMMARY_CLIPPED if clipped else "",
            stats=prompts.DOSSIER_SUMMARY_STATS.format(stats=stats) if stats else "",
            content=content,
        )
        question = prompts.section(prompts.TITLE_QUESTION, self.meta.topic)
        user = f"{question}\n\n{task}" if self.meta.topic else task
        prompt = context.Prompt(system=prompts.DOSSIER_SUMMARY_SYSTEM, user=user)
        self._emit(TurnStarted(speaker=self.moderator.slug, round=self.meta.runtime.round))
        result, _ = await self._complete(self.moderator, prompt, "dossier")
        return result.text.strip()
