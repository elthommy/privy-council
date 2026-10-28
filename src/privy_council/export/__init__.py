"""Session export (M13): a saved session as a readable Markdown or HTML report."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from privy_council.export.html import render_html
from privy_council.export.markdown import render_markdown
from privy_council.export.report import ExportError, Report, build_report
from privy_council.models.persona import Persona

__all__ = ["EXPORTS_DIR", "ExportError", "ExportFormat", "export_session"]

EXPORTS_DIR = "exports"  # under the data dir, gitignored like sessions: reports hold personal data


class ExportFormat(StrEnum):
    """What `council sessions export` writes."""

    MD = "md"
    HTML = "html"
    ALL = "all"  # both files


def _render(report: Report, ext: str, flaws: bool) -> str:
    """Render the report in one concrete format."""
    return render_html(report, flaws) if ext == "html" else render_markdown(report, flaws)


def export_session(
    directory: Path,
    personas: Mapping[str, Persona],
    *,
    user_name: str,
    exports_dir: Path,
    fmt: ExportFormat = ExportFormat.MD,
    out: Path | None = None,
    flaws: bool = True,
) -> list[Path]:
    """Write the session's report(s) and return their paths.

    `out` is a file path for one format, or a directory; the default is
    `exports_dir/<session id>.<ext>`. Raises ExportError.
    """
    report = build_report(directory, personas, user_name)
    exts = ["md", "html"] if fmt is ExportFormat.ALL else [fmt.value]
    if out is not None and not out.is_dir() and len(exts) > 1:
        raise ExportError("--out must be a directory with --format all")
    written = []
    for ext in exts:
        if out is None or out.is_dir():
            path = (out or exports_dir) / f"{report.meta.id}.{ext}"
        else:
            path = out
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(report, ext, flaws), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"can't write {path}: {exc}") from exc
        written.append(path)
    return written
