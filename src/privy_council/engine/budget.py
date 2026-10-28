"""Claude usage ledger (`data/usage.jsonl`) and soft-cap checks.

One JSON line per Claude call, metadata only: prompts contain personal data and
are never written here. Call counts only cover calls made by council, but each
streamed call also carries the CLI's `rate_limit_event`, a snapshot of the real
Pro quota (all usage on the subscription), stored as `quota`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

LEDGER_FILE = "usage.jsonl"


def _epoch(value: Any) -> datetime | None:
    """Convert an epoch-seconds value to an aware datetime, if present."""
    return datetime.fromtimestamp(int(value), UTC) if isinstance(value, int | float) else None


class QuotaSnapshot(BaseModel):
    """Pro quota as reported by the CLI's `rate_limit_event` (utilization is 0..1)."""

    status: str | None = None  # "allowed" normally; anything else means limited
    five_hour: float | None = None
    five_hour_resets_at: datetime | None = None
    seven_day: float | None = None
    seven_day_resets_at: datetime | None = None

    @classmethod
    def from_event(cls, info: dict[str, Any]) -> QuotaSnapshot:
        """Parse the `rate_limit_info` object of a `rate_limit_event`."""
        windows = info.get("unifiedWindows") or {}
        five, seven = windows.get("five_hour") or {}, windows.get("seven_day") or {}
        return cls(
            status=info.get("status"),
            five_hour=five.get("utilization"),
            five_hour_resets_at=_epoch(five.get("resetsAt")),
            seven_day=seven.get("utilization"),
            seven_day_resets_at=_epoch(seven.get("resetsAt")),
        )

    @property
    def resets_at(self) -> datetime | None:
        """Return the reset time of the most constrained window."""
        if (self.seven_day or 0) > (self.five_hour or 0) and self.seven_day_resets_at:
            return self.seven_day_resets_at
        return self.five_hour_resets_at or self.seven_day_resets_at


class UsageEntry(BaseModel):
    """One recorded backend call."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    backend: str = "claude_cli"
    session: str | None = None
    persona: str | None = None
    purpose: str = "turn"
    model: str
    ok: bool = True
    error: str | None = None  # error class name when ok is False
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int | None = None
    latency_s: float = 0.0
    notional_usd: float | None = None  # list-price figure from the CLI; not billed on Pro
    truncated: bool = False  # cut off by the hard output cap (tokens are then estimated)
    quota: QuotaSnapshot | None = None


class UsageSummary(BaseModel):
    """Aggregated call counts and tokens for a period."""

    calls: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, entry: UsageEntry) -> None:
        """Accumulate one entry."""
        self.calls += 1
        self.failed += 0 if entry.ok else 1
        self.input_tokens += entry.input_tokens
        self.output_tokens += entry.output_tokens


class BudgetReport(BaseModel):
    """Usage for the current session, today and the last 7 days."""

    session: UsageSummary = Field(default_factory=UsageSummary)
    today: UsageSummary = Field(default_factory=UsageSummary)
    week: UsageSummary = Field(default_factory=UsageSummary)
    quota: QuotaSnapshot | None = None  # latest snapshot seen
    quota_ts: datetime | None = None


class Ledger:
    """Append-only JSONL ledger of Claude calls."""

    def __init__(self, path: Path) -> None:
        """Use the ledger file at `path` (created on first write)."""
        self.path = path

    @classmethod
    def in_data_dir(cls, data_dir: Path) -> Ledger:
        """Return the ledger stored in the configured data directory."""
        return cls(data_dir / LEDGER_FILE)

    def record(self, entry: UsageEntry) -> None:
        """Append one entry; a single short write keeps lines whole on crash."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def entries(self) -> list[UsageEntry]:
        """Read all entries, skipping (and logging) corrupt lines."""
        if not self.path.is_file():
            return []
        out = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(UsageEntry.model_validate_json(line))
            except ValidationError:
                log.warning("%s:%d: skipping corrupt ledger line", self.path, lineno)
        return out

    def report(self, session: str | None = None, now: datetime | None = None) -> BudgetReport:
        """Summarize usage for `session`, the local calendar day and the last 7 days."""
        now = (now or datetime.now(UTC)).astimezone()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        report = BudgetReport()
        for entry in self.entries():
            ts = entry.ts.astimezone()
            if session is not None and entry.session == session:
                report.session.add(entry)
            if ts >= day_start:
                report.today.add(entry)
            if ts >= week_start:
                report.week.add(entry)
            if entry.quota is not None:
                report.quota, report.quota_ts = entry.quota, entry.ts
        return report


def cap_warnings(
    report: BudgetReport, estimated_calls: int, session_cap: int, daily_cap: int
) -> list[str]:
    """Return warnings if `estimated_calls` more calls would cross a soft cap (0 = no cap)."""
    warnings = []
    checks = [
        ("session", report.session.calls, session_cap),
        ("today", report.today.calls, daily_cap),
    ]
    for label, used, cap in checks:
        if cap and used + estimated_calls > cap:
            warnings.append(
                f"~{estimated_calls} Claude calls would take {label} usage to "
                f"{used + estimated_calls}/{cap} (soft cap)"
            )
    return warnings


def quota_warnings(quota: QuotaSnapshot | None, threshold: float) -> list[str]:
    """Warn when a Pro quota window is at or above `threshold` utilization."""
    if quota is None:
        return []
    warnings = []
    for label, used, resets in [
        ("5-hour", quota.five_hour, quota.five_hour_resets_at),
        ("7-day", quota.seven_day, quota.seven_day_resets_at),
    ]:
        if used is not None and used >= threshold:
            when = f", resets {resets.astimezone():%a %H:%M}" if resets else ""
            warnings.append(f"Claude {label} quota at {used:.0%}{when}")
    return warnings
