from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from privy_council.engine.budget import (
    BudgetReport,
    Ledger,
    QuotaSnapshot,
    UsageEntry,
    UsageSummary,
    cap_warnings,
    quota_warnings,
)

NOW = datetime(2026, 10, 15, 21, 0).astimezone()


def entry(ts: datetime, session: str = "s1", ok: bool = True, tokens: int = 10) -> UsageEntry:
    """Build a ledger entry at `ts`."""
    return UsageEntry(ts=ts, session=session, model="claude-haiku-5-5", ok=ok, input_tokens=tokens)


def test_record_and_read_back(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "nested" / "usage.jsonl")
    ledger.record(entry(NOW))
    ledger.record(entry(NOW, ok=False))
    assert [e.ok for e in ledger.entries()] == [True, False]


def test_missing_file_is_empty(tmp_path: Path) -> None:
    assert Ledger(tmp_path / "usage.jsonl").entries() == []


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    ledger.record(entry(NOW))
    with ledger.path.open("a") as fh:
        fh.write('{"ts": "half a line\n\n')
    ledger.record(entry(NOW))
    assert len(ledger.entries()) == 2


def test_report_periods(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    start_of_day = NOW.replace(hour=0, minute=0)
    ledger.record(entry(NOW - timedelta(minutes=5), session="s1"))
    ledger.record(entry(start_of_day + timedelta(minutes=1), session="s2", ok=False))
    ledger.record(entry(start_of_day - timedelta(minutes=1), session="s1"))  # yesterday
    ledger.record(entry(NOW - timedelta(days=8), session="s1"))  # outside the week
    report = ledger.report(session="s1", now=NOW)
    assert report.session.calls == 3
    assert (report.today.calls, report.today.failed) == (2, 1)
    assert report.week.calls == 3
    assert report.week.input_tokens == 30


def test_cap_warnings() -> None:
    report = BudgetReport(session=UsageSummary(calls=34), today=UsageSummary(calls=100))
    assert cap_warnings(report, 5, session_cap=40, daily_cap=150) == []
    warnings = cap_warnings(report, 10, session_cap=40, daily_cap=150)
    assert warnings == ["~10 Claude calls would take session usage to 44/40 (soft cap)"]
    assert len(cap_warnings(report, 60, session_cap=40, daily_cap=150)) == 2
    assert cap_warnings(report, 1000, session_cap=0, daily_cap=0) == []


QUOTA_EVENT = {
    "status": "allowed",
    "unifiedWindows": {
        "five_hour": {"utilization": 0.85, "resetsAt": 1792105200},
        "seven_day": {"utilization": 0.08, "resetsAt": 1792353600},
    },
}


def test_quota_snapshot_from_event() -> None:
    q = QuotaSnapshot.from_event(QUOTA_EVENT)
    assert (q.status, q.five_hour, q.seven_day) == ("allowed", 0.85, 0.08)
    assert q.resets_at == q.five_hour_resets_at
    assert QuotaSnapshot.from_event({}).five_hour is None


def test_report_keeps_latest_quota(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    old = QuotaSnapshot(status="allowed", five_hour=0.1)
    new = QuotaSnapshot(status="allowed", five_hour=0.3)
    ledger.record(entry(NOW - timedelta(hours=2)).model_copy(update={"quota": old}))
    ledger.record(entry(NOW - timedelta(hours=1)).model_copy(update={"quota": new}))
    ledger.record(entry(NOW))  # no quota (e.g. a truncated call)
    report = ledger.report(now=NOW)
    assert report.quota == new
    assert report.quota_ts == NOW - timedelta(hours=1)


def test_quota_warnings() -> None:
    q = QuotaSnapshot.from_event(QUOTA_EVENT)
    [warning] = quota_warnings(q, threshold=0.8)
    assert warning.startswith("Claude 5-hour quota at 85%, resets ")
    assert quota_warnings(q, threshold=0.9) == []
    assert quota_warnings(None, threshold=0.0) == []
