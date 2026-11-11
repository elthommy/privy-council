"""Label saved debate rounds with a strong Claude model, as a reference for local judges.

One call per round. The model gets what the members could see (question, the
user's lines, research notes, dossier, earlier rounds) and the round's member
turns, and returns three kinds of label:

- flags: each flaw the engine or Chair raised that round, judged real / false / unsure;
- turns: how much of each member turn is new: mostly new, partly new or repeat;
- missed: clear errors no flag caught, with no cap.

The labels are the reference the local decision models are scored against later
(roadmap: decision models). Uses Claude quota: one call per round. Real sessions
go in, so labels stay in data/bench/labels/ (gitignored). Rows are appended as they
arrive; the same command skips rounds already labelled.

    uv run python -u scripts/label_rounds.py --pick 10 --model fable \\
        --out data/bench/labels/pilot.jsonl
    uv run python -u scripts/label_rounds.py --all --councils career finance --limit 20 \\
        --out data/bench/labels/full.jsonl
    uv run python scripts/label_rounds.py --items SESSION_DIR:ROUND ... --dry-run
    uv run python scripts/label_rounds.py --show data/bench/labels/pilot.jsonl
    uv run python scripts/label_rounds.py --check data/bench/labels/pilot.jsonl > check.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from privy_council.backends.base import (
    BackendRateLimitError,
    CallTag,
    ChatMessage,
    CompletionRequest,
)
from privy_council.backends.registry import BackendRegistry
from privy_council.config import load_settings
from privy_council.engine.budget import Ledger
from privy_council.models.flaw import FIGURE_KINDS, FlawKind
from privy_council.models.persona import BackendKind

# Flaw kinds worth a verdict: the figure checks and the Chair's own; formatting ones aren't.
LABELLED_KINDS = FIGURE_KINDS | {
    FlawKind.ARITHMETIC,
    FlawKind.CONTRADICTION,
    FlawKind.LOGIC,
    FlawKind.OTHER,
}
NOT_MEMBERS = {"system", "user", "chair"}
DOSSIER_CAP = 20_000  # chars of dossier files; research notes come from the events
SINCE = "2026-10-22"  # the flaw checks (M11) exist from this date on
TIMEOUT_S = 900.0
MAX_OUTPUT = 16_000
NOVELTY = ("mostly_new", "partly_new", "repeat")
# Quota guard. Measured 2026-11-10/11: 2.8-4.7 % of the 5-hour window and ~0.6 % of
# the 7-day one per Fable call; start from the worst case, with a margin.
FIRST_STEP_5H = 0.07
FIRST_STEP_7D = 0.01
STEP_MARGIN = 1.5

SYSTEM = """\
You review one round of a debate between AI council members advising a user. \
You are the reference labeller: your labels will be used to score smaller models, \
so be careful and literal, and say "unsure" rather than guess.

You get: the user's question, everything the user said, the research notes and dossier \
the members could see, the members' earlier rounds, this round's member turns, and the \
flags raised on this round by an automatic figure check ("engine") or by the moderator.

Return JSON with three lists.

1. flags: one entry per flag id given. verdict "real" if the flagged passage is a genuine \
error (wrong arithmetic, a figure that is in none of the sources and not presented as an \
estimate, a claim contradicting the sources or the user, a misattribution, a \
self-contradiction, a broken inference). verdict "false" if it is not: e.g. correct \
arithmetic from sourced numbers, a figure that is in the sources, a rounded or converted \
figure, a figure clearly labelled as an estimate, an idiom or figure of speech read \
literally. "unsure" when the sources given don't settle it. reason: one sentence.

2. turns: one entry per member turn of this round. Compare it with every earlier turn \
(any member, any round, including earlier in this round); rewording is not new, and a \
changed position counts as new only with a new reason. novelty: \
"mostly_new" if most of the turn's substance (arguments, facts, numbers, \
recommendations) was not made before; \
"partly_new" if it adds at least one new point but most of the turn repeats earlier turns; \
"repeat" if it adds nothing beyond rewording. what_new: the new point(s) in a few words, \
then what it repeats, if anything.

3. missed: clear errors in this round's member turns that no flag covers. Same standard \
as a "real" flag; no cap, but no style remarks or matters of opinion. quote: copied \
exactly from the turn, a short passage. kind: arithmetic, unsourced, contradiction, \
misattribution, logic or other. note: one sentence on what is wrong.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["real", "false", "unsure"]},
                    "reason": {"type": "string"},
                },
                "required": ["id", "verdict", "reason"],
            },
        },
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "novelty": {"type": "string", "enum": list(NOVELTY)},
                    "what_new": {"type": "string"},
                },
                "required": ["id", "novelty", "what_new"],
            },
        },
        "missed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn": {"type": "integer"},
                    "kind": {"type": "string"},
                    "quote": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["turn", "kind", "quote", "note"],
            },
        },
    },
    "required": ["flags", "turns", "missed"],
}


@dataclass
class Round:
    """One saved round with what the members could see before and during it."""

    session: Path
    round: int
    topic: str
    user_lines: list[str] = field(default_factory=list)
    research: list[str] = field(default_factory=list)
    earlier: list[tuple[int, str, str]] = field(default_factory=list)  # (round, speaker, text)
    turns: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    flags: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False  # a later round has started: research from then on wasn't seen

    @property
    def key(self) -> str:
        """Return the row key: session folder name and round."""
        return f"{self.session.name}:{self.round}"


def load_round(session: Path, round_: int) -> Round:
    """Rebuild one round from a saved session's transcript."""
    meta = json.loads((session / "session.json").read_text())
    r = Round(session=session, round=round_, topic=meta["topic"])
    for line in (session / "transcript.jsonl").read_text().splitlines():
        _add_event(r, json.loads(line))
    return r


def _add_event(r: Round, e: dict[str, Any]) -> None:
    """Sort one transcript event into the round's context, turns or flags."""
    if e["kind"] == "round_started" and e.get("round", 0) > r.round:
        r.ended = True
    elif e["kind"] == "research" and not r.ended:
        r.research.append(e["note"].get("text", ""))
    elif e["kind"] == "flaws" and e.get("round") == r.round:
        r.flags += [f for f in e["flaws"] if f["kind"] in LABELLED_KINDS]
    elif e["kind"] == "message":
        m = e["message"]
        if m["speaker"] == "user" and m["round"] <= r.round:
            r.user_lines.append(m["text"])
        elif m["speaker"] not in NOT_MEMBERS and not m.get("to_user"):
            if m["round"] < r.round:
                r.earlier.append((m["round"], m["speaker"], m["text"]))
            elif m["round"] == r.round:
                r.turns.append((m["speaker"], m["text"]))


def dossier_text(session: Path) -> str:
    """Return the session's dossier files (not the research copy), capped."""
    folder = session / "dossier"
    if not folder.is_dir():
        return ""
    parts = [
        f"## {p.name}\n{p.read_text(errors='replace')}"
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name != "research.md"
    ]
    return "\n\n".join(parts)[:DOSSIER_CAP]


def build_prompt(r: Round) -> str:
    """Return the labelling prompt for one round."""
    research = "\n\n".join(r.research)
    earlier = "\n\n".join(f"[round {n}] <{s}> {t}" for n, s, t in r.earlier)
    turns = "\n\n".join(f"[turn {i}] <{s}> {t}" for i, (s, t) in enumerate(r.turns))
    flags = "\n".join(
        f"[flag {i}] by {f['by']}, on {f['member']}, kind {f['kind']}: "
        f'"{f["excerpt"]}" ({f.get("note", "")})'
        for i, f in enumerate(r.flags)
    )
    sections = [
        f"# Question\n{r.topic}",
        "# What the user said\n" + ("\n".join(f"- {u}" for u in r.user_lines) or "(nothing)"),
        f"# Research notes\n{research or '(none)'}",
        f"# Dossier\n{dossier_text(r.session) or '(none)'}",
        f"# Earlier rounds\n{earlier or '(none: this is round 1)'}",
        f"# This round (round {r.round})\n{turns}",
        f"# Flags on this round\n{flags or '(none)'}",
    ]
    return "\n\n".join(sections)


def council(r: Round) -> str:
    """Return the council of a round, from its session folder name."""
    return r.session.name.split("_")[1]


def candidate_rounds(sessions_dir: Path, since: str = SINCE) -> list[Round]:
    """Return every round with member turns, from sessions since a date, newest first."""
    out: list[Round] = []
    for s in sorted(sessions_dir.iterdir(), reverse=True):
        if s.is_symlink() or not (s / "transcript.jsonl").is_file() or s.name[:10] < since:
            continue
        rounds = {
            json.loads(line)["message"]["round"]
            for line in (s / "transcript.jsonl").read_text().splitlines()
            if '"kind":"message"' in line
        }
        out += [load_round(s, n) for n in sorted(rounds) if n >= 1]
    return [r for r in out if r.turns]


def pick(rounds: list[Round], n: int, flagged: bool = True) -> list[Round]:
    """Pick n round 2+ rounds (only flagged ones if flagged), one per session in turn."""
    good = [r for r in rounds if r.round >= 2 and (r.flags or not flagged)]
    by_session: dict[Path, list[Round]] = defaultdict(list)
    for r in good:
        by_session[r.session].append(r)
    chosen: list[Round] = []
    while len(chosen) < n and any(by_session.values()):
        for rs in by_session.values():
            if rs and len(chosen) < n:
                chosen.append(rs.pop(len(rs) // 2))
    return chosen


def check_quotes(r: Round, missed: list[dict[str, Any]]) -> None:
    """Mark each missed error whose quote isn't found word for word in its turn."""
    for m in missed:
        i = m.get("turn", -1)
        text = r.turns[i][1] if 0 <= i < len(r.turns) else ""
        m["speaker"] = r.turns[i][0] if text else "?"
        m["quote_found"] = " ".join(m["quote"].split()) in " ".join(text.split())


async def label(registry: BackendRegistry, model: str, r: Round) -> dict[str, Any]:
    """Run one labelling call and return the row to save."""
    req = CompletionRequest(
        system=SYSTEM,
        messages=[ChatMessage(role="user", content=build_prompt(r))],
        model=model,
        max_output_tokens=MAX_OUTPUT,
        json_schema=SCHEMA,
        timeout_s=TIMEOUT_S,
        tag=CallTag(session=r.session.name, purpose="label"),
    )
    res = await registry.backend(BackendKind.CLAUDE_CLI).complete(req)
    out = res.structured or json.loads(res.text)
    check_quotes(r, out["missed"])
    return {
        "key": r.key,
        "model": res.model,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latency_s": round(res.latency_s, 1),
        "out_tokens": res.usage.output_tokens if res.usage else None,
        "flags_in": r.flags,
        "turn_speakers": [s for s, _ in r.turns],
        **out,
    }


def last_quota(ledger: Path) -> dict[str, Any]:
    """Return the quota of the last ledger row ({} if none)."""
    rows = ledger.read_text().splitlines() if ledger.exists() else []
    quota: dict[str, Any] = (json.loads(rows[-1]).get("quota") or {}) if rows else {}
    return quota


def quota_stop(
    quota: dict[str, Any], step: tuple[float, float], caps: tuple[float, float]
) -> str | None:
    """Return why not to make the next call, or None to go on.

    step is the projected cost of the next call (5-hour, 7-day share): the most a call
    has cost so far, with a margin, since rounds vary 2-3x in size and other sessions
    draw on the same quota meanwhile.
    """
    if not quota:
        return None
    # "allowed_warning" only says a window passed ~90 %: the caps below decide.
    if quota.get("status") not in (None, "allowed", "allowed_warning"):
        return f"quota status {quota['status']}"
    for name, used, add, cap in (
        ("5-hour", quota.get("five_hour") or 0, step[0], caps[0]),
        ("7-day", quota.get("seven_day") or 0, step[1], caps[1]),
    ):
        if used + add > cap:
            return (
                f"{name} window at {used:.0%}, next call may reach {used + add:.0%} (cap {cap:.0%})"
            )
    return None


def next_step(
    step: tuple[float, float], before: dict[str, Any], after: dict[str, Any]
) -> tuple[float, float]:
    """Return the projected next-call cost: the largest seen so far, times STEP_MARGIN."""
    if not before or not after:
        return step
    seen = [
        STEP_MARGIN * ((after.get(k) or 0) - (before.get(k) or 0))
        for k in ("five_hour", "seven_day")
    ]
    return (max(step[0], seen[0]), max(step[1], seen[1]))


async def run(args: argparse.Namespace, rounds: list[Round]) -> None:
    """Label each round not yet in --out, appending rows as they arrive."""
    settings = load_settings()
    done = {row["key"] for row in load_rows(args.out)} if args.out.exists() else set()
    todo = [r for r in rounds if r.key not in done]
    left = len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} round(s) to label now ({left} left, {len(rounds) - left} done)", flush=True)
    registry = BackendRegistry(settings, Ledger.in_data_dir(settings.data_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger = settings.data_dir / "usage.jsonl"
        step = (FIRST_STEP_5H, FIRST_STEP_7D)
        for i, r in enumerate(todo, 1):
            before = last_quota(ledger)
            why = quota_stop(before, step, (args.max_5h, args.max_7d))
            if why:
                print(f"stopped before {r.key}: {why}; rerun the same command to go on", flush=True)
                return
            try:
                row = await label(registry, args.model, r)
            except BackendRateLimitError as exc:
                print(f"stopped before {r.key}: usage limit ({exc})", flush=True)
                return
            with args.out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(todo)}] {brief(row)}", flush=True)
            step = next_step(step, before, last_quota(ledger))
    finally:
        await registry.aclose()


def novelty(turn: dict[str, Any]) -> str:
    """Return a turn's novelty label; pilot rows have a bool `new` instead."""
    if "novelty" in turn:
        return str(turn["novelty"])
    return "new" if turn["new"] else "repeat"


def brief(row: dict[str, Any]) -> str:
    """Return a one-line summary of a labelled round."""
    verdicts = [f["verdict"] for f in row["flags"]]
    counts = ", ".join(f"{v} {verdicts.count(v)}" for v in ("real", "false", "unsure"))
    labels = [novelty(t) for t in row["turns"]]
    turns = ", ".join(f"{v} {labels.count(v)}" for v in dict.fromkeys(labels))
    bad = sum(not m["quote_found"] for m in row["missed"])
    return (
        f"{row['key']}: {row['latency_s']} s; flags {len(verdicts)} ({counts}); "
        f"turns {len(labels)} ({turns}); missed {len(row['missed'])}"
        + (f" ({bad} quote(s) not found)" if bad else "")
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Return the rows of a labels file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def show(path: Path) -> None:
    """Print a labels file as Markdown, for a spot check by hand."""
    for row in load_rows(path):
        print(f"\n# {row['key']}  ({brief(row)})\n")
        for f in row["flags"]:
            src = row["flags_in"][f["id"]] if f["id"] < len(row["flags_in"]) else {}
            print(
                f"- flag {f['id']} {src.get('by')}/{src.get('kind')} on {src.get('member')} "
                f'"{src.get("excerpt", "")[:120]}" → **{f["verdict"]}**: {f["reason"]}'
            )
        for t in row["turns"]:
            who = row["turn_speakers"][t["id"]] if t["id"] < len(row["turn_speakers"]) else "?"
            print(f"- turn {t['id']} {who}: {novelty(t)}: {t['what_new']}")
        for m in row["missed"]:
            mark = "" if m["quote_found"] else " (QUOTE NOT FOUND)"
            print(f'- missed {m["speaker"]} [{m["kind"]}] "{m["quote"]}"{mark}: {m["note"]}')


CHECK_HEADER = """\
# Spot check

After each `you:` put **y** (Fable is right), **n** (Fable is wrong) or **?** (can't tell).
Add a few words only after **n**, or when something is worth noting. Blank = skipped.
"""


def around(text: str, quote: str, n: int = 350) -> str:
    """Return the quote with about n chars of its turn on each side."""
    t, q = " ".join(text.split()), " ".join(quote.split()).strip("…")
    i = t.find(q[:60])
    if i < 0:
        return t[: 2 * n] + "…"
    start, end = max(0, i - n), i + len(q) + n
    return ("…" if start else "") + t[start:end] + ("…" if end < len(t) else "")


def check_items(row: dict[str, Any], r: Round) -> dict[str, list[list[str]]]:
    """Return one round's check items, by section, each as Markdown lines."""
    out: dict[str, list[list[str]]] = defaultdict(list)
    where = f"{r.session.name[:10]} {r.session.name.split('_')[1]} r{r.round}"
    for f in row["flags"]:
        src = row["flags_in"][f["id"]]
        text = next((t for s, t in r.turns if s == src["member"]), "")
        note = f" ({src['note']})" if src.get("note") else ""
        out[f["verdict"]].append(
            [
                f"{where} · {src['member']} · {src['by']}/{src['kind']}",
                f'- **Flagged**: "{src["excerpt"]}"{note}',
                f"- **Context**: {around(text, src['excerpt'])}",
                f"- **Fable**: {f['verdict']}: {f['reason']}",
            ]
        )
    for t in row.get("turns", []):
        speaker, text = r.turns[t["id"]] if 0 <= t["id"] < len(r.turns) else ("?", "")
        out["turns"].append(
            [
                f"{where} · {speaker} · novelty",
                f"- **Turn**: {' '.join(text.split())[:1500]}",
                f"- **Fable**: {novelty(t)}: {t['what_new']}",
            ]
        )
    for m in row["missed"]:
        text = r.turns[m["turn"]][1] if 0 <= m["turn"] < len(r.turns) else ""
        out["missed"].append(
            [
                f"{where} · {m['speaker']} · {m['kind']}",
                f'- **Quote**: "{m["quote"]}"',
                f"- **Context**: {around(text, m['quote'])}",
                f"- **Fable**: {m['note']}",
            ]
        )
    return out


def check_sheet(path: Path, sessions: Path, sample: int = 0) -> str:
    """Return a Markdown sheet to check a labels file by hand: one y/n line per label.

    sample > 0 keeps about that many items, spread over the sections (fixed seed).
    """
    sections = {
        "false": "A. Flags Fable called FALSE",
        "real": "B. Flags Fable called REAL",
        "unsure": "C. Flags Fable was unsure about",
        "missed": "D. Errors Fable found that no flag caught",
        "turns": "E. How new each turn is (mostly_new / partly_new / repeat)",
    }
    items: dict[str, list[list[str]]] = defaultdict(list)
    for row in load_rows(path):
        name, n = row["key"].rsplit(":", 1)
        for key, found in check_items(row, load_round(sessions / name, int(n))).items():
            items[key] += found
    if sample:
        total = sum(len(v) for v in items.values())
        rng = random.Random(0)
        for key, found in items.items():
            keep = max(1, round(sample * len(found) / total))
            items[key] = rng.sample(found, min(keep, len(found)))
    lines, k = [CHECK_HEADER], 0
    for key, title in sections.items():
        if items[key]:
            lines.append(f"\n## {title}\n")
        for head, *body in items[key]:
            k += 1
            lines += [f"### {k}. {head}", *body, "- you: \n"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description="Label saved debate rounds with a strong Claude model.")
    p.add_argument("--items", nargs="*", default=[], help="SESSION_DIR:ROUND ...")
    p.add_argument("--pick", type=int, default=0, help="pick N flagged rounds from data/sessions")
    p.add_argument("--all", action="store_true", help="every round 2+, flagged or not")
    p.add_argument("--councils", nargs="*", default=[], help="only these councils (--pick/--all)")
    p.add_argument("--since", default=SINCE, help="--pick/--all: sessions from this date on")
    p.add_argument(
        "--max-5h",
        type=float,
        default=0.85,
        help="no call that may take the 5-hour window past this",
    )
    p.add_argument(
        "--max-7d", type=float, default=0.6, help="no call that may take the 7-day window past this"
    )
    p.add_argument("--limit", type=int, default=0, help="label at most N rounds not yet done")
    p.add_argument("--model", default="fable")
    p.add_argument("--out", type=Path, default=Path("data/bench/labels/labels.jsonl"))
    p.add_argument("--dry-run", action="store_true", help="list rounds and prompt sizes")
    p.add_argument("--show", type=Path, help="print a labels file as Markdown")
    p.add_argument("--check", type=Path, help="print a y/n sheet to check a labels file")
    p.add_argument("--sample", type=int, default=0, help="--check: keep about N items")
    return p.parse_args()


def main() -> None:
    """Pick or load the rounds, then label them (or show / dry-run)."""
    args = parse_args()
    if args.show:
        show(args.show)
        return
    if args.check:
        print(check_sheet(args.check, load_settings().data_dir / "sessions", args.sample))
        return
    rounds = [load_round(Path(s), int(n)) for s, n in (i.rsplit(":", 1) for i in args.items)]
    if args.pick or args.all:
        found = candidate_rounds(load_settings().data_dir / "sessions", args.since)
        found = [r for r in found if not args.councils or council(r) in args.councils]
        rounds += pick(found, 10**6 if args.all else args.pick, flagged=not args.all)
    if args.dry_run:
        for r in rounds:
            size = len(build_prompt(r)) // 4
            print(f"{r.key}: {len(r.turns)} turns, {len(r.flags)} flags, ~{size} tok")
        return
    asyncio.run(run(args, rounds))


if __name__ == "__main__":
    main()
