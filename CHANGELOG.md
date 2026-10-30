# Changelog

All notable changes, by milestone. Dates are when each milestone was built.

## Unreleased (2026-10-25 – 2026-10-30)

- **Discord front end** (`council chat discord`, `doctor --chat`): friends allowlisted by user ID debate with a local-only council in a forum thread; one webhook per persona with its avatar; a fixed command table (`/ask`, `/status`, `/flaws`, …); `/offline` and `/resume`; message size and per-round caps; no pings.
- **Local web search** (opt-in `[research.local]`): SearXNG in Podman, started and stopped with the session; snippets only; the Chair's summary is checked against the hits (figures and URLs); a follow-up search for missing facts. `[research] provider = "claude" | "local" | "auto"` for normal sessions.
- **Fewer flaws**: a "wrong unit" check (a number the sources give only with another unit), numbers spelled out as words, a member's own written-out arithmetic no longer flagged as unsourced, moderator flags that confirm the quote or that a member already answered are dropped.
- **Questions for the user**: repeated questions are dropped, at most 2 per person in shared chats, the synthesis doesn't re-ask answered ones.
- Councils can add context for the Chair (`moderator_brief`, `research_focus`) in their YAML.
- Ollama `num_batch` per model.
- `council personas new` refuses a drafted slug that would write outside `data/`.
- Discord `/handover @friend` and `/takeback`: the owner lets a friend on the list run, pause, vote, close and take the bot offline for the open debate only (never `/ask` or `/resume`).
- Research notes name the pages the Chair read (WebFetch), in the chat notice, `research.md` and exports; a fetch-only lookup no longer says "answered without searching the web".
- Seats no longer stay on "thinking…" while memory is written at close or during `/vote`; an Ollama CUDA out-of-memory error is retried once.
- Research: later lookups are told which pages earlier ones read and not to fetch them again (a page cut off before the needed part comes back cut off the same way); the Chair skips lookups an earlier note says failed unless it names another source; each page is listed once.
- Pragmatist and Fan move from gpt-oss:120b to Meta's `muse-glimmer:30b` (`ollama pull muse-glimmer:30b`): in a replay bench it invented no figures where gpt-oss invented several, and at 18 GB it stays loaded beside the other local models. gpt-oss:120b is now only the local-only default seat.
- A flaw check's floor line no longer repeats the flaw: the Chair's planned prompt loses its self-address and the sentences restating the error.
- New `economics` council for economic and market forecasts (a state's debt, company defaults, layoffs, crashes): Forecaster, Insider, Contrarian and two new personas, Macro (nemotron-3.5-super) and Analyst (muse-glimmer:30b); its brief has Chair pin down what counts as the outcome. `finance` is described as personal finance.

## M15 — local-only mode (2026-10-24)

- `--local-only`: every Claude seat (Chair included) moves to a local model by a seat map; a hard guard keeps the session from reaching Claude; sticky on resume.

## M14 — comparing council compositions (2026-10-23)

- `council compare new/resume/report/list`: run variants of a council on the same question with shared research, fixed rounds and a final vote; optional judge; Markdown and HTML report.

## M13 — export and persona generator (2026-10-23)

- `council sessions export` (Markdown, HTML) and `/export`; member models recorded per session.
- `council personas new "<description>"`: draft a persona with Claude, validated and confirmed before it's written.

## M11–M12 — fewer flaws (2026-10-20 – 2026-10-22)

- Engine checks on every member turn: unsourced or mislabelled figures, formatting slips, @names of absent people. Flag only: the text is never changed.
- The moderator reports flaws in the call it already makes; up to 2 per round become floor prompts ("check it"); an "Errors caught" section in the synthesis; `/flaws`.
- `scripts/flaw_bench.py` to score models and prompts on recorded turns.

## Councils and models (2026-10-18 – 2026-10-20)

- New councils: `geopolitics`, `politics`, `history`, `purchases`, with their personas; research required where facts matter.
- Local models: qwen3.6, gemma4, gpt-oss and nemotron-3.5-super seats, chosen by replays and benches (see `docs/spike-notes.md`).

## M8–M10 — memory option, salon, French (2026-10-17 – 2026-10-18)

- `memory: on | read_only | off` per council; a `wellbeing` council.
- `salon`: a philosophers' council (Socrates, Plato, Nietzsche, Epicurus).
- French debates: language detected from the question (`--lang`, `/lang`), French labels and headings.

## M6–M7 — robustness and a dynamic moderator (2026-10-17)

- Resume (`council resume`, `council sessions`), `/summon` and `/kick`, streaming, `/vote` and `/summary`, `/attach` (a dossier of files), `num_ctx` 12k.
- The moderator sees engine-counted dynamics (quiet members, repetition), detects loops, may suggest an early stop (overridable), finds consensus and proposes `/summon`.

## M0–M5 — foundations (2026-10-15 – 2026-10-16)

- Spike of the `claude` CLI flags (`docs/spike-notes.md`).
- Personas as Markdown with YAML frontmatter, councils as YAML.
- Backends: `claude -p` subprocess (stdin, API keys stripped, streamed JSON, quota from the CLI) and Ollama; `council doctor`; the usage ledger.
- Debate engine: opening, blind first round, rebuttals, synthesis; the moderator's strict JSON with repair and round-robin fallback; a plain-text REPL and a Textual TUI.
- Web research by the Chair (`/research`, `--research`).
- Memory: private notes per persona and shared user facts with approval; `/forget`, `council memory reset`.
