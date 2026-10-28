# Project Brief — `council`

> Working name. A terminal-based "AI Council": a group of LLM personas you consult on personal topics (finance, career, trips, decisions). They give independent opinions, then confront each other in an IRC-style chat until they reach consensus or a clearly mapped disagreement.

---

## 1. Vision

A single-user, local-first CLI/TUI app where I open a "channel" on a topic. I summon a few personas, each defined by a Markdown file, and watch them debate. A moderator runs the discussion. I can step through it round by round or let it run on its own. Every session ends with a structured synthesis: consensus, open disagreements, and questions only I can answer.

Personas remember past sessions, so a council I consult repeatedly gets better context over time. Memory can be reset at any time.

## 2. Goals

- **G1 — Real diversity of opinion.** Avoid "five costumes on one model" convergence. Use a blind first round, structural roles, private memory, and a mix of models.
- **G2 — Zero marginal cost.** Run only on:
  - the Claude Code Pro subscription, through the `claude` CLI in headless mode;
  - local models through Ollama.
  
  **No Anthropic or OpenAI API keys, ever.**
- **G3 — Controllable debates.** Step one round at a time, run N rounds autonomously, pause, or interject.
- **G4 — Persistent per-persona memory.** Memory is resettable per persona or globally.
- **G5 — Plain-text everything.** Personas, memory, config and transcripts are human-readable files I can edit, version or delete.
- **G6 — Fun to use.** The IRC aesthetic is part of the point: colored nicks, channels, slash commands.

## 3. Non-goals (for now)

- Multi-user use, web UI, or hosting.
- Paid API backends. The backend abstraction must not *prevent* adding them later, but none ship.
- Autonomous actions in the real world: no trading, no emails, no bookings. The council only advises.
- Fine-tuning, embeddings or vector DB. Memory is distilled Markdown, which is enough at this scale.
- Live web browsing by personas in the MVP (see §15, future).

## 4. Hard constraints

| Constraint | Consequence |
|---|---|
| Claude only through the Claude Code Pro subscription | Call `claude -p` as a subprocess. Treat Claude calls as a **scarce, rate-limited budget** and track them. |
| GPU: RTX 4070 Ti, **12 GB VRAM** | Small local models only, from about 4B to 12B at Q4. Realistically **one model loaded at a time**, because model swaps cost seconds. Small context windows. |
| Python, single user, Linux desktop | Python 3.12+, `uv`, async (`asyncio`). |
| Personal data (finances, career) | Everything stays local except the prompts sent to Claude through the CLI. No telemetry. Memory and sessions are gitignored. |

## 5. Core concepts (glossary)

- **Persona** — An actor in the debate, defined by a Markdown file with YAML frontmatter. The frontmatter holds metadata and the backend/model; the body is the system prompt (personality, expertise, biases, style).
- **Role** — A structural mandate a persona can hold, independent of its personality:
  - `advisor` (default);
  - `devil_advocate` (must challenge the leading position);
  - `risk_officer` (must name the failure modes);
  - `moderator` (runs the debate and never gives an opinion);
  - `scribe` (writes the synthesis; may be the same as the moderator).
- **Council** — A named, reusable preset: a list of personas plus a moderator plus default protocol settings. Examples: `finance`, `career`, `travel`.
- **Channel / Session** — One concrete discussion on one topic with one council. It has a transcript, a state, and can be resumed.
- **Dossier** — Files attached to a session as shared context (a CSV of expenses, a job offer, a trip constraints list).
- **Round** — One phase of the protocol in which some or all personas speak once.
- **Debate state** — A compact, moderator-maintained summary of the debate so far:
  - positions per persona;
  - points of agreement;
  - points of disagreement;
  - open questions for the user.
  
  This is what small models see, instead of the full transcript.
- **Memory** — Distilled notes persisted across sessions:
  - **shared**: facts about the user, visible to everyone;
  - **private**: per persona, holding its own past positions and lessons.

## 6. User stories

1. As the user, I run `council new finance "Should I max my PEA before building a bigger emergency fund?"` and get a channel with the finance council already summoned.
2. I attach a file (`/attach budget.csv`) and every persona sees it in its context.
3. I type `/step` and watch round 1: each persona answers independently and blind.
4. I type `/run 3` and watch the cross-examination unfold without me. I press a key to pause if something interesting happens.
5. I interject (`/say I forgot: I have a mortgage ending in 2028`). The moderator integrates it and the next round reacts to it.
6. I address one persona directly (`/ask @Rigor what's your worst case?`).
7. I end the session (`/close`). I get a synthesis, and each persona saves distilled memories.
8. Three months later I open a new finance session. The personas remember my earlier decisions and their own past arguments.
9. I run `/forget @Optimist` or `council memory reset --all` when I want a clean slate.
10. I create a new persona by copying a Markdown file and editing it. No code change is needed.
11. I check `/budget` to see how many Claude calls this session used, and the moderator warns me before an expensive `/run`.

## 7. Persona file format

Location: `personas/<slug>.md`

```markdown
---
name: Rigor                 # display nick (IRC style, short, unique)
slug: rigor                 # must match the filename
color: "#e5c07b"            # nick color in the TUI
role: risk_officer          # advisor | devil_advocate | risk_officer | moderator | scribe
backend: ollama             # ollama | claude_cli
model: qwen3:8b             # backend-specific model id
temperature: 0.6
max_output_tokens: 400      # keeps turns short, IRC-like
expertise: [personal finance, French tax wrappers (PEA, assurance-vie), risk management]
memory: true                # keep private memory across sessions
tags: [finance]
---

# Who you are
You are Rigor, a former credit risk analyst. ...

# How you think
- You always ask "what happens if this goes wrong?"
- You distrust projections that don't state their assumptions.

# Biases you own
- You overweight downside scenarios. Say so when relevant.

# How you speak
Terse, precise, a bit dry. Never more than ~5 sentences per turn unless asked.
```

Rules:
- The body is free Markdown and becomes part of the system prompt.
- The **engine** adds its own instructions around it: protocol rules, round instructions, debate state, memory and dossier. Persona files never need to know the protocol.
- Validate with Pydantic at load time. On errors, give clear messages with file and field.

### Council preset format

Location: `councils/<name>.yaml`

```yaml
name: finance
description: Personal finance decisions
moderator: chair            # persona slug with role: moderator
members: [rigor, compounder, pragmatist, contrarian]
protocol:
  blind_first_round: true
  max_rounds: 6
  auto_stop_on_consensus: true
  rebuttal_style: targeted   # targeted (moderator picks pairs) | round_robin
```

## 8. Debate protocol

### Phases

1. **Opening (moderator)**
   - Restates the question.
   - Lists the key sub-questions.
   - States what a good answer must cover.
2. **Round 1 — Blind takes.** Every member answers **independently**, without seeing the others' answers.
   - These calls may run concurrently: in parallel for Claude, sequentially for Ollama on a single GPU.
   - Required output shape: position, top 3 reasons, confidence (0–100), key assumption.
3. **Moderator update.**
   - Builds the debate state from the round 1 answers.
   - Identifies the 1–3 sharpest disagreements.
4. **Rounds 2..N — Cross-examination.** The moderator chooses who speaks and on what. For example: "@Rigor, respond to @Compounder's claim that X". Each turn:
   - sees the debate state, the last K messages and the specific prompt from the moderator;
   - may change position, and must say so explicitly when it does ("I update: …").
   
   After each round, the moderator updates the debate state.
5. **Stop conditions.** Any of these ends the cross-examination:
   - `max_rounds` reached;
   - the user's `/run N` budget is exhausted (in that case: pause, don't close);
   - the moderator judges consensus reached, or the debate looping with no new arguments in the last round;
   - the Claude call budget for the session is nearly exhausted (warn first).
6. **Synthesis (moderator/scribe).** Contents:
   - consensus points;
   - remaining disagreements, each with who holds which side and why;
   - confidence spread;
   - **questions for the user**: missing info that would change the answer;
   - suggested next steps.
7. **Memory write** (on `/close`, see §10).

### Moderator decisions are structured output

The moderator returns JSON validated by Pydantic, for example:

```json
{
  "debate_state": { "positions": {...}, "agreements": [...], "disagreements": [...], "open_questions": [...] },
  "next_turns": [ {"speaker": "rigor", "prompt": "Respond to compounder on ..."} ],
  "stop": false,
  "stop_reason": null
}
```

Requirements:
- Parse strictly.
- On a parse failure, retry once with a "fix your JSON" message.
- If that still fails, fall back to a round-robin turn order and log a warning.
- Never crash the session.

### Anti-convergence measures (built in)

- The blind first round is on by default.
- A `devil_advocate` must argue against the current leading position, even if it privately agrees. It flags when it is doing so.
- Personas are instructed **not** to agree just to be polite. Agreement must cite a reason that is new or reinforced.
- Private memory differs per persona.
- Members use mixed backends and models.

## 9. Backends

A common async interface:

```python
class Backend(Protocol):
    name: str
    async def complete(self, req: CompletionRequest) -> CompletionResult: ...
    async def health(self) -> BackendHealth: ...
```

`CompletionRequest` holds:
- system prompt;
- messages;
- model;
- temperature;
- max output tokens;
- an optional JSON-mode flag;
- a timeout.

`CompletionResult` holds:
- text;
- token usage, if known;
- latency;
- raw metadata;
- the backend name.

### 9.1 `claude_cli` backend (Claude Code subscription)

- Implementation: `asyncio.create_subprocess_exec("claude", "-p", ..., "--output-format", "json", ...)`, **never** an HTTP API call or an API key.
- **Verify the exact CLI flags in Milestone 0** against `claude --help` on the installed version; they evolve. The intent is:
  - a non-interactive single response (print mode);
  - JSON output to parse the result and usage;
  - a custom system prompt that replaces the Claude Code default, if supported, otherwise appended to it;
  - model selection (e.g. `sonnet`, `haiku`, `opus` aliases);
  - no tools, a single turn, no session persistence, if supported.
- Run the subprocess with `cwd` set to an **empty, dedicated temp dir**, so that no project CLAUDE.md or files leak into persona context.
- Pass the prompt through stdin rather than argv, to avoid argv length limits and to keep personal data out of `ps` output.
- Ensure `ANTHROPIC_API_KEY` is **unset** in the subprocess env, so that the CLI uses the subscription login and never bills API credits. Log a loud warning at startup if it is set in the parent env.
- Detect and surface errors cleanly:
  - not logged in;
  - rate limit / usage limit reached (parse the message and show when to retry if the CLI provides it);
  - timeout.
- The engine must degrade gracefully when Claude is exhausted. Options:
  - pause the session with a clear message;
  - or, if the user enabled it in config, re-route Claude personas to a fallback local model, marking them `[fallback]` in the chat.
- **Budget tracking:**
  - count calls and token usage per session and per day in a local ledger (`data/usage.jsonl`);
  - make a configurable soft cap per session warn before `/run N` if the estimated calls exceed the remaining budget.

### 9.2 `ollama` backend

- Talks to Ollama's HTTP API (`http://localhost:11434`, configurable) with `httpx.AsyncClient` (`/api/chat`, non-streaming first; streaming later for the TUI typing effect).
- Pass `options.temperature`, `options.num_ctx` (configurable per model) and `keep_alive`.
- **GPU-aware scheduling:**
  - ordered by model, to minimize swaps within a round, while keeping the displayed order sensible;
  - a global semaphore with concurrency 1 by default for Ollama calls.
- Health check at startup: is the server reachable, and are the configured models pulled? Print the `ollama pull ...` hint if not.
- Context budget: estimate tokens (a rough chars/4 heuristic is fine) and trim the context to fit `num_ctx` minus `max_output_tokens`.

### Recommended defaults (12 GB VRAM)

These are suggestions to verify in M0, not hard requirements:
- A single "workhorse" local model for most local personas, around 8B (e.g. a Qwen3 8B or Llama 3.1 8B class model) with Q4 quantization.
- Optionally one ~12B model, if the swap cost is acceptable.
- Claude reserved for the **moderator** and at most one or two "heavy reasoning" personas.

## 10. Memory

### Layout

```
data/                         # gitignored
  memory/
    shared/
      user.md                 # facts about the user, shared by all personas
    personas/
      <slug>/
        notes.md              # distilled private memory
  sessions/
    <yyyy-mm-dd>_<council>_<short-topic-slug>/
      session.json            # metadata, council, state, protocol settings, budget used
      transcript.jsonl        # every event (message, round marker, user interjection, error)
      transcript.md           # human-readable render
      debate_state.json       # latest moderator state
      synthesis.md            # final synthesis
      dossier/                # copies of attached files
  usage.jsonl                 # Claude call ledger
```

### Writing memory (on `/close`)

1. The moderator/scribe proposes **shared** updates: new stable facts about the user that were stated **by the user**, never inferred.
   - They are shown to the user for approval, accept/reject per line.
   - Personas may not write shared memory directly.
2. Each persona with `memory: true` writes **private** notes. Examples:
   - "Session 2026-10-15 (finance): Thomas decided X. I argued Y; he disagreed because Z. Lesson: …"
3. Notes are short and dated, as Markdown bullets. They are capped in size per file, configurable at around 4 KB by default.
4. **Consolidation.** When a file exceeds the cap, run a consolidation pass using the persona's own backend: merge, drop stale points, keep decisions and preferences.

### Reading memory

At each turn, the persona's context includes:
- `shared/user.md`;
- its own `notes.md`;
- both truncated to a token budget, with private memory getting priority on small models.

### Reset

- `/forget @persona`: wipe that persona's private notes, after confirmation.
- `/forget shared`: wipe shared user memory, after confirmation.
- `/forget all`: everything, after confirmation.
- CLI: `council memory reset [--persona SLUG | --shared | --all]`.
- Manual: delete `data/memory/...`. Document this in the README.
- Sessions are kept independently; `council sessions prune` can delete old ones.

## 11. Interface

### CLI entry points (Typer)

```
council new <council> "<topic>" [--attach FILE ...]   # start a session and open the TUI
council resume [<session-id>]                         # resume the last or a given session
council sessions list | show <id> | prune --older-than 90d
council personas list | show <slug> | validate
council councils list | show <name>
council memory show [--persona SLUG] | reset [...]
council doctor                                        # check claude CLI login, ollama, models, config
council budget                                        # Claude usage today / this week
```

### TUI (Textual), IRC-style

- **Top bar:** channel name (`#finance`), topic, round counter, state (`paused` / `running 2/3`), Claude budget indicator.
- **Main pane:** the scrolling chat log. Lines look like `[21:48] <Rigor> ...`, with colored nicks, system lines in grey (`-- round 2 begins --`, `* Chair asks Rigor to respond to Compounder`), and Markdown rendering.
- **Right pane:** the nick list with role badges and backend/model (`@Chair` for the moderator, `+` for Claude-backed personas), and status (`thinking…`, `idle`, `fallback`).
- **Collapsible side pane:** the current debate state (agreements, disagreements, open questions).
- **Input line:** slash commands, with plain text treated as `/say`.

### Slash commands

| Command | Effect |
|---|---|
| `/step` | Run the next round, then pause |
| `/run N` | Run up to N rounds autonomously (shows the estimated Claude calls first if over a threshold) |
| `/pause` (or `Esc`) | Pause at the next turn boundary; never cut a call mid-way |
| `/say <text>` or plain text | User interjection; the moderator integrates it in the next round |
| `/ask @nick <text>` | Direct question to one persona; out-of-round, a single turn |
| `/summon <slug>` | Add a persona to the channel; it gets a catch-up on the debate state |
| `/kick <nick>` | Remove a persona |
| `/attach <path>` | Add a file to the dossier |
| `/state` | Print the debate state |
| `/vote` | Each persona gives a final position plus a confidence; shown as a table |
| `/summary` | Generate an interim synthesis without closing |
| `/close` | Final synthesis, memory write (with approval for shared), save |
| `/forget ...` | See §10 |
| `/budget` | Claude calls and tokens used this session and today |
| `/help` | Command list |
| `/quit` | Save state and exit without the synthesis (resumable) |

## 12. Configuration

`config.toml`, plus an optional `config.local.toml` override that is gitignored.

```toml
[paths]
personas = "personas"
councils = "councils"
data = "data"

[claude]
binary = "claude"
default_model = "sonnet"
timeout_s = 180
session_soft_cap_calls = 40
daily_soft_cap_calls = 150
fallback_to_local = false
fallback_model = "qwen3:8b"

[ollama]
base_url = "http://localhost:11434"
timeout_s = 300
concurrency = 1
keep_alive = "10m"

[ollama.models."qwen3:8b"]
num_ctx = 8192

[engine]
context_last_k_messages = 6
max_turn_chars_display = 2000

[memory]
private_cap_bytes = 4096
shared_cap_bytes = 4096
require_shared_approval = true
```

## 13. Architecture (modules)

```
src/privy_council/
  __init__.py
  cli.py                 # Typer app (entry point)
  config.py              # load/merge TOML -> Pydantic settings
  models/                # Pydantic domain models
    persona.py           # Persona, Role, frontmatter schema
    council.py           # CouncilPreset, ProtocolSettings
    session.py           # Session, Message, Event, DebateState
    moderator.py         # ModeratorDecision (structured output schema)
  loaders/
    personas.py          # parse .md + frontmatter, validate
    councils.py
  backends/
    base.py              # Backend protocol, CompletionRequest/Result, errors
    claude_cli.py
    ollama.py
    fake.py              # deterministic scripted backend for tests/demo
    registry.py          # resolves backend per persona, fallback logic
  engine/
    state_machine.py     # session states & transitions (see below)
    protocol.py          # phases: opening, blind round, cross-exam, synthesis
    prompts.py           # ALL engine prompt templates (Jinja2 or str.format), versioned
    context.py           # builds per-turn context with token budgeting
    scheduler.py         # GPU-aware ordering, semaphores, concurrency
    budget.py            # Claude usage ledger & estimates
  memory/
    store.py             # read/write/reset files
    distill.py           # end-of-session note writing & consolidation
  storage/
    sessions.py          # persist/resume sessions (json/jsonl/md)
  tui/
    app.py               # Textual App
    widgets.py           # chat log, nick list, state panel, input
    commands.py          # slash-command parser -> engine actions
  util/
    tokens.py            # rough token estimation
    time.py
personas/                # example personas (committed)
councils/                # example council presets (committed)
tests/
```

### Engine state machine

States:
- `CREATED`
- `OPENING`
- `ROUND_RUNNING`
- `PAUSED`
- `SYNTHESIZING`
- `CLOSING`
- `CLOSED`
- `ERROR_RECOVERABLE`

Rules:
- The engine is **UI-agnostic**. It emits events (`MessageEvent`, `RoundStarted`, `StateUpdated`, `BudgetWarning`, `ErrorEvent`, …) on an async queue. The TUI subscribes; a plain-text CLI renderer can subscribe too (used for tests and the `--no-tui` mode).
- Commands are delivered to the engine via methods or a command queue. The UI never touches internals.
- Pause is cooperative: it is checked at turn boundaries.
- Every event is appended to `transcript.jsonl` **as it happens**, so a crash loses at most one in-flight turn. `resume` rebuilds the state from the files.

## 14. Milestones and acceptance criteria

**M0 — Spike & verification (do first, small).**
- Verify the `claude` CLI headless flags on the installed version: system prompt override, JSON output, model alias, tool disabling, single turn, stdin prompt.
- Confirm that it uses the subscription (no API key in env) and see what the rate-limit error looks like.
- Verify the Ollama `/api/chat` call with a chosen model, and measure the model swap time.
- Record findings in `docs/spike-notes.md`.
- ✅ Two tiny scripts that each get a reply, plus notes.

**M1 — Skeleton & domain.**
- Repo scaffolding (`uv`, ruff, mypy, pytest).
- Pydantic models.
- Persona and council loaders with validation.
- `council personas validate`.
- 4 example personas plus 1 moderator; 1 example council.
- ✅ `uv run pytest` green; `council personas list` works.

**M2 — Backends.**
- `ollama`, `claude_cli` and `fake` backends.
- Registry.
- `council doctor`.
- Budget ledger.
- ✅ The doctor reports the real status; backends are unit-tested with mocks (no real calls in CI).

**M3 — Engine with scripted protocol, plain-text renderer.**
- Opening, blind round, one targeted rebuttal round driven by the moderator JSON, synthesis.
- `/step` semantics via a simple line-based REPL (`--no-tui`).
- Transcript persistence.
- ✅ A full session runs end to end with the fake backend in tests and with real backends manually.

**M4 — Textual TUI.**
- IRC layout, nick list, state panel, slash commands.
- `/step`, `/run N`, `/pause`, `/say`, `/ask`.
- ✅ Usable interactive session; pause works mid-run at turn boundaries.

**M5 — Memory.**
- Private and shared memory, approval flow, consolidation, `/forget` and the CLI reset.
- ✅ A second session demonstrably uses notes from the first; a reset works.

**M6 — Robustness & polish.**
- Resume, fallback-to-local, budget warnings, `/vote`, `/summary`, `/summon`, `/kick`, `/attach`.
- Streaming output from Ollama.
- `council sessions` commands.
- README.
- ✅ Killing the app mid-run and resuming loses at most one turn.

**M7 — Dynamic moderator (stretch).**
- Smarter turn selection, loop detection, consensus detection, early stop.
- Moderator-proposed `/summon` of a missing expertise.

## 15. Future ideas (explicitly out of MVP scope)

- Read-only tools for personas, e.g. a "Data Analyst" persona that runs sandboxed Python on dossier CSVs, or local web fetch.
- Councils that convene on a schedule (e.g. a monthly finance review) and write a report.
- Export a session to a nicely formatted HTML or Markdown report.
- A persona "generator": describe a persona in one line and get a full `.md` draft.
- Comparing the outcomes of the same question across different council compositions.

## 16. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Personas converge or become sycophantic | Blind round, structural roles, explicit "no polite agreement" rule, mixed models, private memory |
| Claude Pro limits hit mid-debate | Claude only where it adds the most (moderator); budget ledger and warnings; pause or local fallback |
| Small models lose the plot (short context, weak instruction following) | Moderator-maintained debate state instead of the full transcript; short turns; strict output shape; robust parsing with fallbacks |
| Moderator JSON is malformed | Pydantic validation, one repair retry, round-robin fallback |
| Model swaps on the GPU make rounds slow | Few distinct local models; schedule by model; `keep_alive`; show `thinking…` status |
| CLI flags change between Claude Code versions | Wrap all flags in one place (`claude_cli.py`); `council doctor` checks the version; spike notes |
| Personal data leakage | Everything gitignored under `data/`; prompts go only to the Claude CLI; no telemetry; argv never carries the prompt |
| Memory grows noisy or wrong | Distilled notes, a size cap with consolidation, user approval for shared memory, easy reset |

## 17. Example personas to ship

| Slug | Nick | Role | Backend | Angle |
|---|---|---|---|---|
| `chair` | Chair | moderator | claude_cli (sonnet) | Neutral facilitator; structured JSON decisions; writes the synthesis |
| `rigor` | Rigor | risk_officer | ollama | Downside first; demands stated assumptions |
| `compounder` | Compounder | advisor | claude_cli (haiku) or ollama | Long-term, growth-oriented, opportunity cost |
| `pragmatist` | Pragmatist | advisor | ollama | Simplicity, time and energy cost, "what will you actually stick to" |
| `contrarian` | Contrarian | devil_advocate | ollama | Attacks the leading position, flags when it is doing so |
| `wanderer` | Wanderer | advisor | ollama | Travel: experiences, slow travel, local culture |
| `logistician` | Logistician | advisor | ollama | Travel: budget, timing, transport, constraints |
| `mentor` | Mentor | advisor | claude_cli | Career: long arc, skills, reputation |

Example councils:
- `finance`: chair + rigor, compounder, pragmatist, contrarian.
- `career`: chair + mentor, pragmatist, rigor, contrarian.
- `travel`: chair + wanderer, logistician, pragmatist.
