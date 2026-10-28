# CLAUDE.md — Privy Council (`privy_council`, command `council`)

Guidance for Claude Code working in this repository. **Read `PROJECT_BRIEF.md` first**: it is the source of truth for scope, behavior and milestones. This file covers *how* to work here.

## Project in one paragraph

`council` is a single-user, local-first Python TUI. A group of LLM personas, each defined in a Markdown file, debates a user's question in an IRC-style chat under a moderator. Models run **only** through:
- the Claude Code CLI (`claude -p`, on the user's Pro subscription);
- local Ollama models (12 GB VRAM, so small models only).

Personas keep resettable memory across sessions.

## Absolute rules

1. **Never use a paid API.**
   - Do not add the `anthropic` or `openai` SDKs, and do not read or require `ANTHROPIC_API_KEY`.
   - Claude is reached **only** by running the `claude` CLI as a subprocess, with `ANTHROPIC_API_KEY` removed from the child env.
   - If a task seems to need an API, stop and ask.
2. **No real model calls in tests.**
   - Tests use `backends/fake.py` or mocked subprocess and HTTP calls (`pytest-httpx`, monkeypatched `asyncio.create_subprocess_exec`).
   - Live checks live in `scripts/` or behind `pytest -m live`, which is skipped by default.
3. **Personal data stays local.**
   - `data/`, `config.local.toml` and anything under `sessions/` or `memory/` are gitignored.
   - Never commit sample data containing real personal info.
   - Never put prompt content on a subprocess argv; use stdin.
4. **All engine prompt text lives in `src/privy_council/engine/prompts.py`.**
   - No prompt strings scattered through the code.
   - Persona prompts live in `personas/*.md` only.
   - A council's context for its moderator (`moderator_brief`, `research_focus`) lives in `councils/*.yaml`: Chair stays one persona.
5. **All `claude` CLI flag handling lives in `src/privy_council/backends/claude_cli.py`.** Flags change between versions. Check `docs/spike-notes.md`, and confirm with `claude --help` before changing them.
6. **The engine is UI-agnostic.** `engine/` must not import from `tui/`. It communicates through events and commands only.

## Security model

Single user, one PC, local-first. Keep these boundaries in mind whenever you add or
change an entry point (a command, a front end, a network call, a file read);
`/security-audit` audits against this section.

- **Trusted**: the owner at the keyboard (TUI, REPL, CLI), the code, `config*.toml`,
  Ollama on localhost, the installed `claude` CLI.
- **Untrusted input, and how it is bounded**:
  - *Model output* (Claude and Ollama), text and JSON: parsed with Pydantic, never
    executed. Moderator decisions can only plan turns, ask questions and propose a
    `/summon`, which the owner must accept. Rendered without markup (TUI markup
    off, HTML export escaped).
  - *Web content* (research by Claude's WebSearch and WebFetch, or SearXNG
    snippets): data for the Chair, never instructions. Local search drops summary
    bullets whose figures or URLs aren't in the hits. **Accepted risk**: Claude
    research pre-approves `WebFetch` (about 30 % of research calls read pages, ledger
    check 2026-10-28), so injected text in a result could make it fetch any URL,
    carrying the question or debate state in the query string. Normal sessions only;
    local-only and Discord sessions never reach Claude. Page fetching in our own code
    (local search) still needs the URL guard (http(s) only, no private or loopback
    IPs, re-checked after redirects, size and time caps).
  - *Discord* (`council chat discord`): the only input reachable from outside the
    PC, through an outbound connection only (no open port). Friends are allowlisted
    by **user ID**; they can only speak, answer the Chair, and read `/status` and
    `/flaws`. The bot's commands are a fixed table (never `/attach`, `/export`,
    `/research`, memory or roster commands), chat text never reaches the REPL/TUI
    parser, nothing pings, local paths and raw errors stay in the local log, and
    messages are capped in size and per round. Chat sessions are always local-only
    with memory off.
  - *Dossier files*: the owner's own, but their content goes to models. The dossier
    itself never goes into web search prompts; the Chair's debate state does, and it
    may summarize the dossier (accepted: queries are short and rarely carry it).
- **Intended data flows**: prompts to Claude only through the CLI on stdin, with
  API keys stripped (normal mode); search queries leave the PC only through the
  claude CLI or, with the `[research.local]` opt-in, SearXNG (local-only and Discord
  sessions; normal sessions only with `[research] provider` "local" or "auto"; the
  Chair's query prompt never carries the dossier itself); everything else stays
  in `data/` (gitignored). Local-only sessions can't reach Claude (registry guard).
- **Attackers that matter**:
  1. Someone posting in the Discord channel (a friend, or anyone who gets in)
     trying to read local files or memory, run commands, push data out through the
     channel, ping people, or flood the debate and the GPU.
  2. Injected text in web results, guest lines or model output steering the Chair:
     it may degrade a debate, but must not reach the PC, the files or memory.
  3. Leaks of secrets or personal data: the bot token (`SecretStr`, only in
     `config.local.toml`, mode 600), personal data in commits, logs (prompts only
     at DEBUG) or on argv.
- **Out of scope**: other local users on the machine (`data/` is 0755; fine on a
  single-user PC), a compromised Discord, Anthropic or Ollama, physical access, and
  what Discord and SearXNG's upstream engines see (documented: Discord isn't
  end-to-end encrypted, and local search is the owner's consent).

## Tech stack

- Python **3.12+**, managed with **uv**.
- **Textual** for the TUI, **Typer** for the CLI, **Pydantic v2** for models and config, **httpx** (async) for Ollama, **python-frontmatter** (or PyYAML) for persona files, **Jinja2** optionally for prompt templates, and `tomllib` for config.
- Testing: **pytest**, **pytest-asyncio**, **pytest-httpx**. Style: **ruff** (lint + format). Types: **mypy** in strict mode on `src/`.
- Use asyncio throughout the engine and backends. No threads, except where Textual requires them.

Ask before adding any dependency outside this list.

## Commands

```bash
uv sync                          # install deps
uv run council --help            # run the app
uv run council doctor            # check claude CLI, ollama, models, config
uv run council new finance "topic" --no-tui   # plain-text REPL mode (handy for debugging)
uv run council memory show       # private notes + shared user facts (data/memory/)
uv run pytest                    # tests (no live calls)
uv run pytest -m live            # opt-in live tests (real claude/ollama) — uses quota!
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Before considering a task done, run `ruff check`, `ruff format`, `mypy src` and `pytest`, and make sure they all pass.

## Repository layout

```
src/privy_council/
  cli.py, config.py
  models/      persona.py council.py session.py moderator.py
  loaders/     personas.py councils.py
  backends/    base.py claude_cli.py ollama.py fake.py registry.py
  engine/      protocol.py (DebateEngine) core.py memory_flow.py research_flow.py
               state_machine.py prompts.py notices.py context.py moderation.py
               research.py local_research.py resume.py scheduler.py budget.py
  search/      base.py searxng.py (local web search, local-only sessions)
  chat/        frontend.py render.py names.py store.py launch.py cli.py
               discord_bot.py (the only module importing discord.py)
  memory/      store.py distill.py
  storage/     sessions.py
  tui/         app.py widgets.py commands.py
  util/        tokens.py time.py
personas/      example persona .md files (committed)
councils/      example council .yaml presets (committed)
docs/          spike-notes.md, design notes
scripts/       manual/live helper scripts
tests/         mirrors src/ layout; fixtures in tests/fixtures/
data/          runtime data (GITIGNORED): memory/, sessions/, usage.jsonl, chat/, searxng/
```

## Key design decisions

These are already decided; don't relitigate them without asking.

- **Persona = Markdown + YAML frontmatter.**
  - The body is the persona's system prompt.
  - The engine wraps it with the protocol and round instructions, the debate state, memory and the dossier.
  - Validate the frontmatter with Pydantic. Errors must name the file and the field.
- **Context strategy for small models.** Never send the full transcript. Each turn gets:
  - the persona prompt;
  - the engine instructions;
  - the moderator's **debate state**;
  - the last K messages;
  - memory;
  - dossier excerpts.
  
  All of this is trimmed to the model's `num_ctx` budget in `engine/context.py`.
- **Blind first round.** In round 1, personas must not see each other's answers. The one exception is the `devil_advocate`: it speaks last in round 1 and sees the blind takes, so it knows which position to argue against.
- **Moderator outputs strict JSON** (`models/moderator.py::ModeratorDecision`).
  - Parse with Pydantic.
  - On failure, retry once with a repair prompt.
  - On a second failure, fall back to a round-robin turn order with a warning event.
  - Never crash a session over bad model output.
- **Ollama concurrency is 1 by default.** Schedule turns to minimize model swaps (`engine/scheduler.py`). Claude calls may run concurrently in the blind round, but also respect the budget.
- **Claude is a scarce budget.**
  - Every call goes into the `data/usage.jsonl` ledger.
  - Warn before `/run N` if the estimated calls exceed the soft caps.
  - Handle usage-limit errors gracefully: pause, or fall back to a local model if enabled.
- **Pause is cooperative.** Check it at turn boundaries only; never kill an in-flight call.
- **Persistence.**
  - Append every event to `transcript.jsonl` immediately.
  - `resume` rebuilds state from disk.
  - A crash should lose at most one in-flight turn.
- **Memory.**
  - Distilled, dated Markdown bullets with a size cap and a consolidation pass.
  - Shared memory updates require user approval and record only what the **user** stated.
  - Reset is available via `/forget`, `council memory reset`, or deleting `data/memory/...`.

## `claude_cli` backend checklist

When implementing or changing the backend, make sure:
- [ ] it calls through `asyncio.create_subprocess_exec` (no shell), with `cwd` set to a dedicated **empty** temp dir;
- [ ] the child env is a copy of `os.environ` **minus** `ANTHROPIC_API_KEY` (and any other API-key variables);
- [ ] the prompt goes through stdin, never argv;
- [ ] it streams JSON events (`--output-format stream-json`), enforces the hard output cap by killing the process past `max_output_tokens`, and parses the final `result` event (text, usage) plus the `rate_limit_event` (Pro quota), raising typed errors (`BackendAuthError`, `BackendRateLimitError`, `BackendTimeoutError`, `BackendOutputError`);
- [ ] it disables tools (`--tools ""`), uses a single turn and a custom system prompt, as supported by the installed version (see `docs/spike-notes.md`). The one exception is moderator web research, which passes `CompletionRequest.tools` (WebSearch/WebFetch, pre-approved via `--allowedTools`; spike 3);
- [ ] it honors the timeout, kills the process on timeout and reaps it;
- [ ] the call is recorded in the budget ledger.

## `ollama` backend checklist

When implementing or changing the backend, make sure:
- [ ] it calls `POST {base_url}/api/chat` via a shared `httpx.AsyncClient`, passing `model`, `messages`, `stream`, `options.temperature`, `options.num_ctx` and `keep_alive`;
- [ ] the health check verifies the server is up and the models are pulled, and prints the `ollama pull <model>` hint otherwise;
- [ ] all calls go through the global semaphore;
- [ ] streaming support comes later (M6), as an async iterator for the TUI typing effect.

## Coding conventions

- Full type hints. `mypy --strict` must be clean on `src/`.
- Pydantic models for anything that crosses a boundary: files, model output, config, events.
- Small, focused modules. Prefer pure functions in `engine/context.py` and `engine/protocol.py`, which are easy to unit-test with the fake backend.
- Errors: use typed exceptions in `backends/base.py` and the engine. Surface them to the UI as `ErrorEvent`s with actionable messages; there should be no bare stack traces in the TUI.
- Logging: the `logging` module, to `data/logs/council.log`. Never log full prompts at INFO level, since they contain personal data; use DEBUG and keep it off by default.
- Paths: `pathlib` only; resolve everything relative to the config `[paths]`.
- Time: timezone-aware datetimes. Display in local time.
- Comments explain *why*, not *what*. Keep docstrings concise.

## Testing strategy

- **Unit tests:**
  - persona and council loaders, including invalid files;
  - context building and trimming;
  - moderator JSON parsing and fallback;
  - state machine transitions;
  - memory cap and consolidation trigger;
  - the slash-command parser.
- **Backend tests:**
  - `claude_cli` with a mocked subprocess: success JSON, auth error, rate limit, timeout, garbage output;
  - also assert the env has no `ANTHROPIC_API_KEY` and the prompt is not in argv;
  - `ollama` with `pytest-httpx`.
- **Engine integration:** a full session with `FakeBackend` scripted responses: blind round, rebuttal, synthesis, pause/resume, crash-resume from `transcript.jsonl`.
- **TUI:** Textual's `App.run_test()` pilot for smoke tests of the commands.
- **Live tests:** marked `@pytest.mark.live`, skipped by default, kept minimal to save quota.

## Workflow for Claude Code in this repo

- Work **milestone by milestone** (see `PROJECT_BRIEF.md` §14). Start with **M0** (spike) and record its findings in `docs/spike-notes.md` before building backends.
- **Read `docs/STATUS.md` at the start of a session, if present.** It's the owner's private dev log (progress, decisions made with the user that supersede the brief) and is left out of the public export (`export-ignore`). Update it at the end of each milestone, and add user-facing changes to `CHANGELOG.md`.
- For each milestone:
  1. propose a short plan;
  2. implement with tests;
  3. run the full check suite;
  4. summarize what changed and what's next.
- Keep commits focused, using conventional-commit style (`feat(engine): ...`, `fix(ollama): ...`, `test: ...`).
- If a spec detail in the brief is ambiguous or seems wrong, ask instead of guessing. Ask especially about anything touching the cost rules, memory semantics or personal data.
- Don't run live model calls (which consume the user's Claude quota) unless explicitly asked.
- When adding example personas or councils, keep them generic. No real personal data.
