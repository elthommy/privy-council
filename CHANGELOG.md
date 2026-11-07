# Changelog

All notable changes, by milestone. Dates are when each milestone was built.

## Unreleased (2026-10-25 – 2026-11-07)

- **A logged-out `claude` CLI stops the step** with "Claude is not logged in… /step retries", like a usage limit, instead of running the debate without the Chair or a member.
- `career` has a brief for the Chair: separate what you saw from how you read it, find who actually holds the decision, challenge your own reading, weigh the fears instead of taking them as given.
- **Better flaw reports**: the Chair no longer flags a figure of speech ("keep your head down") as a contradiction, and now reports base rates given as fact with no data, a member claiming personal experience, specifics nobody gave stated as established, and a date given a meaning the facts don't.
- **A sharper synthesis**: Consensus lists only points no member argued against; the synthesis checks itself against the opening's "a good answer must cover" list and names what the debate never settled; it states one recommendation where the advice leads to one, leaving you only the choices members still disagree on; wording it suggests follows the agreed framing; next steps hold no trigger that already happened.
- **The Chair checks triggers against your facts** (flagging one that already happened or can't happen) and pursues what nobody took up: details in your text, options nobody raised, named things the debate builds on without knowing what they are. It skips disputes too small to change the advice.
- **Private notes no longer credit you with the council's advice**: a recommendation in the synthesis was written down as your decision ("the user will not escalate without proof", "the user accepted the recaps"). Notes now record only what your own messages state, and your silence isn't agreement.
- **The private-notes cap is per council**: a persona that sits in several councils (the Chair) keeps `private_cap_bytes` of notes for each, and a merge rewrites only the council that outgrew it. An unusable merge is retried once with a lower target, and the reason is logged.
- **The Chair asks one question at a time**: its second question comes once you answer the first, and each gets the full wait (`question_timeout_s`); a timeout skips the rest, so an absent user holds the debate no longer than before. On Discord each person gets their own sequence. The Chair's questions now ask one thing each.
- **A reply no longer counts as an answer on its own**: at the next update the Chair judges which of its questions your lines left open, a half-answered one included, and the synthesis lists them.
- **Your facts stay in the debate state**: the cap of 12 forced the Chair to squeeze long stories at every update and facts vanished, so it's now 30, and a fact the Chair drops comes back (a reworded or merged one is left as it is).
- `career`: Skeptic moves to `muse-glimmer:30b`: fewer invented specifics in a seat bench, and it no longer shares a model with Contrarian. Its turns are slower (1-5 min).
- `career` requires web research, on public facts only (law, the employer's announced plans, the job market), never the people or events in your story.
- **Fewer false "unsourced" flags on members' own arithmetic**: written operations with currencies, `k` and `%` ("€18.2k ÷ 25.2%"), computed with the precise figure a rounded operand stands for; a percentage "of" a total from the prompt, in the same sentence; a band's middle; amounts a month ↔ a year; months from a monthly amount; years ↔ months; a derived value written again.
- **Your answers no longer fall out of the members' view mid-round**: members see the last K messages, and the debate state only records your lines at the next Chair update, so an answer given just after an update could scroll out before the round's last speaker (who then asked again). Every line of yours since the last update is now kept in their prompts.
- **TUI `/run` no longer waits forever on the Chair's questions**: it waits up to `[engine] question_timeout_s` (600 s, as on Discord), then goes on without the answer; sending an answer resumes the run at once, a half-typed answer holds it, Esc or another command stops the wait. `0` keeps the old behaviour. The `--no-tui` REPL still waits.
- **The Chair's last update asks you nothing**: no member could use the answer. What it would have asked, and any question the debate went on without, is listed in the synthesis's questions section (answered ones still never come back).
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
- New `economics` council for economic and market forecasts (a state's debt, company defaults, layoffs, crashes): Forecaster, Insider, Contrarian and two new personas, Macro and Analyst, both on nemotron-3.5-super (in a replay bench muse-glimmer ran past the output cap on company questions and kept flagged claims); its brief has Chair pin down what counts as the outcome. `finance` is described as personal finance.
- An error several members share is now caught for each of them: the Chair reports it per member, and the synthesis names everyone who never corrected it. A range whose multiplier is written once ("$860–940M") is no longer flagged as unsourced when the facts give it.
- A Chair flaw report whose own note says the figures hold ("shown only roughly") is dropped for every kind of flaw, not only arithmetic, so a member is no longer asked to correct a right figure. A weighted sum written out ("0.3×35% + 0.7×20% ≈ 24.5%") counts as the member's own arithmetic, not an unsourced figure.

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
