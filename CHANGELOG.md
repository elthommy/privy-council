# Changelog

All notable changes, by milestone. Dates are when each milestone was built.

## Unreleased (2026-10-25 – 2026-11-20)

- **The Chair's first search looks up what the question resolves on**: the latest value of the index, price or rate the question names, the level it is compared to (a record, a threshold), and whether the condition is already met. Applies to Claude research and to local search's queries.
- **The opening asks clearer terms questions** in forecast councils: up to two, each with complete, separate readings ("(a) ... or (b) ...?"). It checks which firms or borderline cases count, whether an event before today counts, and whether several conditions are all needed or any one.
- **A small model's prompt keeps the opening's research** (the check of the user's premises): it now comes first among the fresh facts, so trimming cuts the oldest later notes instead.
- A fact tag right after a plain number ("12% below the record close of 18,450.20 [F9]") vouches for that number, not for an earlier percentage, so the engine no longer reports it as miscited.

- **Forecast councils give numbers the engine can read** (`economics`, `geopolitics`; council option `estimate: probability`): each member ends every turn with `**Estimate:** N% (low–high%)`. The engine keeps each member's numbers, flags a value outside its own range and a number moved without "I update:", and shows the Chair and the synthesis the parsed numbers, so neither re-reads them from the prose. The synthesis gets the council's median and range from the engine.
- **The Chair can split a forecast on one event** the members weigh differently (say, whether lenders extend a loan). Members then also write their probability of the event and of the outcome with and without it, and the engine checks that these add up to their estimate.
- **The devil's advocate gives no probability** in forecast councils: it names the link it thinks the others misprice. Its arguments go under the disagreements, not into the confidence spread as a dissenting number.
- Estimate-line flags are logged and listed by `/flaws`; they reach the Chair only with `[engine] estimate_flaws_to_moderator = true` (off until a bench shows they are precise enough).
- **Fresh facts carry tags such as [F7]**, and members put the tag after a fact they use ("debt was $9.2bn [F7]"). The engine checks a tagged claim against that one fact: a figure the fact doesn't give, or a tag that is no fact, is flagged (with the fact that does hold the figure, if one does). These flags show in `/flaws` and, since a bench found 12 of 13 real, go to the Chair with the other figure flags, worded so it reads the cited fact before reporting one.
- Fact tags written as a span (`[F3-F5]`) are read, a figure a tagged sentence quotes from its facts is no longer flagged for a different word after the number ("5 months" for "five months"), and a range times a number ("$310-350M x 4") counts as the member's own arithmetic.
- **A local member can fix its draft before it is posted** (`[engine] repair_turns = "local"` plus `repair = true` on the model; on for gemma4, nemotron-3.5-bolt and qwen3.6, which fixed 62 of 63 flagged drafts in a bench, one rewrite losing content): when a draft trips a precise check (a wrong fact tag, a missing or inconsistent estimate line, a missing devil's-advocate line, a "Nick:" prefix or a literal "@nick"), the member rewrites it once and the rewrite is posted only if it has fewer such problems. Claude members are never repaired. The draft is kept in the transcript, and `/flaws` lists the fixed drafts with a count per member. Local members lose the typing effect while it is on.
- **Document reader** (`[research.documents] enabled = true`, off by default): `/read <url> <what to look for>` reads a whole public page, and the Chair may ask to read a page the fresh facts already cite. Built for figures deep inside long pages, such as a filing's debt table past the first 100,000 characters a web search reads. The engine fetches the page itself through a URL guard (public addresses only, redirects re-checked, size and time caps, HTML and plain text), picks the passages that match (on a page that repeats one table for several groups, such as small and large banks, the table whose title qualifies it as the question does), and the Chair summarizes them in one call without web tools. A bullet with a number the passages don't hold is dropped. The fetch sends only the URL (no cookies) with a generic User-Agent. Not available from Discord.
- **The Chair reads a page instead of searching again**: with the document reader on, its round update lists the official pages the fresh facts cite (HTML pages on `[research] primary_domains`, with what was already read there), and its rule now says when to read: a figure a note reports missing that one of those pages should hold. A lookup skipped as a repeat becomes a read of an official page those searches cited, if nobody read it yet. On two saved sessions this picks the filing whose debt note three searches couldn't reach. The log records how many searches and reads each update asked for.
- **The document reader finds the right table more often**: a table's title written as plain paragraphs above it (releases without heading tags) now stays with the table, and every part of a table cut by rows keeps its subtitle ("Not seasonally adjusted"). On 35 test lookups over four long official releases the wanted passage is picked 32 times instead of 29, and is the best-ranked one 21 times instead of 14. `scripts/passage_bench.py` saves pages and scores passage choices with no model call.
- **The document reader gets twice the room** (`[research.documents] passages = 12`, `passage_chars = 16000`, up from 6 and 8,000): on the same 35 test lookups the wanted passage is now always among those read. A read costs about 2,000 more input tokens.
- **The Chair doesn't ask the reader for pages of a site that refuses it** (`[research.documents] unreadable_domains`, `sec.gov` by default, which wants a contact e-mail in the User-Agent): those pages aren't offered to it, a read of one is skipped with a notice, and it keeps searching instead. Your `/read` still tries.
- **In forecast councils the Chair settles the question's terms first**: from the update after round 1 it checks what kind of event counts and who must be involved, the exact deadline (calendar or fiscal), and whether something dated before the question counts, and asks the user about any that would move a member's number, instead of leaving them to the synthesis. When members disagree on what the user's own words cover, the Chair is told that's the user's call, not a debate. In replays of two saved round-1 updates, the Chair asked such a question about half the time (5 of 12), against 1 of 6 without the rule.
- **The Chair no longer asks the devil's advocate for a number** in forecast councils: its estimate table says to ask which link the others misprice instead. Replayed on a saved session, the two floor prompts that had asked it for a probability now ask for a link and an argument. `scripts/replay_moderation.py` also prints the Chair's next turns.
- **Over-cap notes no longer stay over the cap**: a merge asks for a number of notes and words per note instead of characters, which local models ignored (Bolt answered "under 2,048 characters" with 4,319 bytes). On real over-cap notes, 13 of 13 bench merges on Bolt and gemma4 ended under the cap (`scripts/merge_bench.py`, no Claude call). If both replies are still too long, the newest notes that fit are kept (of the merge, else of the notes), instead of leaving the file to grow; the old copy stays in `notes.md.bak`. Shared facts are never trimmed.
- **A /run waits for the lookups the Chair hands you** (TUI): when the Chair relays a search or a page read it can't run on its own ("To allow it: /research …"), answering its question no longer resumes the /run. The /run waits until you type the `/research` or `/read` (it runs, and the /run goes on instead of stopping), `/step` or `/run N` (skip them), Esc, or the answer timeout. The command now ends the Chair's line, and a command copied from the chat with its wrap points (a line break before the focus) is read as one line instead of answering "unknown command".
- **A local Chair reads a page when its facts point to one**: when its round update asks for no page read, a local Chair is asked one short question: which of the official pages its facts cite holds a figure still missing. On a saved local-only session it picked the right page (a projections table) 4 times out of 4, where it had asked for nothing live. Release calendars no longer count as readable pages. `scripts/replay_moderation.py --pick-read` replays the pick.
- **The opening gives the date, the round count and the time left**: the Chair's opening task says today's date, how many rounds the debate has, and the time left to each date the question names, counted by the engine (a local Chair had called six months "18 months", and members repeated it). In forecast councils the opening asks you about a term the question leaves open (what counts, which release or period), instead of settling it.
- Local web search drops results whose link has no host (a search engine's redirect link was cited as a source), and a "Missing:" line with two labels keeps only the last part.
- A duration a member states ("over 7 months", "one per 4-5 years") is no longer flagged as a figure in the wrong unit: every such flag reviewed was false.
- **Security audit fixes**: the HTML export no longer loads images (a Markdown image in a turn or a guest's line is shown as text, and the page has a Content-Security-Policy), so opening an export sends nothing off the PC. In Discord, a notice or vote that quotes a raw error (a local URL, a model name) shows a generic word instead; the error stays in the local log.
- **The Chair stops searching for what it can't find**: a lookup of its own that repeats two earlier ones (half the words in common, no new date or number) is skipped with a notice; your `/research` always runs. In saved sessions this would have skipped 7 lookups, all of which found nothing new, and no useful one. Each search is told the queries already run and what is still missing, and the fresh facts start with "Not findable so far", which the synthesis turns into next steps.
- **Facts from official sources are tagged `[F7, primary]`** (regulators, statistics offices, central banks; `[research] primary_domains`), and members are told a primary fact wins when facts disagree. The Chair can report a figure taken from another fact when a primary one says otherwise.
- **`/facts`** lists the fresh facts with their tags (TUI, REPL, and Discord for everyone); the exports show the tags too.
- A member's own range on its estimate line is no longer flagged as an unsourced figure.
- **A figure the member presents as its own estimate is no longer flagged as unsourced**: the "(estimate, from memory)" label, or its sentence saying so in other words ("from memory", "my estimate", "de mémoire"; for a percentage also "I update", "my position"). Every such flag in a labelled review was false. The Chair's own "unsourced" report on such a figure is dropped the same way.
- **A moved estimate is flagged only when no reason is given**: a change explained in other words ("I correct myself", "my 30% was…", "from 9% to 12%", or a turn that names its old number) counts as explained, and the round after you speak to the council isn't checked, since your answer may have changed the question.
- **`gemma4:26b` prefills with `num_batch = 512`** (was 2048): its larger buffer left it short of GPU memory when another program took some, and the Contrarian then missed every round of a debate. About 10 s more per gemma turn.
- **A repeated out-of-memory error logs who holds the GPU memory** (`nvidia-smi`, in `data/logs/council.log`), so the next one shows which program took it.

- **Skeptic goes back to `gemma4:26b`**: with Rigor and Skeptic both on `nemotron-3.5-bolt`, Skeptic gave Rigor's reasons, assumption included. A different model family keeps the career seats apart.
- **The Chair's notes check no longer deletes notes**: it can only rewrite the part of a member's note that misstates you. With a local Chair, the check had emptied most notes about the members' own arguments and lessons.
- **A local model that runs out of GPU memory twice gets a third try 15 s later**, once memory held by other programs (desktop, browser) may be free; a gemma4 turn failed this way while the second try came only 10 s after the first.
- **Eight more seats move from Super to `nemotron-3.5-bolt`**: Macro, Analyst (economics), Realist, Economist (geopolitics), Value (purchases), Vitality (wellbeing), Saver (gacha), PolSci (politics), after a live check of each council. Logistician (history) stays on Super for now: Bolt reversed a historical fact there. Super now serves only Logistician and the local Chair.
- **Rigor and Skeptic move to `nemotron-3.5-bolt`** (74B MoE, ~9B active, 25 GB): in a seat bench it was about as accurate as Super at about a third of the time per turn (~30 s against 76-180 s), and good in French. Pull it with `ollama pull nemotron-3.5-bolt`.
- **The Chair checks what members' notes say about you** before they're saved: at `/close` it reads every member's new notes against your own messages (and the questions you answered) and corrects a claim you never made: a member's argument credited to you, an answer saved without the condition it was given under ("would stay" for "would stay if management improved"). One call on the Chair's model; if it fails, the notes are saved as written.
- **A note that repeats an existing one isn't saved again**, e.g. last session's lesson copied with a new tail. A note that changes a figure counts as new.
- **The Chair's instructions to members no longer add details about you**: what it says about you comes from your facts or your own words.
- **Optimist argues the upside**: its persona now makes the case for the best realistic outcome and the move that bets on it, instead of a softer copy of the cautious plan.
- **Votes no longer all say 85 %**: members set their confidence from the objections still standing, not from an earlier number; a vote JSON followed by extra text (muse-glimmer) is no longer rejected.
- **Evidence nobody gave stays visible**: the Chair keeps a source, base rate or document a member asked for in the open questions until someone gives it, and the synthesis names any the advice still rests on.
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
