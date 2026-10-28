# M0 spike notes

Date: 2026-10-15. Scripts: `scripts/spike_claude.py`, `scripts/spike_ollama.py` (stdlib only, live).

## Environment

| Item | Value |
|---|---|
| Claude Code CLI | `2.1.301 (Claude Code)` at `~/.local/bin/claude` |
| Auth | `claude auth status` → `authMethod: claude.ai`, `subscriptionType: pro` |
| `ANTHROPIC_API_KEY` in parent env | not set |
| Ollama | client `0.40.4`; systemd `ollama.service` (disabled, not running by default); models in `/var/lib/ollama`; drop-in sets `OLLAMA_CONTEXT_LENGTH=32000` |
| GPU | RTX 4070 Ti, 12282 MiB (~2 GB used by desktop at idle) |
| Python | system 3.14.8, uv 0.13.1 |

## `claude` CLI — verified invocation

```
claude -p
  --output-format json
  --model <alias>                 # haiku | sonnet | opus | fable, or a full model id
  --system-prompt-file <path>     # REPLACES the default system prompt; keeps prompt text off argv
  --tools ""                      # disable all built-in tools
  --no-session-persistence        # nothing written to ~/.claude/projects
  --safe-mode                     # no CLAUDE.md, skills, plugins, hooks, MCP, custom agents
  --strict-mcp-config             # belt and braces: no MCP servers
  --setting-sources ""            # ignore user/project/local settings
  [--json-schema '<schema>']      # structured output (see below)
```

- Prompt goes on **stdin**. The system prompt goes in a 0600 file in the call's temp dir, via `--system-prompt-file`. That flag is missing from `--help` but documented under `--bare`, and it works. So **no prompt content is ever on argv**.
- `cwd` = an empty dir inside a `tempfile.TemporaryDirectory`.
- Child env = `os.environ` minus `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`.
- **Do NOT use `--bare`.** It forces API-key auth (`ANTHROPIC_API_KEY` or apiKeyHelper only, OAuth never read), so it would either fail or bill API credits. `--safe-mode` + `--setting-sources ""` give the same isolation and keep subscription auth.
- Single turn: with `--tools ""` there is nothing to loop on, so plain calls report `num_turns: 1`. A hidden `--max-turns N` flag exists and is accepted, but **don't set `--max-turns 1` together with `--json-schema`** (see below).
- Unknown flags fail fast with `error: unknown option '--x'` on stderr, before any API call. `council doctor` can use this to detect flag drift without spending quota.

### Persona call (haiku): works

The model introduced itself as Rigor, reported no tools, and saw no CLAUDE.md. Only ~634 input tokens were sent, which proves the default Claude Code system prompt was replaced. A small env-info block is still injected: the model mentioned "environment details".

- Wall time 2.8 s (API 1.65 s).
- Thinking tokens (162) are counted in `output_tokens` (232). Haiku thinks by default, so turn budgets must leave headroom. Check whether `--effort low` reduces this in M2.

### Structured output (`--json-schema`): works

- The result has `structured_output` (an already-parsed object), and `result` holds the same JSON as a string.
- It reported `num_turns: 2`: the CLI uses an internal structured-output turn, even with `--tools ""`. It is still one CLI invocation, so count it as **1 call** in the ledger and log `num_turns`.
- **Great fit for the moderator:** we can pass `ModeratorDecision.model_json_schema()` and still validate with Pydantic, keeping the repair/fallback path for Ollama moderators and as a safety net.
- The model ignored the "≤3 sentences" instruction in the system prompt here. Length limits need to be enforced in the engine prompts too.

### JSON result envelope (fields to rely on)

| Field | Meaning |
|---|---|
| `type` | `"result"` |
| `result` | reply text (or the error message when `is_error`) |
| `structured_output` | present only with `--json-schema` |
| `is_error` | **the error flag to trust** |
| `subtype` | `"success"` **even for errors**: don't use it to detect failure |
| `terminal_reason` | `"completed"` or `"api_error"` (others likely) |
| `api_error_status` | HTTP status for API errors (null when not logged in) |
| `stop_reason`, `num_turns`, `session_id`, `duration_ms`, `duration_api_ms`, `ttft_ms` | metadata |
| `usage.{input_tokens,output_tokens,cache_creation_input_tokens,cache_read_input_tokens}` | token usage for the ledger |
| `modelUsage.<model-id>` | resolved model id (e.g. `claude-haiku-5-5`), `thinkingTokens`, `contextWindow` |
| `total_cost_usd` | notional list-price cost (`costBasis: "list"`); not billed on Pro, but handy as a size metric in the ledger |

### Errors

- **Not logged in.** Verified at zero quota by pointing `CLAUDE_CONFIG_DIR` at an empty dir: exit code `1`, stdout still valid JSON, `is_error: true`, `terminal_reason: "api_error"`, `api_error_status: null`, `result: "Not logged in · Please run /login"`. Map to `BackendAuthError` by matching `"Not logged in"` / `/login` in `result`.
- **Rate / usage limit: NOT observed.** Triggering it would burn the whole quota window. Assumption for M2: `is_error: true` with `api_error_status: 429`, and/or a `result` mentioning "usage limit" / "limit reached", possibly with a reset time. Map to `BackendRateLimitError`, carry the raw `result` text so the UI can show it, and try to parse a reset timestamp opportunistically. **Revisit the first time it happens for real.** The backend should log the full envelope (minus prompt) on any unrecognised `is_error`.
- **Timeout.** Handled in our code: `asyncio.wait_for` → `proc.kill()` → `await proc.wait()`.
- Non-JSON stdout (e.g. a crash before JSON) → `BackendOutputError`.

### Latency / quota used by this spike

Two live haiku calls (persona 2.8 s, schema 3.8 s wall), plus two logged-out calls that cost no quota.

## Ollama

Server `0.40.4`. **`ollama.service` is disabled**, so it must be started (`sudo systemctl start ollama`) before a session. `council doctor` should say exactly that when it gets `Connection refused`.

Installed models of interest (on-disk sizes): `qwen3:4b` 2.5 GB, `qwen3:8b` 5.2 GB, `gemma3:12b-it-qat` 8.9 GB, `codellama:13b-instruct` 7.4 GB. `qwen3.6` (23.9 GB) and `nemotron-3.5-lightning` (25.4 GB) are too big for 12 GB VRAM.

### `/api/chat`: works

Verified payload: `model`, `messages` (a `system` role message carries the persona prompt), `stream: false`, `think: false`, `keep_alive: "10m"`, `options.{temperature, num_ctx, num_predict}`. The response gives `message.content` plus timing fields in nanoseconds: `load_duration`, `prompt_eval_count/duration`, `eval_count/duration`, `total_duration`. Token usage for free. `/api/ps` reports `size` vs `size_vram` (offload detection); `/api/show` reports `capabilities` (e.g. `thinking`).

### Measurements (short prompt, ~60 tokens in, ≤200 out)

| Model | num_ctx | Resident size / in VRAM | Load: disk-cold | Load: swap (page cache warm) | Gen speed |
|---|---|---|---|---|---|
| `qwen3:8b` | 8192 | 6.3 / 6.3 GB | 6.0 s | **2.8 s** | **~82 tok/s** |
| `qwen3:8b` | 4096 | 5.6 / 5.6 GB | — | 2.8 s | ~82 tok/s |
| `qwen3:4b` | 8192 | 3.9 / 3.9 GB | 4.6 s | 2.6 s | ~141 tok/s |
| `gemma3:12b-it-qat` | 8192 | 9.8 / **7.9 GB** | 12.5 s | 4.8 s | **~23 tok/s** |
| `gemma3:12b-it-qat` | 4096 | 9.7 / **8.0 GB** | — | 5.3 s | ~25 tok/s |

Warm calls (model already resident) take ~0.3 s for a 25-token reply.

### Findings

- **Model swap costs ~3 s (8B) to ~5 s (12B)** with the OS page cache warm, and 6–12 s straight from disk. Only one of these models fits at a time. Scheduling by model within a round (brief §9.2) matters, but a swap is not catastrophic.
- **`gemma3:12b-it-qat` doesn't fit.** Even at `num_ctx` 4096 it needs ~9.7 GB. The desktop holds ~2 GB of VRAM, so ~1.8 GB spills to CPU and generation drops to ~1/3.5 the speed. Not recommended unless the desktop's VRAM use drops.
- **`qwen3:4b` ignores `think: false`.** It emits its chain of thought as plain `content`, with no `<think>` tags, and burns the whole `num_predict`. `qwen3:8b` honours `think: false` and answers cleanly. Don't ship personas on `qwen3:4b`, or the engine would need to handle it specially.
- `qwen3:8b` follows the "≤2 sentences, terse" persona prompt well.
- The `OLLAMA_CONTEXT_LENGTH=32000` drop-in is only a default. We pass `num_ctx` per request, and it is respected (smaller resident size at 4096).

### Recommendation

- **Single workhorse local model: `qwen3:8b`, `num_ctx` 8192** (6.3 GB, fully on GPU, ~82 tok/s). Using it for every local persona means zero swaps.
- Model diversity comes from Claude (moderator + 1–2 personas) vs qwen3:8b, plus structural roles and blind round.
- Always send `think: false` for qwen3-family models (make it a per-model option in config, e.g. `[ollama.models."qwen3:8b"] think = false`).
- Revisit a second local model (e.g. a ~7–9B from another family) later if diversity is lacking. It must fit in ~9.5 GB including context.

## Decisions for M2

1. `claude_cli.py` uses exactly the verified argv above. `doctor` checks `claude --version` and runs a flag probe.
2. Request: the system prompt goes to a temp file; messages are rendered into a single stdin text, since the CLI takes one user turn.
3. Moderator on Claude uses `--json-schema` with the Pydantic schema; still validate, repair once, then round-robin fallback.
4. Error classification keys on `is_error` + `result` text + `api_error_status`, never on `subtype`.
5. Ollama backend sends `think` per model config and reads usage/timings from the response fields. `doctor` warns when `/api/ps` shows `size_vram < size` (CPU offload) and when the server is down (`sudo systemctl start ollama`).
6. Example personas default to `qwen3:8b` locally. The config default `fallback_model = "qwen3:8b"` from the brief is confirmed.

---

# Spike 2: hard output cap for `claude` (2026-10-16)

Script: `scripts/spike_claude_stream.py`. Goal: a generous **hard cap** that stops a runaway turn. Concision itself is a soft instruction in the prompts. The CLI has no max-tokens flag, and temperature is ignored for Claude.

## Options tested

| Option | Result |
|---|---|
| **(a) `--output-format stream-json --verbose --include-partial-messages`, kill the process past the cap** | ✅ Killed 2.5 s into a 600-word essay (haiku) and 3.4 s in (sonnet), keeping the partial text (413 / 608 chars). Stops spending on the runaway answer. **Adopted.** |
| (b) env `CLAUDE_CODE_MAX_OUTPUT_TOKENS=100` | ❌ Returns `is_error: true` with `"API Error: Claude's response exceeded the 100 output token maximum…"`: **the text is lost**. Not usable as a soft cut. |

## stream-json format (Claude Code 2.1.301)

One JSON object per line:
- `system/init`: session metadata (`tools: []` confirms no tools), `system/status`.
- `stream_event` wrapping raw API events:
  - `message_start`: has `message.usage` (input + cache tokens; kept for truncated calls);
  - `content_block_start` / `content_block_delta` / `content_block_stop`.
  
  Delta types:
  - `text_delta` (`.text`): the visible answer;
  - `input_json_delta` (`.partial_json`): **structured output** with `--json-schema`, streamed as a StructuredOutput tool call;
  - `thinking_delta` / `signature_delta`: haiku thinks first. Thinking text is **empty** (`thinking_display: "updates"`), with only estimated tokens. `system/thinking_tokens` events carry `estimated_tokens`.
  
  Sonnet emitted no thinking block.
- `assistant` / `user`: whole messages (large, with signatures). So **raise the asyncio StreamReader line limit** (default 64 KiB → 16 MiB).
- **`rate_limit_event`**, right after `message_stop` and before `result`:
  ```json
  {"rate_limit_info": {"status": "allowed", "resetsAt": 1792105200, "rateLimitType": "five_hour",
    "unifiedWindows": {"five_hour": {"utilization": 0.2, "resetsAt": …},
                       "seven_day": {"utilization": 0.08, "resetsAt": …}}}}
  ```
  This is the **real Pro quota** for all usage on the subscription, not just council's. It is **not** present in `--output-format json`, which is another reason to stream. A killed (truncated) call doesn't get one.
- `result`: **identical in shape to the M0 `--output-format json` envelope**, so the same error rules apply (`is_error`, never `subtype`). Logged out, it still ends with an `is_error` result (`"Not logged in · Please run /login"`), so the zero-quota flag probe works unchanged in stream mode.

## Implementation (`backends/claude_cli.py`)

- Cap = `max_output_tokens × 4` chars (`util/tokens.py`), counted over `text_delta` + `input_json_delta`. Thinking is not counted; the timeout covers runaway thinking.
- On cap: kill, reap, and return the partial text with `truncated=True`. Usage = `message_start` input tokens plus estimated output tokens. A truncated structured call returns `structured=None`, which goes down the moderator's repair/fallback path.
- The quota snapshot goes in `CompletionResult.raw["quota"]` and in the ledger (`UsageEntry.quota`). `council budget` / `doctor` show the latest one, and warn at `[claude] quota_warn_utilization` (default 0.8).
- A quota `status` other than `"allowed"` on an `is_error` result → `BackendRateLimitError`, with `retry_at` = the window's `resetsAt`. This is now a structured signal, rather than only guessing from text.
- Defaults: `[engine] max_output_tokens = 1500` (~1,100 words, a safety net only). Chair overrides it with 4000 (JSON state + synthesis).

## Moderator model comparison (same round-1 input, `ModeratorDecision.model_json_schema()`)

| Model | Wall | Output tokens | Valid JSON | Notes |
|---|---|---|---|---|
| haiku | 9.9 s | 1795 (mostly thinking) | ✅ | wordy positions; invented an "agreement" Rigor didn't state |
| sonnet | 9.4 s | 1056 | ✅ | concise, sharp disagreements; attributed an unstated argument to Rigor |
| opus | 18.7 s | 1770 | ✅ | richest framing, 2× slower |

- The Pydantic schema (with `$defs`/`$ref`) works as-is with `--json-schema`. Every call took `num_turns: 2`.
- **Keep sonnet for Chair.** M3's moderator prompt must say: *only attribute positions/arguments a persona actually stated; mark your own inferences as open questions.*
- Quota cost of the whole spike (≈8 haiku, 3 sonnet, 1 opus calls) was not visible at 1% resolution. The 5-hour window read 20% → 25%, but the user's interactive session shares it.

## Spike 3 — web research via the CLI's own tools (2026-10-16, Claude Code 2.1.301)

Goal: let the moderator search the web so Ollama members get fresh facts.

Flags, otherwise identical to our headless set:
```
--tools WebSearch,WebFetch          # instead of --tools ""
--allowedTools WebSearch,WebFetch   # pre-approves them: no permission prompt in -p mode
```
- **Works headless** in the empty temp cwd with `--safe-mode`, `--setting-sources ""` and stdin prompt. `system/init` lists `tools: ['WebFetch','WebSearch']`, and `permission_denials: []`. It runs on the Pro login with no API key.
- **Cost, two live sonnet calls:**

  | Query | Searches | `num_turns` | Wall time | Output tokens | Input tokens |
  |---|---|---|---|---|---|
  | One narrow question (ECB rate) | 1 | 2 | 20 s | 895 | ~5k |
  | Broad debate brief (gold: price, central banks, real yields) | 5 | 6 | 36 s | 1,906 | ~18k, mostly cache |

  Notional cost was $0.03 and $0.11. It's still one CLI invocation, so count it as 1 call in the ledger with `num_turns` logged.
- **Quality:** dated facts with source URLs; it flagged disagreeing sources and missing data unprompted. Unprompted, the answer is long (~7.5k chars), so the research prompt must ask for a compact brief.
- **Stream shape:**
  - Each search is a `content_block_start` of type `tool_use` (name `WebSearch`). Its input (`{"query": ..., "mode": ...}`) streams as **`input_json_delta`**, followed by a `user` event carrying the tool result.
  - `usage.server_tool_use.web_search_requests` stays 0: WebSearch is a CLI-side tool.
  - **Consequence for `StreamCollector`:** today it treats every `input_json_delta` as structured output and counts it toward the cap. With tools on, a capped call would return the tool's JSON input as its answer. Only count `input_json_delta` inside the `StructuredOutput` tool block (the `--json-schema` mechanism, see above). Collect `WebSearch` inputs separately, to show the queries in the chat.
  - Text before a tool call (if any) also streams as `text_delta`. The final answer is the `result` event's `result`, which holds the last turn's text.
- Thinking may be on (sonnet emitted `thinking_delta`). As before, it isn't counted toward the cap; the timeout covers it. Use a longer timeout for research (≥ 180 s).

## Spike 5: Ollama context size (`num_ctx`) with a dossier (2026-10-17, local only)

Goal: how much does a bigger `num_ctx` cost, and what does 8k lose once a dossier is attached?

Setup: `scripts/bench_models.py qwen3.6:latest=false gemma4:26b=false --personas rigor --num-ctx 8192,12288,16384 --round 3 --session <ai-bubble session: round 6, 10 research notes> --dossier <synthetic 330-row budget CSV, ~3.6k tokens: just under the summary threshold, so verbatim>`.
- The prompt is the engine's round-3 member prompt (debate state + last 6 messages + facts + dossier), trimmed to `num_ctx − 1500`.
- Each size reloads the model, so "cold" includes the load. "Warm" re-sends the identical prompt, so Ollama's prompt cache makes its prefill ~0.1 s: **cold prefill is the real per-turn cost** of a new prompt.

| Model | num_ctx | Prompt tok | Trimmed | Cold total | Load | Prefill | Warm total | Gen tok/s | Memory |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.6 | 8k | 4,098 | **all 6 messages dropped**, dossier + facts clipped | 33.0 s | 11.0 s | 7.8 s | 13.9 s | 40 | 21.7 GiB, 35% GPU |
| qwen3.6 | 12k | 6,146 | facts clipped | 37.7 s | 11.8 s | 11.6 s | 16.2 s | 41 | 21.8 GiB, 34% GPU |
| qwen3.6 | 16k | 8,194 | facts clipped | 44.1 s | 12.1 s | 15.3 s | 11.6 s | 42 | 21.9 GiB, 35% GPU |
| gemma4:26b | 8k | 4,099 | all 6 messages dropped, dossier + facts clipped | 28.0 s | 10.7 s | 6.5 s | 11.9 s | 40 | 80% GPU |
| gemma4:26b | 12k | 6,147 | facts clipped | 32.6 s | 10.7 s | 9.6 s | 11.3 s | 39 | 80% GPU |
| gemma4:26b | 16k | 8,195 | facts clipped | 33.7 s | 10.6 s | 13.1 s | 12.4 s | 37 | 80% GPU; `nvidia-smi` 11.4 / 12 GB |

Findings:
- **At 8k, a ~3.6k-token dossier pushes out the whole recent conversation.** The member then answers from the debate state alone. Without a dossier, this session's round-3 prompt still fit.
- **Memory hardly grows with `num_ctx`** for these MoE models: qwen3.6 goes from 21.7 to 21.9 GiB resident at a steady ~35% GPU share, and gemma stays at 80% GPU. But at 16k gemma brings the card to 11.4 of 12 GB, so there is little headroom past 16k.
- **The cost is prefill: ~1.9 s per 1k prompt tokens** on both models. Prompts fill their budget (the dossier and facts limits are shares of it), so each step up adds ~4 s to a turn with a fresh prompt: ~8 s at 8k, ~11 s at 12k, ~15 s at 16k. Generation speed is unchanged (~40 tok/s). All of this is within the ~30 s turn budget.
- `/api/ps` reports gemma4:26b at ~1.5 GiB, which is wrong for a 26B model; use `nvidia-smi` for it.
- Answer quality wasn't compared here.

## Spike 6 (paper only): what a 32 GB or 96 GB GPU would change (2026-10-25)

No GPU time, no Claude calls. Measured speeds fitted from `data/bench/runs/chair-super*.log` and `chair-gptoss.log` (least squares on wall time = prompt/prefill + output/generation), plus measured member speeds. Projections are estimates from memory bandwidth and what fits in VRAM; none is measured.

Platform: i7-10700K, **PCIe 3.0 x16** (~12 GB/s; `nvidia-smi` max gen 3), dual-channel DDR4 (~45 GB/s), 125 GB RAM, RTX 4070 Ti 12 GB. Weights (blob sizes): qwen3.6:27b 15.7 GiB, gpt-oss:120b 60.9 GiB, Super 82.3 GiB.

Measured today:

| Model | In VRAM | Prefill | Generation | Typical call |
|---|---|---|---|---|
| qwen3.6:27b (dense) | ~70% | — | ~5.5 tok/s | 34–128 s per member turn |
| gpt-oss:120b (5B active) | ~18% | 172 tok/s | 14 tok/s | Chair ~144 s (6.9k in, 1.5k out) |
| Super (12B active) | ~13% | 44–60 tok/s | 9–11 tok/s | Chair ~280–330 s, over half of it prefill |

Projected:

| Model | RTX 5090, 32 GB | RTX PRO 6000, 96 GB |
|---|---|---|
| qwen3.6:27b | fits; ~55–70 tok/s, ~8–12 s per turn (~8×); Q6_K (~22 GB) possible | same |
| gemma4:26b | fits; ~150+ tok/s, ~3–5 s per turn | same |
| qwen ↔ gemma swap | still swaps, ~3–5 s instead of 11–13 s | both resident, no swap |
| gpt-oss:120b | ~45% in VRAM; ~20–25 tok/s, prefill ~250–400; Chair ~80–90 s (~1.7×) | fits; ~150 tok/s, Chair ~12 s (~12×) |
| Super | ~33% in VRAM; ~12 tok/s (DDR4-bound), prefill ~100 (more with a bigger batch); Chair ~150–200 s (~1.7×) | fits (82.3 GiB + <1 GiB KV at 24k + ~2 GiB buffers of 95.6 GiB); ~60–100 tok/s, Chair ~20–30 s (~10–15×) |
| Super ↔ gpt-oss swap | 10–47 s, as today | ~10–45 s, then the largest remaining cost |

Per round: normal mode with 4 qwen/gemma turns ~5 min → ~40 s (5090) / ~30 s (PRO). Local-only finance round (Chair + Rigor on Super, 2 gpt-oss, gemma, swaps) ~10 min → ~6 min / ~2 min.

Hypothesis for Super's slow prefill (to test): for a prompt, llama.cpp copies the RAM-resident layers to the GPU once per batch of `num_batch` tokens (default 512). 78 GiB over PCIe 3 is ~6.5 s per batch, i.e. a ceiling near 80 tok/s, close to the measured 44–60. If right, `num_batch = 2048` would copy them 4× less often and cut prefill severalfold on the current card. Test plan: one replayed Chair item on Super at 512 and 2048 (~15 min of GPU, no Claude), comparing `prompt_eval_duration` and VRAM.

### Spike 6 result: `num_batch` (2026-10-25, ~20 min of GPU, no Claude calls)

`scripts/num_batch_bench.py`: one Chair round update (geopolitics Brent, round 1) sent straight to `/api/chat` with the engine's options plus `num_batch`, model unloaded before each size, a nonce at the top of the system prompt so no run hits the prompt cache, output capped at 200 tokens. Rows: `data/bench/runs/num-batch.jsonl`.

| Model (num_ctx) | num_batch | Prefill | Gen | Wall (200 out) |
|---|---|---|---|---|
| Super (24k), 9.7k tok | 512 | 147.4 s (66 tok/s) | 9.2 tok/s | 169 s |
| | 2048 | **57.7 s (168 tok/s)** | 7.8 tok/s | 83 s |
| | 4096 | **46.0 s (211 tok/s)** | 8.1 tok/s | 71 s |
| | 512 (control) | 148.4 s (65 tok/s) | 9.1 tok/s | 171 s |
| gpt-oss:120b (12k), 8.0k tok | 512 | 54.2 s (147 tok/s) | 16.8 tok/s | 66 s |
| | 2048 | **17.6 s (454 tok/s)** | 16.9 tok/s | 30 s |

- **Hypothesis confirmed**: prefill is bound by copying the RAM-resident layers over PCIe once per batch. 4× bigger batches give 2.5× (Super) and 3.1× (gpt-oss) faster prefill; the control repeats within 1%.
- Super loses ~12% generation speed at 2048/4096 (the bigger compute buffer leaves less VRAM for layers); gpt-oss loses nothing. On a full ~1,500-token Chair answer, Super at 4096 is still ~80 s faster per call (~5 → ~3.7 min).
- `/api/ps` `size` (4.7–9.4 GiB) is wrong for both models, as for gemma in spike 5; `nvidia-smi` showed ~10 GiB used on Super at 2048.
- Not measured: 4096 on gpt-oss, qwen3.6:27b (mostly on GPU, so less to gain), answer quality (the setting doesn't change the maths, only the batching).

Tuning round (same day, `config.toml` now sets `num_batch` per model):

| Model (num_ctx) | 512 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| Super (24k) prefill / gen | 147 s / 9.2 | 58 s / 7.8 | **46 s / 8.1** | 167 s / 4.1 (only 1.3 GiB left on GPU) |
| gpt-oss:120b (12k) | 54 s / 16.8 | 18 s / 16.9 | **11 s / 16.5** | — |
| qwen3.6:27b (12k) | **25 s / 5.8** | 15 s / 5.5 | 14 s / 4.9 | — |
| gemma4:26b (12k) | 13.7 s / 52.6 | **5.7 s / 51.3** | 4.8 s / 44.4 | — |
| qwen3.6:latest (12k, Claude fallback) | 16.4 s / 41.7 | 6.5 s / 42.6 | **4.7 s / 43.5** (repeat: 4.7 s / 40.0) | 4.1 s / 40.8 |

Kept (bold): the setting with the best whole turn. qwen3.6:latest gains only 0.6 s more at 8192 (tok/s within run-to-run noise, ±4%) for a bigger buffer, so it stays at 4096. qwen's turns are generation-bound (~400 tokens at ~5.5 tok/s), so 2048's 10 s of prefill is eaten by 5% fewer tok/s; it stays at the default. End to end, the 3 Super plant items at 24k ran in 247 / 204 / 251 s against 272 / 346 / 333 s with the same catches.
