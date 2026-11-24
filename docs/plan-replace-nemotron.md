# Plan: replace Nemotron with models Ollama supports better

Written 2026-11-24. **Result (same evening): phase 1 done, the plan stops at its stop rule.
Bolt stays, without its draft model** (`draft_num_predict = 0` in `config.toml`): 5/5 first-try
loads on 0.40.16 (median turn 31 s) and 5/5 on 0.40.17 (40 s), 900 MiB of GPU memory free.
With the draft: 0/5 first-try on 0.40.16 (retry works, 56 s) and no load on 0.40.17.
Granite failed the 90 s limit (148 s); qwen3.6:27b 86 s, flint 34 s, muse 266 s, Super 104 s,
all loading at once. Phases 2-6 are kept for the next time Bolt breaks or a new model appears.
Owner's decisions: Bolt without draft is fine; 80-90 s turns are fine; Super is out of scope.

Original status: proposed. Each phase needs the
owner's go-ahead (GPU time, downloads, Claude quota) and can be stopped between steps.

## Why

`nemotron-3.5-bolt` failed every load under Ollama 0.40.17 (42 of 42 requests on
2026-11-24), so every Bolt seat was silent. It loads again on 0.40.16.

- Bolt's package has an external **DFlash draft model** (about 1,260 MiB) that
  llama.cpp can't measure ("failed to measure the memory of the extra model"), and a
  vision projector (about 1,650 MiB) that never fitted on this 12 GB GPU: the first
  load failed every time, on every version we have logs for (since 2026-11-07).
- It worked only because Ollama's retry, with the projector on the CPU, kept 2,717 MiB
  free. On 0.40.17 that retry keeps 1,064 MiB. The llama.cpp change behind it isn't
  identified.
- Upstream: llama.cpp PR #31610 would measure such draft models (open, awaiting review).
  Ollama #19851 may be the same regression (no logs in it).

## Scope

| Seat | Council | Model | Councils' other local families |
|---|---|---|---|
| Rigor | finance, career, purchases | Bolt | gemma, muse (finance, purchases), qwen (career) |
| Value | purchases | Bolt | gemma, muse |
| Macro, Analyst | economics | Bolt | gemma |
| Realist, Economist | geopolitics | Bolt | qwen, gemma |
| PolSci | politics | Bolt | qwen, gemma |
| Saver | gacha | Bolt | qwen, muse, gemma |
| Vitality | wellbeing | Bolt | qwen, gemma |
| Logistician | history | Super | qwen, gemma |
| Chair, Forecaster, Specialist, Empiricist | local-only mode | Super (repo default) | |

**First target: the nine Bolt personas.** Super has no external draft model and no
failure was seen, but it wasn't loaded on 0.40.17: check it in phase 1, and replace it
only in phase 5, if the owner wants Nemotron gone altogether.

## What "better supported" means (hard gate, checked before any quality bench)

1. No external draft model (`DRAFT` in `ollama show --modelfile`), or one that loads
   with at least 1 GiB of GPU memory to spare.
2. Loads on the **first** try, 5 times out of 5, with about 1.5 GB held by the desktop,
   after another seat's model, with no out-of-memory line in the journal.
3. Passes 2 on **both** 0.40.16 and the newer Ollama version, or the owner stays pinned.
4. `think = false` is obeyed; `num_ctx = 12288` fits.

Note: `gemma4:26b` also ships an external draft model. It loaded normally on 0.40.17,
but it has the same exposure, so it isn't a safer choice on gate 1 by itself.

## Candidates

Verdicts are from earlier benches (STATUS.md): small samples, read by hand.

| Candidate | Draft model | Known so far | Place in the plan |
|---|---|---|---|
| `granite4.2:30b` (installed) | none, no projector | few inventions; ~80-90 s a turn; a little flat | **First choice to test**: safest load, a family no council uses |
| `qwen3.6:27b` (installed) | none external | passed the estimate-line and repair benches; 65-82 s | Second. Adds no new family where qwen already sits |
| `qwen3.8:27b` (installed) | none external | earlier replay: no clear gain over qwen3.6:27b, one false update marker | Only if qwen3.6 wins phase 2 |
| `muse-glimmer:30b` (installed) | none external | best accuracy, but 140-300 s a turn | Slow fallback where it isn't already seated |
| `flint:40b` (installed) | none | 15 s; some inventions, too many "I update" | Fast fallback |
| `gemma4:26b` (installed) | external | 15-17 s; more inventions; already Contrarian everywhere | Temporary stand-in only |
| `inkling-2-mini` | unknown | not supported on PC by Ollama yet | Add when it is (STATUS.md table) |
| Bolt without its draft | none loaded | `draft_num_predict = 0`: in Ollama's source, not tested; speed unknown | **Control**: may make a replacement unnecessary |

Not retested, already out: `qwen3.6:latest` (invents figures and law), `gpt-oss:120b`
(most inventions), `glm-4.7-flash`, `mistral-small3.2`, `nemotron-cascade-2`, `lfm2.5`
(ignores `think: false`), `mimo-v2.6:9b`, `bonsai-2:27b`. Before phase 2, look once more
at ollama.com's newest list: on 2026-11-24 the only other new local text model of a
usable size was `ripple-preview` (65B dense, coding; likely too slow here).

## Phases

**0. Stay safe now (no GPU).** Pin `ollama ollama-cuda` in `/etc/pacman.conf`. Owner
action. Don't replace anything in a hurry: Bolt works on 0.40.16.

**1. Load gate (~25 min GPU, 0 Claude).** Granite, qwen3.6:27b, flint, muse, Bolt as is,
Bolt with `draft_num_predict = 0`, and Super, on 0.40.16. Record first-try success, GPU
headroom, tok/s and load time from the journal. New script `scripts/load_gate.py`.
Repeat on the newer version only if the owner chooses to upgrade.
*Stop here if Bolt without its draft passes and stays under ~45 s a turn*: add a per-model
`draft_num_predict` option, keep Bolt, and keep the rest of this plan for later. It must
also pass on 0.40.17 or later before unpinning.

**2. Quality bench (~2 h GPU for granite + qwen, 0 Claude, resumable).** Same prompts for
all, always `MODEL=false`. Reuse saved Bolt answers where the prompts are the same.
- `data/bench/m17-estimates.toml` (14 turns): estimate line.
- `data/bench/m18-tags.toml` (16): fact tags, score with `scripts/m18_compliance.py`.
- `data/bench/career-rigor.toml` (9), `econ-macro.toml` (6), `econ-analyst.toml` (4).
- `data/bench/fr-bolt.toml` (3 French turns).
- `scripts/repair_bench.py`, before `repair = true`; `scripts/merge_bench.py` (notes).

**3. Choose.** A model may replace Bolt in a seat if all hold, against Bolt on the same
prompts:
- invented or wrong specifics, read by hand: no more than Bolt's;
- readable estimate line: at least 95 %; tag compliance within 10 points of Bolt's;
- median turn at most 90 s, p90 under 180 s;
- role kept, French fluent, no self-address;
- not the family of another seat in that council, unless a blind-round replay shows no
  echo. Different models may win different seats; economics and geopolitics each have
  two Bolt seats, a chance to split them.
If none passes, keep Bolt on the pinned version and re-run at the next new model.

**4. Live check and switch (~1 h GPU, ~50 Claude calls: ask first).** One short
made-up-question session in economics, purchases and geopolitics, in scratch roots. Then,
seat by seat: persona files, `config.toml` (new entry, `repair`), tests that name Bolt
(`test_fake_and_registry`, `test_doctor`, `test_repair`, `test_dossier`), README table
(it still lists Super for most seats), CHANGELOG. Full check suite. Commit after approval.

**5. Super (optional).** Logistician first (phase 2 method). The local Chair needs
`scripts/chair_bench.py` on `chair_manifest_labels.toml` (Super 12/79, qwen3.6 7/79), so
replacing it costs quality today: keep it unless phase 1 shows a load problem.

**6. Clean up.** Unpin Ollama only after every seated model passes the load gate on the
new version. Remove unused models from disk only with the owner's explicit yes.

## Engine work that helps whichever model wins (separate approval)

- After repeated load failures, stop giving that member the floor and tell the Chair.
- Show a short error instead of raw llama-server output.
- `council doctor --load`: load each seated model once and report failures before a debate.

## Decisions for the owner

1. Is keeping Bolt without its draft model acceptable if it passes, or must Nemotron go?
2. Is ~80-90 s a turn (granite, qwen) acceptable, against Bolt's ~30 s?
3. Is Super in scope?
4. When may phase 1 use the GPU?
