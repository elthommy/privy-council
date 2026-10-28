# Importing a sharded GGUF into Ollama

Reference for getting a Hugging Face GGUF that is split into shards into Ollama.
First done on 2026-10-24 for Mistral Small 4 (119B MoE, ~6.5B active) at Q4_K_M,
to test it as a new model family. Script:
`scripts/ollama_import_sharded.py`.

## The problem

Hugging Face splits big GGUF files (over ~50 GB) into shards named
`<name>-00001-of-0000N.gguf`. Ollama 0.40.9 can't pull them:

```
$ ollama pull hf.co/bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF:Q4_K_M
Error: pull model manifest: 400: {"error":"This tag is a sharded GGUF. Ollama does not
yet support pulling sharded GGUF via the registry; please use a single-file quantization
tag or \"latest\", or download the shards and merge them locally with `ollama create`
(workaround detailed at https://github.com/ollama/ollama/issues/5245)."}
```

Every Q4 build of Mistral Small 4 is sharded (bartowski Q4_K_M: 2 shards, 72.6 GB;
unsloth UD-Q4_K_M: 3 shards). Only 1–3-bit quants are single files (bartowski IQ3_XXS
49.9 GB, unsloth UD-Q3_K_S 49.6 GB, UD-IQ3_S 44.4 GB). Those pull directly, but 3-bit
invents more, so a bad result wouldn't be a fair verdict on the model. Chosen: Q4_K_M
plus a local merge (user decision).

## Finding the shards

The HF API lists files with their size and LFS sha256 (the download's checksum):

```bash
curl -s "https://huggingface.co/api/models/<repo>/tree/main?recursive=true"
# or one folder: .../tree/main/<folder>
```

bartowski's Q4_K_M, folder `mistralai_Mistral-Small-4-119B-2603-Q4_K_M/`:

| Shard | Bytes | sha256 |
|---|---|---|
| `…-Q4_K_M-00001-of-00002.gguf` | 39,189,441,344 | `2d80b2d3…2e12f47` |
| `…-Q4_K_M-00002-of-00002.gguf` | 33,447,330,432 | `c3c4e085…2fc972da` |

## The merge tool

`llama-gguf-split --merge` from llama.cpp. It isn't packaged with Ollama (Arch
`ollama` / `ollama-cuda` ship no llama.cpp tools), so use a prebuilt release in its
own directory (nothing installed system-wide):

- Releases: `https://api.github.com/repos/ggml-org/llama.cpp/releases`. `releases/latest`
  pointed to a `v0.7.0` tag with no binaries (2026-10-24); the `bNNNNN` tags are the real
  builds, published several times a day as prereleases.
- Take the CPU build, `llama-bNNNNN-bin-ubuntu-x64.tar.gz`: merging needs no GPU, and the
  CUDA builds need a matching `cudart`. Used **b11552** (2026-10-10, two weeks old):
  `~/models-tmp/llama-cpp-b11552/llama-b11552/llama-gguf-split`
  (tarball sha256 `a45a6afd…cb99bb727`). It runs on Arch as is.

## Gotchas

- **Disk, not `/tmp`**: `/tmp` (and the Claude scratchpad under it) is tmpfs, i.e. RAM,
  63 GB here. Shards + merged file need about twice the model size (~145 GB for 72 GB),
  and `ollama create` copies the merged file into `~/.ollama/models/blobs` once more.
  Work dir used: `~/models-tmp/mistral-small-4` (not in the repo).
- **Keep the shard names**: `--merge` takes the *first* shard and finds the others from
  its `-00001-of-0000N` suffix. Renamed files (e.g. `shard-1.gguf`) break it. The script
  saves them under their original names.
- **Resume**: `curl -C -` resumes a cut download. Check sizes and sha256 before merging.
- `--delete-splits` deletes shards during the merge; llama.cpp itself calls it unsafe.
  Delete them after `ollama create` succeeds instead.

## Steps

```bash
uv run python scripts/ollama_import_sharded.py \
    bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF Q4_K_M mistral-small4:119b-q4_K_M \
    --work ~/models-tmp/mistral-small-4 \
    --gguf-split ~/models-tmp/llama-cpp-b11552/llama-b11552/llama-gguf-split
# add --cleanup to delete the shards and the merged file once Ollama has its copy
```

By hand: download each shard with `curl -sSfL -C - -o <original name> <url>`,
`sha256sum`, then `llama-gguf-split --merge <first shard> merged.gguf`, a
`Modelfile` with `FROM /abs/path/merged.gguf`, and `ollama create <name> -f Modelfile`.

## Log (2026-10-24, Mistral Small 4 Q4_K_M)

- Removed `hf.co/DhruvalLabs/Kimi-Linear-48B-A3B-Instruct-GGUF:Q8_0` (52 GB) first.
  441 GB free on `/`.
- `ollama pull` failed with the sharded-GGUF error above.
- Downloaded llama.cpp b11552 (CPU); `llama-gguf-split --help` OK.
- Shard downloads 21:26–21:36 (both shards in parallel, ~115 MB/s in all). By hand,
  I first saved them as `shard-1.gguf` / `shard-2.gguf`, then had to rename them to
  the original names for `--merge` (the script now keeps the names).
- sha256 of both shards matched HF's LFS oids (2 min 20 s).
- `llama-gguf-split --merge`: 2 min 30 s, "merged from 2 split with 579 tensors",
  72,636,771,680 bytes (= the two shards' sum). 306 GB left on `/` at that point.
- `ollama create mistral-small4:119b-q4_K_M` from `Modelfile` (`FROM <abs path>`)
  started 21:42 (disk and CPU only; it doesn't load the model, so it's safe while a
  council session runs). It took **8 min 10 s** (copy into `/var/lib/ollama/blobs`,
  then "parsing GGUF"). `ollama show`: `architecture mistral4`, `parameters 119.0B`,
  `Q4_K_M`, `template source gguf_chat_template`, capabilities `tools`, `completion`
  (no `thinking`, no vision: the mmproj file wasn't imported).
- **`format` probe** (one-off scratch script, ~2 min of GPU): the model was asked to
  break the schema on purpose.
  - `ModeratorDecision` schema: valid JSON that passes `ModeratorDecision` validation,
    106 s including the cold load, 708 tokens.
  - Vote schema with "answer in prose": JSON anyway, 6 s.
  - Bare `format: "json"` with "answer in prose": an endless generation (7,800+ tokens
    at 16 tok/s), killed. The council never sends bare `"json"`, only schemas.

  So Ollama **does** enforce schemas here. Lesson: `template source gguf_chat_template`
  alone doesn't mean `format` is lost. Kimi-Linear's failure was specific to that
  model (or architecture), so always probe rather than infer from `ollama show`.
- Speed: ~16 tok/s generation, `12%/88% CPU/GPU` at `num_ctx` 12288 (like gpt-oss:120b).
- Cleanup: deleted the shards and the merged file (Ollama keeps its own blob); kept
  `Modelfile` and the llama.cpp binary in `~/models-tmp/`. Total time, pull failure
  to probe: about 35 min, most of it download (10.5 min) and `ollama create` (8 min).

## Tips for next time

- Use the script; it keeps the original shard names and checks the sha256.
- Expect for a ~70 GB model: ~10 min download, ~2.5 min sha256, ~2.5 min merge,
  ~8 min `ollama create`. Free disk: ~3× the model size at the peak.
- `ollama create` and the merge don't touch the GPU, so they can run while a council
  session runs; the probe can't.
- Then add a `[ollama.models."<name>"]` entry to `config.toml` and run a `format` probe
  before any bench.
