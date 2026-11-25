# Before upgrading Ollama: run the load gate

A new Ollama version can stop a seated model from loading: 0.40.17 broke
`nemotron-3.5-bolt` on 2026-11-24 and silenced its members for a whole session.
`scripts/load_gate.py` shows it beforehand. No Claude calls, about 30 min of GPU. It
unloads every model as it goes: don't run it during a session.

1. **Before** the upgrade, get a reference (Ollama 0.40.16 was the last version used):

   ```bash
   uv run python -u scripts/load_gate.py \
     --prompts-from data/bench/runs/m18-vouch.jsonl \
     --out data/bench/runs/load-gate-upgrade.jsonl --runs 3 \
     --models nemotron-3.5-bolt:latest+nodraft qwen3.6:27b muse-glimmer:30b nemotron-3.5-super:latest
   # gemma4 is the model loaded first above, so it needs its own run
   uv run python -u scripts/load_gate.py \
     --prompts-from data/bench/runs/m18-vouch.jsonl \
     --out data/bench/runs/load-gate-upgrade.jsonl --runs 3 \
     --primer qwen3.6:27b --models gemma4:26b
   ```

2. **Upgrade** (on Arch, first comment out `IgnorePkg = ollama ollama-cuda` in
   `/etc/pacman.conf`), then `sudo systemctl restart ollama`.
3. **Run the same two commands again.** Results are kept per Ollama version.
4. **Read the summary**: every model should show `pass 3/3`, `starts [1, 1, 1]` and
   `oom 0`. Two starts mean Ollama hid a failed load behind its own retry. Compare the
   turn times and the free memory with the earlier version.
5. **If a model fails**, go back, and put `IgnorePkg` back:

   ```bash
   sudo pacman -U /var/cache/pacman/pkg/ollama-OLD-x86_64.pkg.tar.zst \
                  /var/cache/pacman/pkg/ollama-cuda-OLD-x86_64.pkg.tar.zst
   sudo systemctl restart ollama
   ```

Notes:

- `+nodraft` tests Bolt without its draft model, as `config.toml` runs it
  (`draft_num_predict = 0`). The script doesn't read that option: without the suffix
  Bolt fails the gate.
- If another model's seat changes, update the lists above.
- To test a version without installing it, run its server on another port and add
  `--url http://localhost:PORT --log ITS_LOG_FILE`.
- The prompts and answers are real sessions: keep them in `data/bench/` (gitignored).
- The gate checks loading and speed on single turns, not answer quality.
