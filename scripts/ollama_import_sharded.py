"""Import a sharded Hugging Face GGUF into Ollama, which can't pull one itself.

Ollama (0.40.9) refuses `ollama pull hf.co/...:<quant>` when the quant is split
into `-0000N-of-0000M.gguf` shards (HF splits files over ~50 GB). Workaround
(ollama/ollama#5245): download the shards, merge them with llama.cpp's
`llama-gguf-split --merge`, then `ollama create` from the merged file.
Full notes: docs/ollama-sharded-gguf.md.

    uv run python scripts/ollama_import_sharded.py \\
        bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF Q4_K_M mistral-small4:119b-q4_K_M \\
        --work ~/models-tmp/mistral-small-4 \\
        --gguf-split ~/models-tmp/llama-cpp-b11552/llama-b11552/llama-gguf-split

Steps are skipped when already done (sizes and sha256 match), so a rerun resumes.
The work dir must be on disk: /tmp is RAM-backed here, and the merge needs about twice the
model size free (shards + merged file); `ollama create` copies it once more.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import httpx

HF = "https://huggingface.co"
CHUNK = 64 * 1024 * 1024


def list_shards(repo: str, quant: str) -> list[dict[str, object]]:
    """Return the repo's shard entries for one quant (path, size, lfs sha256), in order."""
    url = f"{HF}/api/models/{repo}/tree/main?recursive=true"
    files = httpx.get(url, timeout=60).raise_for_status().json()
    shards = [
        f
        for f in files
        if f["path"].endswith(".gguf") and f"-{quant}-" in f["path"] and "-of-" in f["path"]
    ]
    if not shards:
        raise SystemExit(f"no sharded {quant} GGUF in {repo}; try `ollama pull` directly")
    return sorted(shards, key=lambda f: f["path"])


def sha256(path: Path) -> str:
    """Hash a file in large chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo: str, shard: dict[str, object], work: Path) -> Path:
    """Download one shard under its original name (merge needs it), resuming with curl."""
    rel = str(shard["path"])
    dest = work / Path(rel).name
    if not (dest.exists() and dest.stat().st_size == shard["size"]):
        url = f"{HF}/{repo}/resolve/main/{rel}"
        subprocess.run(["curl", "-sSfL", "-C", "-", "-o", str(dest), url], check=True)
    oid = shard.get("lfs", {}).get("oid")  # type: ignore[union-attr]
    if oid and sha256(dest) != oid:
        raise SystemExit(f"sha256 mismatch for {dest.name}; delete it and rerun")
    return dest


def merge(gguf_split: Path, first: Path, out: Path) -> None:
    """Merge the shards into one GGUF; llama-gguf-split finds the rest from FIRST's name."""
    if out.exists():
        return
    subprocess.run([str(gguf_split), "--merge", str(first), str(out)], check=True)


def create(name: str, merged: Path, work: Path) -> None:
    """Register the merged GGUF with Ollama (the chat template comes from the GGUF)."""
    modelfile = work / "Modelfile"
    modelfile.write_text(f"FROM {merged}\n")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)


def main() -> None:
    """Download, verify, merge and import; optionally delete the big files after."""
    ap = argparse.ArgumentParser(description="Import a sharded HF GGUF into Ollama.")
    ap.add_argument("repo")
    ap.add_argument("quant")
    ap.add_argument("name", help="Ollama model name to create")
    ap.add_argument("--work", type=Path, required=True, help="dir on disk, not /tmp")
    ap.add_argument("--gguf-split", type=Path, required=True)
    ap.add_argument("--cleanup", action="store_true", help="delete shards and merged file")
    args = ap.parse_args()
    work = args.work.expanduser()
    work.mkdir(parents=True, exist_ok=True)
    paths = [download(args.repo, s, work) for s in list_shards(args.repo, args.quant)]
    merged = work / f"{args.name.replace(':', '-').replace('/', '-')}.gguf"
    merge(args.gguf_split.expanduser(), paths[0], merged)
    create(args.name, merged, work)
    if args.cleanup:
        for path in [*paths, merged]:
            path.unlink()


if __name__ == "__main__":
    main()
