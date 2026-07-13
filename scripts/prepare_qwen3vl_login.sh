#!/usr/bin/env bash
# Prepare all Qwen3-VL dependencies and weights on an internet-connected login node.
# Compute jobs are then strictly offline.

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python
UV=/cluster/home/spanwar/.local/bin/uv
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

mkdir -p "$DEPS_DIR" "$HF_HOME"

"$UV" pip install \
  --python "$D4RT_PYTHON" \
  --target "$DEPS_DIR" \
  --upgrade \
  --no-deps \
  'transformers>=4.57,<5' \
  'accelerate>=1.1,<2' \
  'huggingface-hub>=0.34,<1' \
  'tokenizers>=0.22,<=0.23' \
  'safetensors>=0.4.3' \
  regex \
  sentencepiece

export PYTHONPATH="$DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME

"$D4RT_PYTHON" - <<PY
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="$MODEL")
print(f"Cached $MODEL at {path}")
PY

# Prove that the cache is sufficient without contacting the Hub. Copy the small
# dependency layer to node-local storage to avoid slow package scanning on Lustre.
LOCAL_DEPS="$(mktemp -d /tmp/qwen3vl_python.XXXXXX)"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"
PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$D4RT_PYTHON" - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot = Path(snapshot_download(repo_id="$MODEL", local_files_only=True))
index = snapshot / "model.safetensors.index.json"
shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
assert index.is_file(), index
assert len(shards) == 4, shards
assert all(shard.stat().st_size > 1_000_000_000 for shard in shards), shards
print(f"Offline snapshot preflight passed: {snapshot} ({len(shards)} weight shards).")
PY
