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

# Prove that the cache is sufficient without contacting the Hub.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$D4RT_PYTHON" - <<PY
from transformers import AutoConfig, AutoProcessor

AutoConfig.from_pretrained("$MODEL", local_files_only=True)
AutoProcessor.from_pretrained("$MODEL", local_files_only=True)
print("Offline model and processor preflight passed.")
PY
