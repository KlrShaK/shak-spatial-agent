#!/usr/bin/env bash
# Pre-stage the SAM3 and Orient-Anything-V2 checkpoints for the traj3d pipeline.
#
#   ./scripts/prepare_traj3d_weights.sh
#
# Run this on a LOGIN node: compute nodes have no internet, and the traj3d SLURM
# script runs with HF_HUB_OFFLINE=1 so every weight must already be in the cache.
#
# Weights land in /cluster/work/igp_psr/spanwar/hf_cache, matching the existing
# DSI-Bench scripts. Deliberately NOT $HF_HOME's interactive default, which points
# at /cluster/scratch -- that filesystem is purged periodically and a purge would
# silently break every queued job.
#
# facebook/sam3 is a gated repo; HF_TOKEN must be exported and approved for it.
# Re-running is cheap: hf_hub_download resolves to the cached blob and returns.

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
export HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache

cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; facebook/sam3 is gated and will 401." >&2
  exit 2
fi

echo "HF_HOME=$HF_HOME"

"$D4RT_PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download

TARGETS = [
    ("facebook/sam3", "sam3.pt"),
    ("facebook/sam3", "config.json"),
    ("Viglong/OriAnyV2_ckpt", "demo_ckpts/rotmod_realrotaug_best.pt"),
]

for repo_id, filename in TARGETS:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    size = os.path.getsize(path)
    print(f"{repo_id}/{filename}\n    -> {path}\n    {size / 1e9:.2f} GB", flush=True)
PY

echo "All traj3d weights staged."
