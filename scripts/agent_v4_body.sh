#!/usr/bin/env bash
# Shared body for the agent-v4 launchers.
#
# Sourced by run_agent_v4_a100.slurm and run_agent_v4_blackwell.slurm, which
# differ only in their #SBATCH allocation header. Everything below -- the venv,
# the offline HF cache, the race claim, the command -- is identical on both, and
# keeping it in one file is what stops the two drifting apart.
#
# Environment expected from the submitter:
#   RUN_DIR      results directory; the manifest and answers live under it
#   ATTEMPT_ID   token identifying THIS pair of racing jobs
#   EXTRA_ARGS   optional, passed through to the runner

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
# The third-party venv, because it is the only one carrying SAM3 and the timm
# that Orient-Anything needs.
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache

# ...but that venv's transformers is 4.38, which has no Qwen3-VL loader, and v4
# is the first thing that needs Qwen and the pipeline in ONE process. The DSI
# launchers already solved this: a staged 4.57.6 tree prepended to PYTHONPATH.
# Verified to co-exist with the pipeline -- neither SAM3 nor Orient-Anything
# imports transformers at all, and both still build with the staged
# huggingface_hub 0.36.2 shadowing the venv's 0.26.5.
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"

cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:?RUN_DIR must be set by the submitter}"
ATTEMPT_ID="${ATTEMPT_ID:?ATTEMPT_ID must be set by the submitter}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source "$REPO_ROOT/scripts/race_claim.sh"


# ---------------------------------------------------------------------------
# Offline environment. Compute nodes have no internet; weights must be cached.
# ---------------------------------------------------------------------------
# Copied to node-local scratch rather than read over Lustre: every rank imports
# from it, and the shared filesystem is the wrong place for that.
mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# SAM3 and the VGGT backbone both benefit, and neither needs fp32 matmul.
export NVIDIA_TF32_OVERRIDE=1

echo "RUN_DIR=$RUN_DIR"
echo "ATTEMPT_ID=$ATTEMPT_ID"
echo "EXTRA_ARGS=$EXTRA_ARGS"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# shellcheck disable=SC2086
"$D4RT_PYTHON" -m d4rt_agent.agent_v4.run \
  --results-dir "$RUN_DIR" --qwen-model "$MODEL_DIR" $EXTRA_ARGS

echo "agent_v4 finished: $RUN_DIR"
