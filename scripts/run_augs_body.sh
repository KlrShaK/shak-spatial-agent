#!/usr/bin/env bash
# Run one method across the three DSI-Bench augmentations, in one job.
#
# The published leaderboard averages `std`, `reverse`, `hflip` and
# `reverse_hflip`. Those are not re-renders: the video is time-reversed or
# mirrored so the TRUE answer inverts, while the ground-truth letter is held
# fixed by permuting the option text. A model answering from semantic priors
# ("people walk forward") scores well on `std` and fails `reverse`, which is
# exactly the bias the benchmark exists to expose -- and exactly what a
# `std`-only number cannot see.
#
# All three splits run sequentially in a single job so the whole four-way mean
# comes from one code version, and so the race resolves once rather than
# three times.
#
# Environment expected from the submitter:
#   MODE         "agent" (v4) or "control" (tool-free Qwen)
#   RUN_DIR      bookkeeping dir for the race claim; per-split results dirs are
#                derived from MODE and the split name
#   ATTEMPT_ID   token identifying THIS pair of racing jobs

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
cd "$REPO_ROOT"

MODE="${MODE:?MODE must be agent or control}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by the submitter}"
ATTEMPT_ID="${ATTEMPT_ID:?ATTEMPT_ID must be set by the submitter}"
AUGS="${AUGS:-reverse hflip reverse_hflip}"

source "$REPO_ROOT/scripts/race_claim.sh"

# The venv's own transformers is 4.38, which has no Qwen3-VL loader; the staged
# 4.57.6 tree is prepended for both modes. See scripts/agent_v4_body.sh.
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=1
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODEL_DIR="$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" \
             -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR

echo "MODE=$MODE  AUGS=$AUGS"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Each split is independent: one that fails should not cost the other two, and
# per-question resume means a re-submission continues rather than restarts.
FAILED=()
for AUG in $AUGS; do
  echo ""
  echo "================ $MODE / $AUG ================"
  if [[ "$MODE" == "agent" ]]; then
    RESULTS="$REPO_ROOT/d4rt_agent/results/agent_v4_${AUG}"
    CMD=("$D4RT_PYTHON" -m d4rt_agent.agent_v4.run
         --manifest "$RESULTS/manifest.json" --results-dir "$RESULTS"
         --qwen-model "$MODEL_DIR")
  else
    RESULTS="$REPO_ROOT/d4rt_agent/results/control200_${AUG}"
    CMD=("$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run --baseline
         --manifest "$RESULTS/manifest.json" --results-dir "$RESULTS"
         --qwen-model "$MODEL_DIR")
  fi
  if ! "${CMD[@]}"; then
    echo "!!! $MODE / $AUG FAILED, continuing with the remaining splits"
    FAILED+=("$AUG")
  fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "finished with failures in: ${FAILED[*]}"
  exit 1
fi
echo "all splits complete for MODE=$MODE"
