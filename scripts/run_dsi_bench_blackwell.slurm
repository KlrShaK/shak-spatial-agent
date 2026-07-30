#!/usr/bin/env bash
# DSI-Bench evaluation of the simple_v2 agent, on the Blackwell (RTX PRO 6000) nodes.
# Derived from run_simple_v2_blackwell.slurm; same cu128 venv and offline HF setup.
#
#   sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,LIMIT=2 scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5,RESULTS_DIR=/scratch/run,MANIFEST_PATH=/scratch/run/manifest.json,ONLY_FILE=/scratch/question_ids.txt,FORCE=0 scripts/run_dsi_bench_blackwell.slurm
#
# The runner writes one file per question and skips questions already answered, so
# a job that runs out of time is continued by resubmitting the same command. That
# is preferable to sharding: each shard would reload the 14 GB D4RT checkpoint,
# which is the cost the single-load design exists to avoid.

#SBATCH --job-name=d4rt-dsi-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# The site submit filter flattens the typed GPU request and also adds gpupr.4h,
# whose A100 nodes share the cuda13/gpupr features -- so pin to the Blackwell
# nodes explicitly. EPYC_9654 is unique to eu-g7-[001-010] among reachable GPU nodes.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/dsi_bench_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
RESULTS_DIR="${RESULTS_DIR:-d4rt_agent/results/dsi_bench}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/d4rt_agent/results/dsi_bench/manifest.json}"
ONLY_FILE="${ONLY_FILE:-}"
FORCE="${FORCE:-0}"
MODE="${MODE:-agent}"
POINT_MODE="${POINT_MODE:?submit with POINT_MODE=centroid or POINT_MODE=ensemble5}"

if [[ "$POINT_MODE" != "centroid" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Unsupported POINT_MODE=$POINT_MODE" >&2
  exit 2
fi
if [[ "$MODE" != "agent" && "$MODE" != "baseline" ]]; then
  echo "Unsupported MODE=$MODE (expected agent or baseline)" >&2
  exit 2
fi
if [[ "$MODE" == "agent" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Phase 2 agent evaluation requires host-fixed POINT_MODE=ensemble5" >&2
  exit 2
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "Unsupported FORCE=$FORCE (expected 0 or 1)" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest does not exist: $MANIFEST_PATH" >&2
  exit 2
fi

ONLY_IDS=()
if [[ -n "$ONLY_FILE" ]]; then
  if [[ ! -f "$ONLY_FILE" ]]; then
    echo "ONLY_FILE does not exist: $ONLY_FILE" >&2
    exit 2
  fi
  declare -A SEEN_ONLY_IDS=()
  while IFS= read -r raw_id || [[ -n "$raw_id" ]]; do
    question_id="${raw_id#"${raw_id%%[![:space:]]*}"}"
    question_id="${question_id%"${question_id##*[![:space:]]}"}"
    if [[ -z "$question_id" ]]; then
      continue
    fi
    if [[ -n "${SEEN_ONLY_IDS[$question_id]:-}" ]]; then
      echo "Duplicate question ID in ONLY_FILE: $question_id" >&2
      exit 2
    fi
    SEEN_ONLY_IDS["$question_id"]=1
    ONLY_IDS+=("$question_id")
  done < "$ONLY_FILE"
  if [[ "${#ONLY_IDS[@]}" -eq 0 ]]; then
    echo "ONLY_FILE contains no non-empty question IDs: $ONLY_FILE" >&2
    exit 2
  fi
fi

echo "RESULTS_DIR=$RESULTS_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "ONLY_FILE=${ONLY_FILE:-<unset>} (${#ONLY_IDS[@]} selected ID(s))"
echo "FORCE=$FORCE"
echo "MODE=$MODE  immutable POINT_MODE=$POINT_MODE  LIMIT=${LIMIT:-all}"
echo "GROUNDING_TOOL=isolated_qwen bbox_tokens=96 points_tokens=192 seed=42"
echo "CODE_SHA=$(git rev-parse HEAD)"

mkdir -p logs/d4rt_agent "$RESULTS_DIR" "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR

EXTRA=()
if [[ "$MODE" == "baseline" ]]; then
  EXTRA+=(--baseline)
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "${#ONLY_IDS[@]}" -gt 0 ]]; then
  EXTRA+=(--only "${ONLY_IDS[@]}")
fi
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" -m d4rt_agent.dsi_bench_run \
  --results-dir "$RESULTS_DIR" \
  --manifest "$MANIFEST_PATH" \
  --point-mode "$POINT_MODE" \
  --qwen-model "$MODEL_DIR" \
  --d4rt-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-checkpoint checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --d4rt-dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --max-new-tokens 768 \
  --max-steps 14 \
  "${EXTRA[@]}"
