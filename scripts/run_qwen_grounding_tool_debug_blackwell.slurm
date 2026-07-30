#!/usr/bin/env bash
# Isolated one-frame Qwen bbox/points smoke test. This launcher never loads D4RT.

#SBATCH --job-name=qwen-ground-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/qwen_grounding_tool_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
RESULTS_DIR="${RESULTS_DIR:?submit with a fresh scratch RESULTS_DIR}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/d4rt_agent/results/dsi_bench/manifest.json}"
REQUESTS_PATH="${REQUESTS_PATH:-$REPO_ROOT/d4rt_agent/fixtures/qwen_grounding_tool_smoke.json}"
FORCE="${FORCE:-0}"

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "Unsupported FORCE=$FORCE (expected 0 or 1)" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST_PATH" || ! -f "$REQUESTS_PATH" ]]; then
  echo "Manifest or request fixture does not exist" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p logs/d4rt_agent "$RESULTS_DIR" "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing" >&2
  exit 2
fi

echo "RESULTS_DIR=$RESULTS_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "REQUESTS_PATH=$REQUESTS_PATH"
echo "FORCE=$FORCE"
echo "CODE_SHA=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

EXTRA=()
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi

"$D4RT_PYTHON" -m d4rt_agent.qwen_grounding_tool_debug \
  --manifest "$MANIFEST_PATH" \
  --requests "$REQUESTS_PATH" \
  --results-dir "$RESULTS_DIR" \
  --qwen-model "$MODEL_DIR" \
  --seed 42 \
  "${EXTRA[@]}"
