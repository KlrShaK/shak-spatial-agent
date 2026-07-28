#!/usr/bin/env bash
# Plain one-image Qwen grounding probe over the saved DSI-Bench agent traces.
#
#   sbatch scripts/run_qwen_grounding_debug_a100.slurm

#SBATCH --job-name=qwen-ground-debug
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/qwen_grounding_debug_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
RESULTS_DIR=d4rt_agent/results/qwen_grounding_debug

cd "$REPO_ROOT"
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
export QWEN_DEVICE_MAP=cuda

MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" -m d4rt_agent.qwen_grounding_debug \
  --answers-dir d4rt_agent/results/dsi_bench/answers \
  --results-dir "$RESULTS_DIR" \
  --qwen-model "$MODEL_DIR" \
  --seed 42 \
  --max-new-tokens 96
