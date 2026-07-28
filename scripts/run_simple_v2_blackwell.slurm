#!/usr/bin/env bash
# Blackwell (RTX PRO 6000, sm_120) variant of run_simple_v2_a100.slurm.
# Uses the separate cu128 venv (envs/d4rt_bw) so it never disturbs the A100 env.
# Submit once per immutable host point policy:
#   sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_blackwell.slurm
#   sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_blackwell.slurm

#SBATCH --job-name=d4rt-simple-v2-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# The site submit filter flattens the typed GPU request and also adds gpupr.4h,
# whose A100 nodes share the cuda13/gpupr features -- so pin to the Blackwell
# nodes explicitly. EPYC_9654 is unique to eu-g7-[001-010] among reachable GPU nodes.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/simple_v2_bw_%j.out

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw/bin/python
DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
POINT_MODE="${POINT_MODE:?submit with POINT_MODE=centroid or POINT_MODE=ensemble5}"

if [[ "$POINT_MODE" != "centroid" && "$POINT_MODE" != "ensemble5" ]]; then
  echo "Unsupported POINT_MODE=$POINT_MODE" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p logs/d4rt_agent d4rt_agent/results/basketball_6 "$LOCAL_DEPS"
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

RESULT="d4rt_agent/results/basketball_6/simple_v2_${POINT_MODE}_blackwell.json"

echo "Immutable POINT_MODE=$POINT_MODE"
echo "Result artifact=$RESULT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"$D4RT_PYTHON" -m d4rt_agent.simple_v2 \
  --point-mode "$POINT_MODE" \
  --qwen-model "$MODEL_DIR" \
  --d4rt-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-checkpoint checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --d4rt-dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --max-new-tokens 768 \
  --max-steps 12 \
  --output "$RESULT"
