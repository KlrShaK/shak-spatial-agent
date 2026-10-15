#!/usr/bin/env bash
# Qwen3-VL-8B on Dyn-Bench VQA with the benchmark's official protocol (dyn_dataset_exploration/official_eval.py).
#
#   RUN_DIR=d4rt_agent/results/qwen3vl8b_dynbench_official_<stamp> SHARD=0/2 \
#     sbatch --export=ALL -o logs/d4rt_agent/qwen_official_eval/shard0_%j.out scripts/run_dynbench_official_qwen.slurm
#
# Blackwell by default; for an A100 80GB pass on the command line:
#   -p gpupr.24h --gpus=nvidia_a100_80gb_pcie:1 --constraint="EPYC_7413|EPYC_9354" --exclude=eu-a65-05
# Resumable: resubmit with the same RUN_DIR and SHARD; answered questions are skipped.

#SBATCH --job-name=dynbench-official-qwen
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=24:00:00

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
SHARD="${SHARD:-0/1}"
PY=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
# the venv's transformers is 4.38; the official code needs >= 4.57 (Qwen3-VL): staged tree first on the path
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
mkdir -p "$LOCAL_DEPS" && cp -a "$REPO_ROOT/.cache/qwen3vl_python/." "$LOCAL_DEPS/"
MODEL_DIR="$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "RUN_DIR=$RUN_DIR SHARD=$SHARD MODEL_DIR=$MODEL_DIR CODE_SHA=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -m dyn_dataset_exploration.official_eval --results-dir "$RUN_DIR" --model-path "$MODEL_DIR" --shard "$SHARD"
echo "shard $SHARD finished"
