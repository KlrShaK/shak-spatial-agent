#!/usr/bin/env bash
# Shared environment for a frozen AgentV6 evaluation worker; invoked by its CPU controller.
set -euo pipefail
PLAN="${V6_PLAN:?set the frozen campaign plan}"
REPO_ROOT="$(dirname "$PLAN")/source"
MAIN_REPO=/cluster/work/igp_psr/spanwar/shak-spatial-agent
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
mkdir -p "$LOCAL_DEPS"
cp -a "$MAIN_REPO/.cache/qwen3vl_python/." "$LOCAL_DEPS/"
export HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
export SHAK_THIRD_PARTY="$MAIN_REPO/third-party-tools"
export D4RT_CORE_ROOT="$SHAK_THIRD_PARTY/Open-D4RT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT:/cluster/work/igp_psr/spanwar/Point4D${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
exec /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python -u \
    -m d4rt_agent.agent_v6.evaluation.campaign worker \
    --plan "${V6_PLAN:?}" --stage "${V6_STAGE:?}" --bench "${V6_BENCH:?}"
