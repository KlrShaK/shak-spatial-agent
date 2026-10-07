#!/usr/bin/env bash
# Shared body for the jev-flow launcher (run_jev_flow_a100.slurm).
#
# Environment expected from the submitter:
#   STAGE        caption | run   -- caption: Qwen3-VL-2B describes each clip once; run: the Decisions-driven agent
#   RUN_DIR      results directory: captions/, decisions/, answers/, cache/, run_config.json
#   BACKBONE     point4d (default) | d4rt
#   SEED_CACHE   a finished AgentV5 run's cache dir whose scenes and groundings are linked in (read-only)
#   API_MODE     live (default) | collect | cached   (run stage)
#   EXTRA_ARGS   passed to the runner (e.g. --smoke 25, --limit 40, --shard 0/4)
#
# The caption stage is offline (local weights). The run stage calls the OpenAI Decisions API, so it
# loads the ETH proxy module unless USE_PROXY=0.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAIN_REPO="${MAIN_REPO:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
POINT4D_REPO="${POINT4D_REPO:-/cluster/work/igp_psr/spanwar/Point4D}"

STAGE="${STAGE:?STAGE must be caption or run}"
BACKBONE="${BACKBONE:-point4d}"
API_MODE="${API_MODE:-live}"
case "$STAGE" in caption|run) ;; *) echo "STAGE must be caption or run, got '$STAGE'" >&2; exit 2;; esac
case "$BACKBONE" in point4d|d4rt) ;; *) echo "BACKBONE must be point4d or d4rt, got '$BACKBONE'" >&2; exit 2;; esac

pick() { if [[ -e "$1" ]]; then echo "$1"; else echo "$2"; fi; }
DEPS_DIR="$(pick "$REPO_ROOT/.cache/qwen3vl_python" "$MAIN_REPO/.cache/qwen3vl_python")"
SHAK_THIRD_PARTY="$(pick "$REPO_ROOT/third-party-tools/Orient-Anything-V2/vision_tower.py" \
                         "$MAIN_REPO/third-party-tools/Orient-Anything-V2/vision_tower.py")"
SHAK_THIRD_PARTY="$(dirname "$(dirname "$SHAK_THIRD_PARTY")")"
D4RT_CORE_ROOT="${D4RT_CORE_ROOT:-$SHAK_THIRD_PARTY/Open-D4RT}"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
CAPTION_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"

cd "$REPO_ROOT"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by the submitter}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"
CAPTION_MODEL="$(find "$CAPTION_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ "$STAGE" == "caption" && -z "$CAPTION_MODEL" ]]; then
  echo "Qwen3-VL-2B snapshot missing under $CAPTION_CACHE (stage it on a login node first)." >&2
  exit 2
fi
export HF_HOME SHAK_THIRD_PARTY D4RT_CORE_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1
export CODE_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [[ "$STAGE" == "run" && "$API_MODE" != "cached" && "${USE_PROXY:-1}" == "1" ]]; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load eth_proxy
  echo "https_proxy=${https_proxy:-unset}"
fi

if [[ "${PROBE_API:-0}" == "1" ]]; then   # can this node reach the Decisions API, directly or through the ETH proxy?
  body='{"model":"gpt-6-luna","input":"A person walks left to right while the camera stays still.","questions":[{"type":"choice","name":"cam","instructions":"Does the camera move?","choices":[{"value":"A","description":"stationary"},{"value":"B","description":"moves"}]}]}'
  probe() { curl -s --max-time 40 -o /dev/null -w "HTTP %{http_code}\n" https://api.openai.com/v1/decisions \
              -H "Authorization: Bearer ${OPENAI_API_KEY:-none}" -H "Content-Type: application/json" -d "$body" || echo "no connection"; }
  echo "API probe, direct:    $(probe)"
  ( source /etc/profile.d/modules.sh 2>/dev/null || true; module load eth_proxy
    echo "API probe, via proxy: $(probe)" )
fi

echo "STAGE=$STAGE RUN_DIR=$RUN_DIR BACKBONE=$BACKBONE API_MODE=$API_MODE"
echo "EXTRA_ARGS=$EXTRA_ARGS CODE_SHA=$CODE_SHA"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

SEED=()
[[ -n "${SEED_CACHE:-}" ]] && SEED=(--seed-cache "$SEED_CACHE")
if [[ "$STAGE" == "caption" ]]; then
  # shellcheck disable=SC2086
  "$D4RT_PYTHON" -m d4rt_agent.agent_v5_jev.caption --backbone "$BACKBONE" --results-dir "$RUN_DIR" \
    --caption-model "$CAPTION_MODEL" "${SEED[@]}" $EXTRA_ARGS
else
  # shellcheck disable=SC2086
  "$D4RT_PYTHON" -m d4rt_agent.agent_v5_jev.run --backbone "$BACKBONE" --results-dir "$RUN_DIR" \
    --api-mode "$API_MODE" "${SEED[@]}" $EXTRA_ARGS
fi
echo "jev-flow $STAGE finished: $RUN_DIR"
