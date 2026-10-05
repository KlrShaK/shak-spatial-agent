#!/usr/bin/env bash
# Shared body for the agent-v5 launchers (run_agent_v5_{a100,blackwell}.slurm and the smoke test).
#
# Environment expected from the submitter:
#   RUN_DIR      results directory: answers/, cache/ (clips, scenes, groundings), run_config.json
#   ATTEMPT_ID   token identifying THIS pair of racing jobs (unset: no race, e.g. the smoke test)
#   BACKBONE     point4d (default) | d4rt -- the 4D model behind every geometric tool
#   BENCH        dynbench (default) | dsi
#   EXTRA_ARGS   optional, passed through to the runner (e.g. --limit 40, --shard 0/4, --all)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# The checkout with the staged Qwen deps and the third-party checkpoints. A git worktree of the
# repo has neither (submodules and caches are not checked out), so it borrows them from here.
MAIN_REPO="${MAIN_REPO:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
POINT4D_REPO="${POINT4D_REPO:-/cluster/work/igp_psr/spanwar/Point4D}"

BACKBONE="${BACKBONE:-point4d}"
BENCH="${BENCH:-dynbench}"
case "$BACKBONE" in point4d|d4rt) ;; *) echo "BACKBONE must be point4d or d4rt, got '$BACKBONE'" >&2; exit 2;; esac
case "$BENCH" in dynbench|dsi) ;; *) echo "BENCH must be dynbench or dsi, got '$BENCH'" >&2; exit 2;; esac

pick() {  # the first of the two paths that exists
  if [[ -e "$1" ]]; then echo "$1"; else echo "$2"; fi
}
DEPS_DIR="$(pick "$REPO_ROOT/.cache/qwen3vl_python" "$MAIN_REPO/.cache/qwen3vl_python")"
SHAK_THIRD_PARTY="$(pick "$REPO_ROOT/third-party-tools/Orient-Anything-V2/vision_tower.py" \
                         "$MAIN_REPO/third-party-tools/Orient-Anything-V2/vision_tower.py")"
SHAK_THIRD_PARTY="$(dirname "$(dirname "$SHAK_THIRD_PARTY")")"
D4RT_CORE_ROOT="${D4RT_CORE_ROOT:-$SHAK_THIRD_PARTY/Open-D4RT}"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
MODEL_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"

cd "$REPO_ROOT"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set by the submitter}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "${ATTEMPT_ID:-}" ]]; then
  source "$REPO_ROOT/scripts/race_claim.sh"
fi

# Offline environment: compute nodes have no internet; weights must be cached.
# The venv's transformers (4.38) has no Qwen3-VL loader: a staged 4.57 tree goes first on the path,
# copied to node-local scratch (every import reads it).
mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"
MODEL_DIR="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Offline Qwen snapshot missing. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
export MODEL_DIR HF_HOME SHAK_THIRD_PARTY D4RT_CORE_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1
export CODE_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "RUN_DIR=$RUN_DIR BACKBONE=$BACKBONE BENCH=$BENCH"
echo "EXTRA_ARGS=$EXTRA_ARGS"
echo "CODE_SHA=$CODE_SHA THIRD_PARTY=$SHAK_THIRD_PARTY"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# shellcheck disable=SC2086
"$D4RT_PYTHON" -m d4rt_agent.agent_v5.run \
  --bench "$BENCH" --backbone "$BACKBONE" --results-dir "$RUN_DIR" --qwen-model "$MODEL_DIR" $EXTRA_ARGS

echo "agent_v5 finished: $RUN_DIR"
"$D4RT_PYTHON" -m d4rt_agent.agent_v5.report "$RUN_DIR" --out "$RUN_DIR/report.md" | head -20
