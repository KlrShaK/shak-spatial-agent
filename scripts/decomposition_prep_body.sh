#!/usr/bin/env bash
# Shared body for the decomposition prep launchers (A100 and Blackwell headers differ only in
# their #SBATCH allocation). Sourced, not executed.
#
#   PART=point4d   -> Point4D scene pass + DA3 metric scale
#   PART=qwen_sam3 -> Qwen3-VL parse/name/answer/evidence + SAM3 grounding
#   PART=d4rt      -> D4RT depth vs Point4D depth: a second metres-per-unit per clip
#   PART=d4rt_scene -> D4RT as the 3D backend: scene pass (clips/<slug>/d4rt/scene.npz) + DA3 metric scale on it
#   PART=census    -> Qwen3-VL names the kind of every question the rules did not read (text only)
#   PART=metric_gt -> Point4D / DA3 / D4RT depth against metric ground truth (decomposition/metric_gt_check.py)
#   PLATFORM=a100|blackwell picks the Python for point4d: envs/point4d (torch cu124, A100 only) or
#   envs/d4rt_bw_third_party (torch cu128, runs on Blackwell as well; Point4D is added by PYTHONPATH).
#
# Optional: STAGES="scene", ONLY="<ids>", OUT=<catalog dir>, SHARD=i/n (this job takes every n-th clip).
# Stage outputs are skipped when they already exist, so a resubmission continues.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PART="${PART:?set PART=point4d or PART=qwen_sam3}"
PLATFORM="${PLATFORM:-a100}"
STAGES="${STAGES:-}"
ONLY="${ONLY:-}"
OUT="${OUT:-}"
SHARD="${SHARD:-}"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
POINT4D_REPO=/cluster/work/igp_psr/spanwar/Point4D
cd "$REPO_ROOT"
mkdir -p logs/decomposition

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=1

EXTRA=()
if [[ -n "$STAGES" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--stages $STAGES)
fi
if [[ -n "$ONLY" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--only $ONLY)
fi
[[ -n "$OUT" ]] && EXTRA+=(--out "$OUT")
[[ -n "$SHARD" ]] && EXTRA+=(--shard "$SHARD")

echo "PART=$PART PLATFORM=$PLATFORM STAGES=${STAGES:-<default>} ONLY=${ONLY:-<all>} OUT=${OUT:-<default>} SHARD=${SHARD:-0/1} CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# NPROC>1 runs that many shard processes side by side on the one GPU (the 96 GB Blackwell cards
# hold several, and each process leaves the GPU idle while it decodes video or lifts masks).
NPROC="${NPROC:-1}"
SHARD_I="${SHARD%%/*}"; SHARD_N="${SHARD##*/}"
[[ -z "$SHARD" ]] && { SHARD_I=0; SHARD_N=1; }

run_part() {   # $1 = shard "i/n"
  local args=("${EXTRA[@]}")
  local filtered=()
  local skip=0
  for a in "${args[@]}"; do
    if [[ "$skip" == 1 ]]; then skip=0; continue; fi
    if [[ "$a" == "--shard" ]]; then skip=1; continue; fi
    filtered+=("$a")
  done
  case "$PART" in
    point4d)
      if [[ "$PLATFORM" == "blackwell" ]]; then
        export PYTHONPATH="$REPO_ROOT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
        PY=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
      else
        export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        PY=/cluster/work/igp_psr/spanwar/envs/point4d/bin/python
      fi
      "$PY" -m dyn_dataset_exploration.decomposition.prep_point4d "${filtered[@]}" --shard "$1"
      ;;
    qwen_sam3)
      # Same environment as scripts/agent_v4_body.sh: the third-party venv carries SAM3, and a staged
      # transformers 4.57 tree (copied to node-local scratch) carries Qwen3-VL.
      LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
      MODEL_DIR="$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"
      export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT:$REPO_ROOT/third-party-tools/sam3${PYTHONPATH:+:$PYTHONPATH}"
      /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
        -m dyn_dataset_exploration.decomposition.prep_qwen_sam3 --qwen-model "$MODEL_DIR" "${filtered[@]}" --shard "$1"
      ;;
    d4rt)
      # a second metric scale per clip from D4RT depth (decomposition/prep_d4rt.py)
      export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third-party-tools/Open-D4RT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
      /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
        -m dyn_dataset_exploration.decomposition.prep_d4rt --out "${OUT:?set OUT}" --shard "$1"
      ;;
    d4rt_scene)
      # D4RT as the 3D backend (scene4d.BACKEND="d4rt"): scene pass, then DA3-Nested metric scale on it.
      # STAGES defaults to "scene metric"; the stage list is handled here, not passed through.
      export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third-party-tools/Open-D4RT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
      local common=(--out "${OUT:?set OUT}")
      # shellcheck disable=SC2206
      [[ -n "$ONLY" ]] && common+=(--only $ONLY)
      local d4rt_py=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
      if [[ " ${STAGES:-scene metric} " == *" scene "* ]]; then
        "$d4rt_py" -m dyn_dataset_exploration.decomposition.prep_d4rt_scene "${common[@]}" --shard "$1"
      fi
      if [[ " ${STAGES:-scene metric} " == *" metric "* ]]; then
        "$d4rt_py" -m dyn_dataset_exploration.decomposition.prep_point4d --stages metric --backend d4rt \
          "${common[@]}" --shard "$1"
      fi
      ;;
    census)
      # text-only kind census of all 7000 questions (decomposition/census.py); same environment as qwen_sam3
      LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
      MODEL_DIR="$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"
      export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
        -m dyn_dataset_exploration.decomposition.census qwen --qwen-model "$MODEL_DIR" --shard "$1"
      ;;
    metric_gt)
      # windows come from `metric_gt_check prep` (CPU); one process, the stage list is not passed through
      export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third-party-tools/Open-D4RT:$POINT4D_REPO${PYTHONPATH:+:$PYTHONPATH}"
      /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
        -m dyn_dataset_exploration.decomposition.metric_gt_check run report
      ;;
    *)
      echo "unknown PART=$PART" >&2
      return 2
      ;;
  esac
}

# Copied once, before any process starts: two processes copying into the same directory at the same
# time left one of them importing a half-copied transformers tree.
if [[ "$PART" == "qwen_sam3" || "$PART" == "census" ]]; then
  mkdir -p "${TMPDIR:-/tmp/$USER}/qwen3vl_python"
  cp -a "$REPO_ROOT/.cache/qwen3vl_python/." "${TMPDIR:-/tmp/$USER}/qwen3vl_python/"
fi

# SHARDS="0/6 3/6" runs exactly those sub-shards (to redo failed ones); otherwise NPROC of them.
if [[ -n "${SHARDS:-}" ]]; then
  # shellcheck disable=SC2206
  SUBS=($SHARDS)
else
  SUBS=()
  for ((j = 0; j < NPROC; j++)); do SUBS+=("$((SHARD_I * NPROC + j))/$((SHARD_N * NPROC))"); done
fi

pids=()
for sub in "${SUBS[@]}"; do
  echo "starting process on shard $sub"
  ( run_part "$sub" 2>&1 | sed -u "s|^|[$sub] |" ) &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
echo "decomposition prep ($PART, $PLATFORM) finished with status $status"
exit "$status"
