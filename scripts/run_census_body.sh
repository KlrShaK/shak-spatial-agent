#!/usr/bin/env bash
# Agent v4 over the full DSI-Bench census: 1769 questions x 4 augmentations.
#
# ~7,076 questions at ~40 s each is roughly 80 GPU-hours, which does not fit one
# 24 h slot. Two mechanisms cover that:
#
#   * the runner writes one file per question and skips answered ones, so any
#     restart continues rather than repeats;
#   * this body leaves a MARKER naming its progress before it exits, and
#     scripts/census_monitor.sh -- running on the login node -- submits the
#     next attempt once it sees the marker and no job left running. This job
#     never calls sbatch itself: submitting from a compute node once let the
#     site filter silently append gpupr.24h to the partition list, which
#     produced a BadConstraints job that pended forever (see run_census.sh).
#     All submission happens from the login node now.
#
# `--lean` is not optional: of ~10.9 MB of artifacts per question only ~144 KB is
# the measurement, and keeping the rest for 7,076 questions would add ~150 GB to
# a filesystem already at 96%.
set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:?RUN_DIR must be set by the submitter}"
ATTEMPT_ID="${ATTEMPT_ID:?ATTEMPT_ID must be set by the submitter}"
AUGS="${AUGS:-std reverse hflip reverse_hflip}"
RACE_DIR="$REPO_ROOT/d4rt_agent/results/_census_race"

source "$REPO_ROOT/scripts/race_claim.sh"

DEPS_DIR="$REPO_ROOT/.cache/qwen3vl_python"
LOCAL_DEPS="${TMPDIR:-/tmp/$USER}/qwen3vl_python"
mkdir -p "$LOCAL_DEPS"
cp -a "$DEPS_DIR/." "$LOCAL_DEPS/"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=1
export PYTHONPATH="$LOCAL_DEPS:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODEL_DIR="$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" \
             -mindepth 1 -maxdepth 1 -type d -print -quit)"
[[ -n "$MODEL_DIR" ]] || { echo "Offline Qwen snapshot missing." >&2; exit 2; }
export MODEL_DIR

# Recorded before any work, so we can tell at the end whether this attempt
# actually accomplished anything.
START_TOTAL=0
for A in $AUGS; do
  START_TOTAL=$((START_TOTAL + $(find "$REPO_ROOT/d4rt_agent/results/census/${A}/answers" \
                  -name '*.json' 2>/dev/null | wc -l || true)))
done

echo "AUGS=$AUGS  START_TOTAL=$START_TOTAL"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Budget: ask SLURM how long this job actually has, and stop 20 min short so the
# last question completes and the chain marker gets written inside the allocation.
BUDGET=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null || true)
  if [[ -n "$LEFT" ]]; then
    BUDGET=$($D4RT_PYTHON - "$LEFT" <<'PYEOF'
import re, sys
t = sys.argv[1].strip()                        # [DD-]HH:MM:SS or MM:SS
m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", t)
if not m:
    print(0); raise SystemExit
d, h, mi, s = (int(x) if x else 0 for x in m.groups())
print(max(0, d*86400 + h*3600 + mi*60 + s - 1200))
PYEOF
)
  fi
fi
echo "time budget for this job: ${BUDGET}s"

started=$(date +%s)
FAILED=()
for AUG in $AUGS; do
  spent=$(( $(date +%s) - started ))
  remaining=$(( BUDGET - spent ))
  if [[ "$BUDGET" -gt 0 && "$remaining" -lt 300 ]]; then
    echo "budget spent; stopping before $AUG"
    break
  fi
  RESULTS="$REPO_ROOT/d4rt_agent/results/census/${AUG}"
  done_n=$(find "$RESULTS/answers" -name '*.json' 2>/dev/null | wc -l || true)
  echo ""
  echo "================ census / $AUG  ($done_n already answered) ================"
  EXTRA=()
  [[ "$BUDGET" -gt 0 ]] && EXTRA+=(--max-seconds "$remaining")
  if ! "$D4RT_PYTHON" -m d4rt_agent.agent_v4.run \
        --manifest "$RESULTS/manifest.json" --results-dir "$RESULTS" \
        --qwen-model "$MODEL_DIR" --lean "${EXTRA[@]}"; then
    echo "!!! census/$AUG exited non-zero"
    FAILED+=("$AUG")
  fi
done

echo ""
echo "=== progress ==="
total=0
for AUG in $AUGS; do
  n=$(find "$REPO_ROOT/d4rt_agent/results/census/${AUG}/answers" -name '*.json' 2>/dev/null | wc -l || true)
  printf "  %-16s %5d/1769\n" "$AUG" "$n"; total=$((total + n))
done
echo "  TOTAL            $total/7076"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  splits that exited non-zero: ${FAILED[*]}"

# ---------------------------------------------------------------------------
# Leave a marker for the login-node monitor instead of submitting anything
# ourselves. Guards mirror the old in-job chain: work must remain, and this
# attempt must have actually progressed -- a marker left after doing nothing
# would have the monitor spin through the queue burning slots.
# ---------------------------------------------------------------------------
mkdir -p "$RACE_DIR"
if [[ "$total" -ge 7076 ]]; then
  echo "CENSUS COMPLETE"
  rm -f "$RACE_DIR/.needs_chain"
elif [[ "$total" -gt "$START_TOTAL" ]]; then
  echo "$total" > "$RACE_DIR/.needs_chain"
  echo "wrote chain marker ($total/7076); scripts/census_monitor.sh will submit the next attempt from the login node"
else
  echo "NO PROGRESS this attempt ($total answered, started at $START_TOTAL); not leaving a chain marker."
  echo "Investigate before continuing -- the monitor will not resubmit blindly either."
  exit 1
fi
exit 0
