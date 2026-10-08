#!/usr/bin/env bash
# Submit one jev-flow stage on several GPU types and keep whichever job starts first
# (an explicit exception to the one-job-per-piece-of-work rule, on the user's request).
#
#   STAGE=caption RUN_DIR=d4rt_agent/results/jev_pilot EXTRA_ARGS="--smoke 30" PROBE_API=1 \
#     SEED_CACHE=<V5 run>/cache scripts/run_jev_flow_race.sh [4090 a100 blackwell]
#
# Presets: 4090 | a100 (32 GB) | blackwell | 3090. Default: 4090 a100 blackwell. Run from the repo root.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${STAGE:?STAGE must be caption or run}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
PRESETS=("$@"); [[ ${#PRESETS[@]} -gt 0 ]] || PRESETS=(4090 a100 blackwell)
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR/.race"
JOBS_FILE="$RUN_DIR/.race/$ATTEMPT_ID.jobs"
: > "$JOBS_FILE"
for preset in "${PRESETS[@]}"; do
  case "$preset" in
    4090)      opts=(--gpus=rtx_4090:1) ;;
    3090)      opts=(--gpus=rtx_3090:1) ;;
    a100)      opts=(--gpus=1 --gres=gpumem:32G --constraint="EPYC_7413|EPYC_9354|XeonGold_6240R") ;;
    blackwell) opts=(--partition=cuda13pr.4h --gpus=nvidia_rtx_pro_6000:1 --constraint=EPYC_9654) ;;
    *) echo "unknown preset $preset" >&2; exit 2 ;;
  esac
  id="$(sbatch --parsable "${opts[@]}" --job-name="jev-$STAGE-$preset" \
        --export=ALL,ATTEMPT_ID="$ATTEMPT_ID" scripts/run_jev_flow.slurm)"
  echo "$id" >> "$JOBS_FILE"
  echo "submitted $preset -> $id"
done
echo "attempt $ATTEMPT_ID: the first to start claims it and cancels the rest ($JOBS_FILE)"
