#!/usr/bin/env bash
# Submit one jev-flow stage on several GPU types and keep whichever job starts first.
# (An explicit exception to the one-job-per-piece-of-work rule, on the user's request.)
#
#   STAGE=caption RUN_DIR=d4rt_agent/results/jev_pilot EXTRA_ARGS="--smoke 30" PROBE_API=1 \
#     SEED_CACHE=<V5 run>/cache scripts/run_jev_flow_race.sh [rtx_4090:1 rtx_2080_ti:1]
#
# The GPU types are the arguments (default: a 4090 and a 2080 Ti). Run it from the repo root.

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${STAGE:?STAGE must be caption or run}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
GPUS=("$@"); [[ ${#GPUS[@]} -gt 0 ]] || GPUS=(rtx_4090:1 rtx_2080_ti:1)
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR/.race"
JOBS_FILE="$RUN_DIR/.race/$ATTEMPT_ID.jobs"
: > "$JOBS_FILE"
for gpu in "${GPUS[@]}"; do
  id="$(sbatch --parsable --gpus="$gpu" --job-name="jev-$STAGE-${gpu%%:*}" \
        --export=ALL,ATTEMPT_ID="$ATTEMPT_ID" scripts/run_jev_flow.slurm)"
  echo "$id" >> "$JOBS_FILE"
  echo "submitted $gpu -> $id"
done
echo "attempt $ATTEMPT_ID: the first to start claims it and cancels the rest ($JOBS_FILE)"
