#!/usr/bin/env bash
# Submit agent v4 to BOTH partitions and keep whichever starts first.
#
#   ./scripts/run_agent_v4_race.sh                       # the 25-question sample
#   ./scripts/run_agent_v4_race.sh --random-sample 200   # the uniform 200
#   ./scripts/run_agent_v4_race.sh --limit 1             # smoke test
#   RUN_DIR=d4rt_agent/results/agent_v4 ./scripts/run_agent_v4_race.sh   # resume
#
# Queue waits on one partition routinely exceed the job's own runtime, and there
# is nothing to gain by waiting them out when the other partition is free. Both
# jobs are submitted; the first to start claims the attempt and cancels the
# other. See the claim protocol in scripts/agent_v4_body.sh.
#
# RESUMING: pass the same RUN_DIR. The runner writes one file per question and
# skips questions already answered, so resubmitting the identical command
# continues where the time limit cut it off. Each submission mints a fresh
# ATTEMPT_ID, so the two jobs race again rather than both losing to the previous
# attempt's claim.
#
# Weights must already be cached -- compute nodes have no internet and the job
# runs with HF_HUB_OFFLINE=1. Run scripts/prepare_traj3d_weights.sh once on a
# login node first.

set -euo pipefail

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
cd "$REPO_ROOT"

# Minted here rather than inside the job so that a resubmission continues the
# same run instead of silently starting a fresh one.
RUN_DIR="${RUN_DIR:-d4rt_agent/results/agent_v4_$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EXTRA_ARGS="$*"

# Validated on the login node, before queueing: a missing checkpoint should fail
# now rather than after hours in the queue.
for cache in \
    "$HF_HOME/hub/models--facebook--sam3" \
    "$HF_HOME/hub/models--Viglong--OriAnyV2_ckpt"; do
  if [[ ! -d "$cache" ]]; then
    echo "Missing cached weights: $cache" >&2
    echo "Run ./scripts/prepare_traj3d_weights.sh on a login node first." >&2
    exit 2
  fi
done
for weight in \
    "$REPO_ROOT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
    "$REPO_ROOT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"; do
  if [[ ! -f "$weight" ]]; then
    echo "Missing D4RT weight: $weight" >&2
    exit 2
  fi
done

# v4 needs Qwen3-VL and the SAM3/OAV2 pipeline in one process, and the venv's
# own transformers is too old for the former. Checked here so an unstaged tree
# fails on the login node instead of after hours in the queue.
if [[ ! -d "$REPO_ROOT/.cache/qwen3vl_python/transformers" ]]; then
  echo "Missing staged Qwen3-VL deps: $REPO_ROOT/.cache/qwen3vl_python" >&2
  echo "Run ./scripts/prepare_qwen3vl_login.sh on a login node first." >&2
  exit 2
fi
if [[ -z "$(find "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" \
            -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)" ]]; then
  echo "Missing offline Qwen3-VL snapshot under $HF_HOME" >&2
  echo "Run ./scripts/prepare_qwen3vl_login.sh on a login node first." >&2
  exit 2
fi

mkdir -p logs/d4rt_agent "$RUN_DIR/.race"
RACE_DIR="$RUN_DIR/.race"
JOBS_FILE="$RACE_DIR/$ATTEMPT_ID.jobs"

echo "RUN_DIR=$RUN_DIR"
echo "ATTEMPT_ID=$ATTEMPT_ID"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

if [[ -d "$RACE_DIR/$ATTEMPT_ID.claim" ]]; then
  echo "attempt $ATTEMPT_ID is already claimed -- refusing to submit" >&2
  exit 2
fi

submit() {
  sbatch --parsable \
    --export=ALL,RUN_DIR="$RUN_DIR",ATTEMPT_ID="$ATTEMPT_ID",EXTRA_ARGS="$EXTRA_ARGS" \
    "$1"
}

JOB_A100="$(submit scripts/run_agent_v4_a100.slurm)"
JOB_BW="$(submit scripts/run_agent_v4_blackwell.slurm)"

# Written after both submits so the winner can cancel the loser. A job that
# starts before this lands simply finds no file and skips the cancellation; the
# loser's own claim check still stops it doing any work.
printf '%s\n%s\n' "$JOB_A100" "$JOB_BW" > "$JOBS_FILE"

echo
echo "submitted a100=$JOB_A100  blackwell=$JOB_BW"
echo "the first to start claims the attempt and cancels the other"
echo
echo "  squeue -j $JOB_A100,$JOB_BW"
echo "  tail -f logs/d4rt_agent/agentv4_{a100,bw}_*.out"
