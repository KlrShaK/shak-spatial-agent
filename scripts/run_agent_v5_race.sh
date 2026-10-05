#!/usr/bin/env bash
# Submit agent v5 to BOTH partitions (A100 80GB, Blackwell) and keep whichever starts first.
#
#   BACKBONE=point4d ./scripts/run_agent_v5_race.sh --limit 200          # first 200 of the eval split
#   BACKBONE=d4rt    ./scripts/run_agent_v5_race.sh                      # the whole 2800-question eval split
#   BENCH=dsi        ./scripts/run_agent_v5_race.sh --split std --random-sample 200
#   RUN_DIR=d4rt_agent/results/agent_v5_point4d_<stamp> ./scripts/run_agent_v5_race.sh   # resume
#
# Arguments are passed to `python -m d4rt_agent.agent_v5.run`. Resuming: pass the same RUN_DIR; one
# answer file per question, existing ones are skipped. Several shards in parallel: give each its own
# `--shard i/n` (same RUN_DIR is fine -- shards touch disjoint clips).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_REPO="${MAIN_REPO:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
BACKBONE="${BACKBONE:-point4d}"
BENCH="${BENCH:-dynbench}"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-d4rt_agent/results/agent_v5_${BENCH}_${BACKBONE}_$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EXTRA_ARGS="$*"

# Checked on the login node, before queueing.
for cache in "$HF_HOME/hub/models--facebook--sam3" "$HF_HOME/hub/models--Viglong--OriAnyV2_ckpt"; do
  [[ -d "$cache" ]] || { echo "Missing cached weights: $cache (run scripts/prepare_traj3d_weights.sh)" >&2; exit 2; }
done
if [[ "$BACKBONE" == "point4d" ]]; then
  [[ -d "$HF_HOME/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1" ]] \
    || { echo "Missing DA3-Nested weights in $HF_HOME (Point4D's metric scale)" >&2; exit 2; }
  [[ -f /cluster/work/igp_psr/spanwar/Point4D/checkpoints/point4d_final.pt ]] \
    || { echo "Missing Point4D checkpoint" >&2; exit 2; }
fi
if [[ "$BACKBONE" == "d4rt" ]]; then
  D4RT="${D4RT_CORE_ROOT:-$MAIN_REPO/third-party-tools/Open-D4RT}"
  [[ -f "$D4RT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" ]] \
    || { echo "Missing D4RT checkpoint under $D4RT" >&2; exit 2; }
fi
[[ -d "$MAIN_REPO/.cache/qwen3vl_python/transformers" || -d "$REPO_ROOT/.cache/qwen3vl_python/transformers" ]] \
  || { echo "Missing staged Qwen3-VL deps (run scripts/prepare_qwen3vl_login.sh)" >&2; exit 2; }

mkdir -p "$MAIN_REPO/logs/d4rt_agent" "$RUN_DIR/.race"
JOBS_FILE="$RUN_DIR/.race/$ATTEMPT_ID.jobs"
echo "RUN_DIR=$RUN_DIR BACKBONE=$BACKBONE BENCH=$BENCH ATTEMPT_ID=$ATTEMPT_ID"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<none>}"

submit() {
  sbatch --parsable \
    --export=ALL,RUN_DIR="$RUN_DIR",ATTEMPT_ID="$ATTEMPT_ID",EXTRA_ARGS="$EXTRA_ARGS",BACKBONE="$BACKBONE",BENCH="$BENCH" \
    "$1"
}
JOB_A100="$(submit scripts/run_agent_v5_a100.slurm)"
JOB_BW="$(submit scripts/run_agent_v5_blackwell.slurm)"
printf '%s\n%s\n' "$JOB_A100" "$JOB_BW" > "$JOBS_FILE"
echo "submitted a100=$JOB_A100 blackwell=$JOB_BW; the first to start claims the attempt and cancels the other"
echo "  squeue -j $JOB_A100,$JOB_BW"
echo "  tail -f $MAIN_REPO/logs/d4rt_agent/agentv5_{a100,bw}_*.out"
