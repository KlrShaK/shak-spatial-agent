#!/usr/bin/env bash
# Race one method across all three augmentations.
#
#   ./scripts/run_augs_race.sh control     # ~30 min
#   ./scripts/run_augs_race.sh agent       # ~6.5 h
#
# Two jobs per call (A100 + Blackwell); the first to start claims the attempt
# and cancels the other, exactly as scripts/run_agent_v4_race.sh does.
set -euo pipefail
REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache
cd "$REPO_ROOT"

MODE="${1:?usage: run_augs_race.sh control OR run_augs_race.sh agent}"
case "$MODE" in control|agent) ;; *) echo "MODE must be control or agent" >&2; exit 2 ;; esac

RUN_DIR="d4rt_agent/results/_augs_race_${MODE}"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

if [[ ! -d "$REPO_ROOT/.cache/qwen3vl_python/transformers" ]]; then
  echo "Missing staged Qwen3-VL deps. Run scripts/prepare_qwen3vl_login.sh first." >&2
  exit 2
fi
for aug in reverse hflip reverse_hflip; do
  dir=$([[ "$MODE" == agent ]] && echo "agent_v4_${aug}" || echo "control200_${aug}")
  if [[ ! -f "$REPO_ROOT/d4rt_agent/results/$dir/manifest.json" ]]; then
    echo "Missing manifest: d4rt_agent/results/$dir/manifest.json" >&2
    exit 2
  fi
done

mkdir -p logs/d4rt_agent "$RUN_DIR/.race"
JOBS_FILE="$RUN_DIR/.race/$ATTEMPT_ID.jobs"
echo "MODE=$MODE  RUN_DIR=$RUN_DIR  ATTEMPT_ID=$ATTEMPT_ID"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

submit() {
  sbatch --parsable --export=ALL,MODE="$MODE",RUN_DIR="$RUN_DIR",ATTEMPT_ID="$ATTEMPT_ID" "$1"
}
if [[ "$MODE" == agent ]]; then
  A=$(submit scripts/run_agent_v4_augs_a100.slurm)
  B=$(submit scripts/run_agent_v4_augs_blackwell.slurm)
else
  A=$(submit scripts/run_control_augs_a100.slurm)
  B=$(submit scripts/run_control_augs_blackwell.slurm)
fi
printf '%s\n%s\n' "$A" "$B" > "$JOBS_FILE"
echo "submitted a100=$A  blackwell=$B"
