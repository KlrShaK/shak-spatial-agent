#!/usr/bin/env bash
# Agent v4 over the full DSI-Bench census (1769 x 4 = 7,076 questions).
#
#   ./scripts/run_census.sh            # start, or CONTINUE an unfinished run
#   ./scripts/run_census.sh --status   # progress only, submit nothing
#
# RESUMING IS THE SAME COMMAND. One answer file per question; anything already
# answered is skipped. A running job cannot chain itself across the 24 h limit
# -- it only leaves a marker (see run_census_body.sh) -- so
# scripts/census_monitor.sh, running on the LOGIN NODE, watches for that marker
# and calls this script again to submit the next attempt. Every sbatch call for
# this census happens from the login node; a compute node never submits its own
# successor. Running this by hand is only needed to start the census, or to
# restart it if the monitor itself is not running.
#
# SUBMISSION STRATEGY. Blackwell is preferred and goes out immediately. `--fallback`
# adds the A100 as a competitor, but only if SLURM says it would start sooner --
# see the --fallback branch below. Whichever starts first claims the attempt and
# cancels the other.
set -euo pipefail
REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
cd "$REPO_ROOT"
AUGS="std reverse hflip reverse_hflip"
FALLBACK_DELAY_MIN="${FALLBACK_DELAY_MIN:-10}"

MODE="${1:-}"

# --fallback: consider adding the A100 as a competitor to a Blackwell job that
# has not started. Submits ONLY if SLURM says the A100 would start sooner --
# blindly adding it is worse than nothing, since it consumes a queue slot and
# can delay the Blackwell job it was meant to hedge.
if [[ "$MODE" == "--fallback" ]]; then
  running=$(squeue -h -u "$USER" -o "%T %j" 2>/dev/null | grep "d4rt-census-" | grep -c RUNNING || true)
  if [[ "$running" -gt 0 ]]; then
    echo "a census job is already RUNNING; no fallback needed"; exit 0
  fi
  pending=$(squeue -h -u "$USER" -t PENDING -o "%i %j" 2>/dev/null | grep "d4rt-census-blackwell" | awk '{print $1}' | head -1)
  if [[ -z "$pending" ]]; then
    echo "no pending blackwell job to hedge; nothing to do"; exit 0
  fi
  if squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -q "d4rt-census-a100"; then
    echo "an A100 job is already queued; nothing to do"; exit 0
  fi
  bw_start=$(scontrol show job "$pending" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StartTime=//p')
  a1_start=$(sbatch --test-only --export=ALL,RUN_DIR=x,ATTEMPT_ID=x \
               scripts/run_census_a100.slurm 2>&1 | sed -n 's/.*to start at \([^ ]*\).*/\1/p')
  echo "blackwell $pending estimated $bw_start ; a100 would be $a1_start"
  if [[ -z "$a1_start" || -z "$bw_start" || ! "$a1_start" < "$bw_start" ]]; then
    echo "A100 would not start sooner -- not submitting it"
    exit 0
  fi
  RUN_DIR="d4rt_agent/results/_census_race"
  ATTEMPT=$(ls -t "$RUN_DIR/.race"/*.jobs 2>/dev/null | head -1)
  AID=$(basename "${ATTEMPT:-unknown.jobs}" .jobs)
  A=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR",ATTEMPT_ID="$AID" \
        --partition=gpupr.24h scripts/run_census_a100.slurm)
  echo "$A" >> "$ATTEMPT"
  echo "submitted a100=$A as a faster alternative; the claim protocol keeps whichever starts"
  exit 0
fi

progress() {
  local total=0 n
  echo "split             answered   remaining"
  for aug in $AUGS; do
    n=$(find "d4rt_agent/results/census/${aug}/answers" -name '*.json' 2>/dev/null | wc -l || true)
    printf "  %-16s %5d      %5d\n" "$aug" "$n" "$((1769 - n))"
    total=$((total + n))
  done
  printf "  %-16s %5d/7076  (%d%%)\n" "TOTAL" "$total" "$((total * 100 / 7076))"
  REMAINING=$((7076 - total))
}

progress
[[ "$MODE" == "--status" ]] && exit 0

if [[ "$REMAINING" -eq 0 ]]; then
  echo ""; echo "census complete -- nothing to submit."; exit 0
fi

# One attempt at a time: two jobs sharing an answers directory would race per
# question. Only PENDING/RUNNING blocks a new submit. A job stuck in COMPLETING
# is not going to do any work -- a wedged node can sit there for a long time --
# and treating it as live would leave the census stalled behind a corpse.
if squeue -h -u "$USER" -t PENDING,RUNNING -o "%j" 2>/dev/null | grep -q "^d4rt-census-"; then
  echo "" >&2
  echo "A census job is already queued or running:" >&2
  squeue -h -u "$USER" -o "  %.10i %.20j %.10T %.8M" 2>/dev/null | grep census >&2
  echo "It chains itself, so nothing needs doing. Use --status to watch." >&2
  exit 3
fi

[[ -d "$REPO_ROOT/.cache/qwen3vl_python/transformers" ]] || { echo "Missing staged deps" >&2; exit 2; }
for aug in $AUGS; do
  [[ -f "d4rt_agent/results/census/${aug}/manifest.json" ]] || { echo "Missing manifest: $aug" >&2; exit 2; }
done

RUN_DIR="d4rt_agent/results/_census_race"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p logs/d4rt_agent "$RUN_DIR/.race"
echo ""
echo "$REMAINING remaining (~$((REMAINING * 40 / 3600))h). ATTEMPT_ID=$ATTEMPT_ID"

EXPORTS="ALL,RUN_DIR=$RUN_DIR,ATTEMPT_ID=$ATTEMPT_ID"

# Blackwell only. The A100 fallback is NOT submitted alongside: submitting it
# with `--begin=+10min` put it behind Blackwell in the backfill plan (estimated
# start a day later, while holding a queue slot and pushing Blackwell's own
# estimate back). `--fallback` below adds it later, and only when it helps.
# Every submit here runs on the login node (this script is never invoked from
# inside a compute-node job). That still matters for the launcher itself: once,
# a compute-node submit let the site filter silently append gpupr.24h, whose
# nodes cannot satisfy EPYC_9654 -> BadConstraints, pending forever.
B=$(sbatch --parsable --export="$EXPORTS" scripts/run_census_blackwell.slurm)
printf '%s\n' "$B" > "$RUN_DIR/.race/$ATTEMPT_ID.jobs"
echo "submitted blackwell=$B"

# A job the scheduler will NEVER run still sits in PENDING, so it satisfies every
# "is something queued?" check while making no progress -- the census silently
# stalls. This bit us once already: the site filter appends a second partition,
# and a constraint unsatisfiable there is rejected as BadConstraints. Verify, and
# switch to the A100 launcher if so.
sleep 20
REASON=$(squeue -h -j "$B" -o "%r" 2>/dev/null | tr -d ' ')
if [[ "$REASON" == "BadConstraints" || "$REASON" == "PartitionConfig" || "$REASON" == "Resources,BadConstraints" ]]; then
  echo "  blackwell $B is unschedulable ($REASON); cancelling and using the A100 launcher"
  scancel "$B" 2>/dev/null || true
  A=$(sbatch --parsable --export="$EXPORTS" scripts/run_census_a100.slurm)
  printf '%s\n' "$A" > "$RUN_DIR/.race/$ATTEMPT_ID.jobs"
  REASON2=$(squeue -h -j "$A" -o "%r" 2>/dev/null | tr -d ' ')
  echo "  submitted a100=$A (${REASON2:-queued})"
  if [[ "$REASON2" == "BadConstraints" ]]; then
    echo "  BOTH launchers are unschedulable -- needs a human" >&2
    exit 4
  fi
fi
scontrol show job "$B" 2>/dev/null | tr ' ' '\n' | grep -E "^StartTime=" | sed 's/^/  estimated /'
echo "  (A100 fallback is added by --fallback only if it would start sooner)"
