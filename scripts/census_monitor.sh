#!/usr/bin/env bash
# Watch the census chain and exit loudly if it stalls.
#
#   ./scripts/census_monitor.sh [poll_seconds]
#
# This IS what drives the chain forward: a compute-node job never calls sbatch
# for its own successor, it only leaves a marker (run_census_body.sh) or falls
# silent. This monitor, running on the login node, is the only thing that ever
# submits -- on a marker, a stall, a wedged node, or an unschedulable job. It
# exits non-zero on a condition a human should look at, so whatever launched it
# is notified rather than the run quietly dying.
set -uo pipefail
REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
cd "$REPO_ROOT"
POLL="${1:-300}"
AUGS="std reverse hflip reverse_hflip"
STALL_LIMIT=3          # consecutive idle polls with no job and no progress
FALLBACK_AFTER_MIN="${FALLBACK_AFTER_MIN:-10}"   # hedge onto A100 after this long unstarted
WEDGE_LIMIT="${WEDGE_LIMIT:-6}"                  # polls of a RUNNING job with zero answers
RESUBMIT_MAX="${RESUBMIT_MAX:-12}"               # cap on auto-resubmits this monitor performs
RACE_DIR="d4rt_agent/results/_census_race"
CHAIN_MARKER="$RACE_DIR/.needs_chain"
LOG=logs/d4rt_agent/census_monitor.log

total_answered() {
  local t=0
  for aug in $AUGS; do
    t=$((t + $(find "d4rt_agent/results/census/${aug}/answers" -name '*.json' 2>/dev/null | wc -l || true)))
  done
  echo "$t"
}
# COMPLETING jobs are excluded: they hold no useful allocation and a wedged
# node can leave one there for hours.
jobs_running() { squeue -h -u "$USER" -t PENDING,RUNNING -o "%j" 2>/dev/null | grep -c "^d4rt-census-" || true; }
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

mkdir -p logs/d4rt_agent "$RACE_DIR"
say "monitor started (poll ${POLL}s, A100 hedge after ${FALLBACK_AFTER_MIN} min unstarted)"
last=$(total_answered); stall=0; wedged=0; resubmits=0; idle_start_total=$last; queued_since=$(date +%s)

# Every sbatch call for this census happens from THIS monitor, on the login
# node -- a compute-node job only ever leaves the marker below, never submits
# its own successor (see run_census_body.sh / run_census.sh for why).
resubmit() {
  resubmits=$((resubmits + 1))
  if [[ "$resubmits" -gt "$RESUBMIT_MAX" ]]; then
    say "hit $RESUBMIT_MAX auto-resubmits from this monitor -- needs a human before continuing"
    exit 6
  fi
  ./scripts/run_census.sh >>"$LOG" 2>&1
}

while true; do
  sleep "$POLL"
  now=$(total_answered); running=$(jobs_running)

  # An unschedulable job is PENDING forever, so every "is something queued?"
  # check passes while nothing happens. Catch it explicitly.
  bad=$(squeue -h -u "$USER" -t PENDING -o "%i %r %j" 2>/dev/null \
        | grep "d4rt-census-" | grep -E "BadConstraints|PartitionConfig" | awk '{print $1}' | head -1)
  if [[ -n "$bad" ]]; then
    say "job $bad is PENDING but UNSCHEDULABLE ($(squeue -h -j "$bad" -o '%r' 2>/dev/null)); cancelling and resubmitting"
    scancel "$bad" 2>/dev/null || true
    sleep 20
    if resubmit; then say "resubmitted"; else
      say "RESUBMIT AFTER BadConstraints FAILED -- needs a human"; exit 5; fi
    last=$(total_answered); continue
  fi

  if [[ "$now" -ge 7076 ]]; then
    say "CENSUS COMPLETE: $now/7076"
    exit 0
  fi

  if [[ "$running" -gt 0 ]]; then
    if [[ "$now" -gt "$last" ]]; then
      say "progress $now/7076 (+$((now - last)))"
      stall=0; wedged=0
    elif squeue -h -u "$USER" -o "%T %j" 2>/dev/null | grep "d4rt-census-" | grep -q RUNNING; then
      # A RUNNING job that answers nothing is the wedged-node case: job 12166132
      # held eu-g7-004 for 3.5 h without writing a byte or claiming the attempt.
      # Give it a grace period (model loading alone is ~5 min), then kill it so
      # the chain can try a different node.
      wedged=$((wedged + 1))
      say "running but no new answers ($now/7076, $wedged/$WEDGE_LIMIT polls)"
      if [[ "$wedged" -ge "$WEDGE_LIMIT" ]]; then
        jid=$(squeue -h -u "$USER" -o "%i %T %j" 2>/dev/null | grep "d4rt-census-" | grep RUNNING | awk '{print $1}' | head -1)
        out="logs/d4rt_agent/census_blackwell_${jid}.out"
        [[ -s "$out" ]] || out="logs/d4rt_agent/census_a100_${jid}.out"
        if [[ -s "$out" ]]; then
          say "job $jid is producing output but no answers -- leaving it alone, check $out"
          wedged=0
        else
          say "job $jid wrote NO output in $((WEDGE_LIMIT * POLL / 60)) min -- wedged node, cancelling"
          scancel "$jid" 2>/dev/null || true
          sleep 30
          resubmit && say "resubmitted after wedge" || { say "RESUBMIT AFTER WEDGE FAILED -- needs a human"; exit 4; }
          wedged=0
        fi
      fi
    fi
    stall=0
    # A job exists but nothing is running yet -- consider hedging onto the A100.
    # run_census.sh --fallback submits only if SLURM says it would start sooner,
    # so calling this repeatedly is safe and usually a no-op.
    if ! squeue -h -u "$USER" -o "%T %j" 2>/dev/null | grep "d4rt-census-" | grep -q RUNNING; then
      pending_min=$(( $(date +%s) - queued_since ))
      if [[ "$pending_min" -ge $((FALLBACK_AFTER_MIN * 60)) ]]; then
        say "nothing started after $((pending_min / 60)) min; checking the A100 fallback"
        ./scripts/run_census.sh --fallback 2>&1 | sed 's/^/    /' | tee -a "$LOG"
        queued_since=$(date +%s)      # re-arm, so this is not checked every poll
      fi
    else
      queued_since=$(date +%s)
    fi
  elif [[ -f "$CHAIN_MARKER" ]]; then
    # Fast path: the last job finished cleanly and left a marker naming its
    # progress instead of submitting anything itself. Submit the successor
    # right away rather than waiting out STALL_LIMIT idle polls.
    marker_total=$(cat "$CHAIN_MARKER" 2>/dev/null || echo "?")
    say "chain marker found ($marker_total/7076); submitting the next attempt from the login node"
    rm -f "$CHAIN_MARKER"
    if resubmit; then say "resubmitted via chain marker"; stall=0
    else say "CHAIN-MARKER RESUBMIT FAILED -- needs a human"; exit 2; fi
  else
    # No job and no marker. Either the chain is mid-handover, or it has stopped.
    # idle_start_total is the baseline from BEFORE this idle streak began -- not
    # the previous poll's total, which by the time stall hits STALL_LIMIT has
    # already caught up to $now even when the job that just ended made plenty
    # of progress. Comparing against the pre-idle baseline is what tells "job
    # finished normally, needs a restart" apart from "genuinely wedged".
    if [[ "$stall" -eq 0 ]]; then idle_start_total="$last"; fi
    stall=$((stall + 1))
    say "no census job queued or running ($now/7076, idle poll $stall/$STALL_LIMIT)"
    if [[ "$stall" -ge "$STALL_LIMIT" ]]; then
      if [[ "$now" -gt "$idle_start_total" ]]; then
        say "chain stopped after making progress -- restarting it"
        if resubmit; then
          say "restarted"; stall=0
        else
          say "RESTART FAILED -- needs a human"; exit 2
        fi
      else
        say "STALLED: no job and no progress since $idle_start_total. Not restarting blindly."
        say "  last log: $(ls -t logs/d4rt_agent/census_*.out 2>/dev/null | head -1)"
        exit 3
      fi
    fi
  fi
  last="$now"
done
