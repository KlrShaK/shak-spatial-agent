#!/usr/bin/env bash
# The two-partition race claim, shared by every racing launcher.
#
# Sourced -- not executed -- by a launcher body that has already set RUN_DIR and
# ATTEMPT_ID. Extracted so the agent and control launchers cannot drift apart on
# the one piece of logic that stops two jobs writing to the same run directory.
# Covered by scripts/test_agent_v4_race.sh.

# ---------------------------------------------------------------------------
# Claim the attempt.
#
# Both partitions were submitted; whichever starts first does the work and the
# other exits. `mkdir` is atomic on the shared filesystem, so exactly one job
# wins even if both reach this line in the same instant.
#
# The claim is keyed on ATTEMPT_ID, not on RUN_DIR. A run longer than the time
# limit is resumed by resubmitting against the SAME RUN_DIR, and a per-run claim
# would already exist by then -- both new jobs would lose and nothing would run.
# ---------------------------------------------------------------------------
RACE_DIR="$RUN_DIR/.race"
CLAIM="$RACE_DIR/$ATTEMPT_ID.claim"
JOBS_FILE="$RACE_DIR/$ATTEMPT_ID.jobs"
mkdir -p "$RACE_DIR"

if ! mkdir "$CLAIM" 2>/dev/null; then
  WINNER="$(cat "$CLAIM/job" 2>/dev/null || echo unknown)"
  echo "Lost the race for attempt $ATTEMPT_ID to job $WINNER; exiting."
  # Exit 0 on purpose. Losing is the designed outcome for one of the two jobs,
  # and a non-zero exit would make every successful run show a failed sibling.
  exit 0
fi

echo "${SLURM_JOB_ID:-local}" > "$CLAIM/job"
echo "$(hostname)" > "$CLAIM/host"
date -u +%Y-%m-%dT%H:%M:%SZ > "$CLAIM/started_at"
echo "Won the race for attempt $ATTEMPT_ID as job ${SLURM_JOB_ID:-local} on $(hostname)"

# Cancel the sibling so it does not hold a queue slot it will only exit from.
# Best-effort: a queued job can dispatch before the submitter has finished
# writing the id file, in which case there is nobody to cancel here and the
# sibling's own claim check is what stops it. That check is the guarantee; this
# is only queue hygiene.
for _ in 1 2 3 4 5; do
  [[ -s "$JOBS_FILE" ]] && break
  sleep 1
done
if [[ -s "$JOBS_FILE" ]]; then
  while read -r sibling; do
    [[ -n "$sibling" && "$sibling" != "${SLURM_JOB_ID:-}" ]] || continue
    echo "Cancelling sibling job $sibling"
    scancel "$sibling" 2>/dev/null || true
  done < "$JOBS_FILE"
fi

# Exercised by scripts/test_agent_v4_race.sh, which needs the real claim code
# path -- a reimplementation of it in a test would not be evidence about this.
if [[ "${AGENT_V4_CLAIM_ONLY:-0}" == "1" ]]; then
  echo "claim-only mode: stopping before the run"
  exit 0
fi
