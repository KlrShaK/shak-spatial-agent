#!/usr/bin/env bash
# Exercise the claim protocol in scripts/agent_v4_body.sh, without SLURM.
#
# The protocol has one job: of two jobs submitted for the same attempt, exactly
# one does the work. Getting that wrong means two jobs sharing a RUN_DIR and
# racing on the same per-question files -- the hazard run_traj3d_a100.slurm
# warns about -- so it is worth a test that does not need a GPU to run.

set -euo pipefail
cd /cluster/work/igp_psr/spanwar/Open-d4rt

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export AGENT_V4_CLAIM_ONLY=1
BODY=scripts/agent_v4_body.sh
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- two jobs, same attempt: exactly one wins ------------------------------
RUN_DIR="$TMP/run" ATTEMPT_ID=attempt-1 SLURM_JOB_ID=1001 bash "$BODY" > "$TMP/a.log" 2>&1
RUN_DIR="$TMP/run" ATTEMPT_ID=attempt-1 SLURM_JOB_ID=1002 bash "$BODY" > "$TMP/b.log" 2>&1
grep -q "^Won the race" "$TMP/a.log" || fail "first job should have won"
grep -q "^Lost the race" "$TMP/b.log" || fail "second job should have lost"
grep -q "to job 1001" "$TMP/b.log" || fail "loser should name the winner"
[[ "$(cat "$TMP/run/.race/attempt-1.claim/job")" == 1001 ]] || fail "claim records the winner"
echo "ok: exactly one job claims an attempt"

# --- the loser exits 0: losing is a designed outcome, not a failure --------
set +e
RUN_DIR="$TMP/run" ATTEMPT_ID=attempt-1 SLURM_JOB_ID=1003 bash "$BODY" > /dev/null 2>&1
status=$?
set -e
[[ $status -eq 0 ]] || fail "a losing job must exit 0, got $status"
echo "ok: the loser exits 0"

# --- a new attempt against the SAME run dir races again --------------------
# The regression this guards: keying the claim on RUN_DIR instead of ATTEMPT_ID
# would make every resumption find a claim already taken, so BOTH jobs would
# lose and a run past the 3 h limit could never be continued.
RUN_DIR="$TMP/run" ATTEMPT_ID=attempt-2 SLURM_JOB_ID=2001 bash "$BODY" > "$TMP/c.log" 2>&1
grep -q "^Won the race" "$TMP/c.log" || fail "a fresh attempt must be claimable in the same run dir"
echo "ok: resuming the same run dir races again"

# --- the winner cancels its sibling ---------------------------------------
mkdir -p "$TMP/run2/.race"
printf '3001\n3002\n' > "$TMP/run2/.race/attempt-3.jobs"
cat > "$TMP/fakescancel" <<'STUB'
#!/usr/bin/env bash
echo "$1" >> "$SCANCEL_LOG"
STUB
chmod +x "$TMP/fakescancel"
ln -sf "$TMP/fakescancel" "$TMP/scancel"
SCANCEL_LOG="$TMP/cancelled" PATH="$TMP:$PATH" \
  RUN_DIR="$TMP/run2" ATTEMPT_ID=attempt-3 SLURM_JOB_ID=3001 bash "$BODY" > "$TMP/d.log" 2>&1
[[ "$(cat "$TMP/cancelled")" == 3002 ]] || fail "winner should cancel only its sibling, got: $(cat "$TMP/cancelled" 2>/dev/null)"
echo "ok: the winner cancels the sibling and not itself"

echo
echo "claim protocol: all checks passed"
