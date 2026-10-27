#!/usr/bin/env bash
# Race allocation first; execute the original frozen evaluation worker only after claiming.
set -euo pipefail
PLAN="${V6_PLAN:?set the frozen campaign plan}"
REPO_ROOT="$(dirname "$PLAN")/source"
RUN_DIR="$(dirname "$PLAN")"
ATTEMPT_ID="${V6_ATTEMPT_ID:?set the shared race attempt}"
source "$REPO_ROOT/scripts/race_claim.sh"
exec bash "$REPO_ROOT/scripts/agent_v6_job.sh"
