#!/usr/bin/env bash
# Submit from a research.snapshot directory. Each invocation evaluates one benchmark/split/shard.
# Use disjoint RUN_DIRs per variant, split and shard; resubmit the same invocation to resume.
#SBATCH --job-name=agentv5-eval
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --partition=gpupr.24h
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/composition_eval_%j.out

set -euo pipefail
export REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from a frozen source directory}"
export MAIN_REPO="${MAIN_REPO:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
export BENCH="${BENCH:?dsi, dynbench or vlm4d}"
export BACKBONE="${BACKBONE:-point4d}"
export RUN_DIR="${RUN_DIR:?set a unique result directory for this variant/split/shard}"
STAGE="${STAGE:-diagnostics}"
VARIANT="${VARIANT:-candidate}"
DSI_SPLIT="${DSI_SPLIT:-std}"
TASK_SHARD="${TASK_SHARD:-0/1}"

if [[ ! -f "$REPO_ROOT/source_manifest.json" ]]; then
    echo "Create a frozen source snapshot before submitting evaluation." >&2
    exit 2
fi
/usr/bin/python3 - "$REPO_ROOT/source_manifest.json" "$VARIANT" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
original = sys.argv[2] in ("baseline", "legacy_budget32")
if manifest['baseline'] != original:
    raise SystemExit('baseline/legacy_budget32 require a --baseline snapshot; other variants require a candidate snapshot')
PY

TASK_ARGS="--max-steps 32 --token-budget 16000 --shard $TASK_SHARD"
case "$VARIANT" in
    baseline) TASK_ARGS="$TASK_ARGS --stack legacy --max-steps 16" ;;
    legacy_budget32) TASK_ARGS="$TASK_ARGS --stack legacy" ;;
    candidate) TASK_ARGS="$TASK_ARGS --stack compositional" ;;
    legacy_menu) TASK_ARGS="$TASK_ARGS --stack compositional --legacy-menu" ;;
    no_repair) TASK_ARGS="$TASK_ARGS --stack compositional --answer-policy rated" ;;
    workflow_plain) TASK_ARGS="$TASK_ARGS --stack compositional --answer-policy plain --repair-cycles 0" ;;
    workflow_rated) TASK_ARGS="$TASK_ARGS --stack compositional --answer-policy rated" ;;
    workflow_repair) TASK_ARGS="$TASK_ARGS --stack compositional --answer-policy rated_repair --repair-tool-limit 4" ;;
    standard_views) TASK_ARGS="$TASK_ARGS --stack compositional --answer-views standard" ;;
    *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
case "$BENCH" in
    dsi)
        case "$DSI_SPLIT" in std|hflip|reverse|reverse_hflip) ;; *) exit 2 ;; esac
        TASK_ARGS="$TASK_ARGS --random-sample 1769 --split $DSI_SPLIT" ;;
    dynbench) TASK_ARGS="$TASK_ARGS --all" ;;
    vlm4d) ;;
    *) echo "Unknown BENCH=$BENCH" >&2; exit 2 ;;
esac
case "$STAGE" in
    diagnostics)
        DIAGNOSTIC_MANIFEST="${DIAGNOSTIC_MANIFEST:?set the reviewed diagnostic manifest}"
        [[ -f "$DIAGNOSTIC_MANIFEST" ]] || exit 2
        TASK_ARGS="$TASK_ARGS --selection-file $DIAGNOSTIC_MANIFEST" ;;
    full) ;;
    *) echo "STAGE must be diagnostics or full" >&2; exit 2 ;;
esac
export EXTRA_ARGS="$TASK_ARGS"
bash "$REPO_ROOT/scripts/agent_v5_body.sh"
