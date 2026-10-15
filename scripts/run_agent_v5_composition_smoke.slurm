#!/usr/bin/env bash
# Submit from a frozen source snapshot (research.snapshot); MAIN_REPO supplies staged model dependencies.
#SBATCH --job-name=agentv5-composition
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/composition_smoke_%j.out

set -euo pipefail
export REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the frozen source directory}"
export MAIN_REPO="${MAIN_REPO:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
export BACKBONE="${BACKBONE:-point4d}"
TASK_RUN_ROOT="${TASK_RUN_ROOT:?set the smoke result directory}"
for BENCH in dsi dynbench vlm4d; do
    export BENCH
    export RUN_DIR="$TASK_RUN_ROOT/$BENCH"
    export EXTRA_ARGS="--stack compositional --answer-policy rated_repair --repair-tool-limit 4 --smoke 3 --max-steps 32"
    bash "$REPO_ROOT/scripts/agent_v5_body.sh"
done

/cluster/work/igp_psr/spanwar/envs/spatialstack-qwen35/bin/python - "$TASK_RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys
records = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('*/answers/*.json')]
failed = [r['question_id'] for r in records if not r.get('answered') or r.get('status') == 'error']
workflow_missing = [r['question_id'] for r in records if r.get('schema_version') != 3 or
                    r.get('answer_protocol') != 'freeform_assessment_v1' or not r.get('assessment_present')]
summary = {'questions': len(records), 'answered': sum(bool(r.get('answered')) for r in records), 'failed': failed,
           'workflow_missing': workflow_missing, 'schema_version': 3, 'scope': 'runtime smoke, not an accuracy evaluation'}
(Path(sys.argv[1]) / 'smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
if len(records) != 9 or failed or workflow_missing:
    raise SystemExit(1)
PY
