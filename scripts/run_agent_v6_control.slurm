#!/usr/bin/env bash
#SBATCH --job-name=v6-evaluation-control
#SBATCH --partition=normal.120h
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/v6_control_%j.out
set -euo pipefail
PLAN="${V6_PLAN:?set the frozen campaign plan}"
cd "$(dirname "$PLAN")/source"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec /cluster/work/igp_psr/spanwar/envs/spatialstack-qwen35/bin/python -u \
    "${V6_CONTROL_CODE:-$PWD/d4rt_agent/agent_v6/evaluation/campaign.py}" control --plan "$PLAN"
