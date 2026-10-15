#!/usr/bin/env bash
# Lightweight scheduler controller: no models, no media decoding, no GPU.
# Submit from the frozen source snapshot with EVALUATION_PLAN set.
#SBATCH --job-name=v5-subset-control
#SBATCH --partition=normal.120h
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/subset_control_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from frozen source}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAIN_REPO=/cluster/work/igp_psr/spanwar/shak-spatial-agent
exec /cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python -u \
    -m d4rt_agent.agent_v5.research.subset_eval control --plan "${EVALUATION_PLAN:?set campaign plan}"
