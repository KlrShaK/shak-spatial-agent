#!/usr/bin/env bash
# Agent v4 on the A100 80GB nodes. Submitted as one half of a race by
# scripts/run_agent_v4_race.sh -- do not sbatch this directly unless you want
# only this partition.

#SBATCH --job-name=d4rt-agentv4-a100
# The typed GPU request is NOT sufficient on its own: the site submit filter
# flattens it, and job 11182996 asked for nvidia_a100_80gb_pcie and landed on a
# 40 GB A100-PCIE. So constrain the node too. The 80 GB A100 hosts are
# eu-a65-[05-07] (EPYC_7413) and eu-ts-[01-02] (EPYC_9354); the 40 GB ones are
# Intel XeonGold_6240R, so the CPU feature separates them cleanly. Four
# co-resident models need ~46 GB, so a 40 GB card would OOM at load.
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
# Capped at 4 by the es_schin share policy for BAUG-IGP-PRS-HPC-STUDENTS.
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/agentv4_a100_%j.out

# Under sbatch, $0 is SLURM's spool copy of this file, which is nowhere near
# the repo. SLURM_SUBMIT_DIR is the directory sbatch was invoked from -- the
# repo root, per the usage above and every race submitter, which all cd there
# first. The $0 fallback keeps the file runnable directly, outside SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/scripts/agent_v4_body.sh" ]]; then
  echo "REPO_ROOT=$REPO_ROOT does not look like the repo (no scripts/agent_v4_body.sh)." >&2
  echo "Submit from the repository root, or set SLURM_SUBMIT_DIR." >&2
  exit 2
fi
source "$REPO_ROOT/scripts/agent_v4_body.sh"
