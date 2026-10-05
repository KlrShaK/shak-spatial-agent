#!/usr/bin/env bash
# Agent v5 smoke test: one question per question group of the eval split, with one backbone, on
# an A100 80GB. Checks that Qwen3-VL, SAM3, the backbone (and DA3-Nested for Point4D) and OAV2 load
# together in one process, and records peak GPU memory in every answer file.
#
#   BACKBONE=point4d sbatch scripts/run_agent_v5_smoke.slurm
#   BACKBONE=d4rt    sbatch scripts/run_agent_v5_smoke.slurm
#
# Submit from the repository (or worktree) root. Results: d4rt_agent/results/agent_v5_smoke_<backbone>/.

#SBATCH --job-name=d4rt-agentv5-smoke
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/agentv5_smoke_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export BACKBONE="${BACKBONE:-point4d}"
export RUN_DIR="${RUN_DIR:-d4rt_agent/results/agent_v5_smoke_${BACKBONE}}"
export EXTRA_ARGS="--one-per-group ${EXTRA_ARGS:-}"
unset ATTEMPT_ID
source "$REPO_ROOT/scripts/agent_v5_body.sh"
