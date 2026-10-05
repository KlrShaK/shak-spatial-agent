#!/usr/bin/env bash
# Agent v5 on a100. Normally submitted by scripts/run_agent_v5_race.sh (both partitions, the first
# to start wins); BACKBONE=point4d|d4rt and BENCH=dynbench|dsi come from the environment.

#SBATCH --job-name=d4rt-agentv5-a100
# 80 GB A100 only: the site filter flattens the typed GPU request, the CPU feature separates the
# 80 GB hosts (EPYC_7413 / EPYC_9354) from the 40 GB ones (see run_agent_v4_a100.slurm).
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --partition=gpupr.24h
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/agentv5_a100_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/scripts/agent_v5_body.sh" ]]; then
  echo "REPO_ROOT=$REPO_ROOT does not look like the repo (no scripts/agent_v5_body.sh); submit from the repo root." >&2
  exit 2
fi
source "$REPO_ROOT/scripts/agent_v5_body.sh"
