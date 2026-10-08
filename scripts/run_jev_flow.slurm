#!/usr/bin/env bash
# Jev-flow (caption + Decisions-driven AgentV5 phases) on one RTX 4090 (24 GB): the caption model is 2B and the agent loads no Qwen.
# The run stage's peak memory is not measured yet; if it needs more, override with e.g. --gpus=1 --gres=gpumem:32G. One job at a time; see jev_flow_body.sh.
#
#   STAGE=caption RUN_DIR=d4rt_agent/results/jev_pilot EXTRA_ARGS="--smoke 25" sbatch --time=00:30:00 scripts/run_jev_flow.slurm
#   STAGE=run     RUN_DIR=d4rt_agent/results/jev_pilot SEED_CACHE=<V5 run>/cache EXTRA_ARGS="--smoke 25" sbatch scripts/run_jev_flow.slurm
#
# Submit from the worktree root (it uses SLURM_SUBMIT_DIR).

#SBATCH --job-name=jev-flow-a100
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent-jev-fast-reason/logs/jev_flow_%x_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/scripts/jev_flow_body.sh" ]]; then
  echo "REPO_ROOT=$REPO_ROOT does not look like the repo (no scripts/jev_flow_body.sh); submit from the repo root." >&2
  exit 2
fi
source "$REPO_ROOT/scripts/jev_flow_body.sh"
