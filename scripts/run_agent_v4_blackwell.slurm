#!/usr/bin/env bash
# Agent v4 on the RTX PRO 6000 Blackwell nodes. Submitted as one half of a race
# by scripts/run_agent_v4_race.sh.

#SBATCH --job-name=d4rt-agentv4-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# The site submit filter flattens the typed GPU request and also adds gpupr.4h,
# whose A100 nodes share the cuda13/gpupr features -- so pin to the Blackwell
# nodes explicitly. EPYC_9654 is unique to eu-g7-[001-010] among reachable GPU nodes.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/agentv4_bw_%j.out

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
