#!/usr/bin/env bash
#SBATCH --job-name=d4rt-ctlaugs-a100
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/d4rt_agent/ctlaugs_a100_%j.out
# Under sbatch, $0 is SLURM's spool copy of this file, which is nowhere near
# the repo. SLURM_SUBMIT_DIR is the directory sbatch was invoked from -- the
# repo root, per the usage above and every race submitter, which all cd there
# first. The $0 fallback keeps the file runnable directly, outside SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ ! -f "$REPO_ROOT/scripts/run_augs_body.sh" ]]; then
  echo "REPO_ROOT=$REPO_ROOT does not look like the repo (no scripts/run_augs_body.sh)." >&2
  echo "Submit from the repository root, or set SLURM_SUBMIT_DIR." >&2
  exit 2
fi
MODE=control source "$REPO_ROOT/scripts/run_augs_body.sh"
