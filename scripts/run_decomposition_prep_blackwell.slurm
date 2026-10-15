#!/usr/bin/env bash
# The same prep on the Blackwell nodes (RTX PRO 6000, 96 GB). See run_decomposition_prep_a100.slurm.
#
#   PART=qwen_sam3 sbatch scripts/run_decomposition_prep_blackwell.slurm
#   PART=point4d   sbatch scripts/run_decomposition_prep_blackwell.slurm   # envs/d4rt_bw_third_party + Point4D on PYTHONPATH
#
# Pinned to eu-g7-[001-010]: the site submit filter flattens a typed GPU request and would also add
# gpupr.4h, so the Blackwell partition and the one CPU feature unique to those hosts are explicit
# (same as run_traj3d_blackwell.slurm).

#SBATCH --job-name=dyn-decomp-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/decomposition/prep_%x_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PLATFORM=blackwell
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/decomposition_prep_body.sh"
