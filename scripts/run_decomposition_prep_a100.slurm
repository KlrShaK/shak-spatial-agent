#!/usr/bin/env bash
# GPU prep for dyn_dataset_exploration/testing_decomposition.ipynb on an A100 (40 or 80 GB).
#
#   PART=point4d   sbatch scripts/run_decomposition_prep_a100.slurm   # Point4D scene + DA3 metric
#   PART=qwen_sam3 sbatch scripts/run_decomposition_prep_a100.slurm   # Qwen + SAM3
#   OUT=<catalog dir> SHARD=i/n STAGES="..." ONLY="..." are passed through (see the body).
#
# Any A100, 40 or 80 GB: the stages fit in 32 GB (peak GPU memory is logged per clip). gpumem filters
# by memory; the CPU feature names the A100 hosts (40 GB: XeonGold_6240R, 80 GB: EPYC_7413/EPYC_9354).

#SBATCH --job-name=dyn-decomp-prep
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32G
#SBATCH --constraint="EPYC_7413|EPYC_9354|XeonGold_6240R"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --tmp=16G
#SBATCH --time=01:30:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/decomposition/prep_%x_%j.out

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PLATFORM=a100
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/decomposition_prep_body.sh"
