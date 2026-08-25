#!/usr/bin/env bash
# The traj3d pipeline (SAM3 -> D4RT -> Orient-Anything-V2) on the A100 80GB nodes.
# Derived from run_traj3d_blackwell.slurm; identical body, different allocation.
#
#   sbatch scripts/run_traj3d_a100.slurm
#   RUN_DIR=d4rt_agent/results/traj3d_manual sbatch scripts/run_traj3d_a100.slurm
#   STAGES="segment query" sbatch scripts/run_traj3d_a100.slurm
#
# Or execute it instead of submitting, to have it resolve the run dir and check
# the weight cache on the login node before queueing:
#
#   ./scripts/run_traj3d_a100.slurm
#
# Each stage records completion per video, so a job killed by the time limit is
# continued by resubmitting with the SAME RUN_DIR -- finished work is skipped.
# Without RUN_DIR the script mints a timestamped one and prints it.
#
# Do NOT point this at a RUN_DIR another queued job is already using: the stage
# markers are per video, not per job, so two jobs sharing a run dir race on the
# same files instead of dividing the work.
#
# Runs on the same envs/d4rt_bw_third_party venv as the Blackwell launcher. The
# "bw" is a build provenance, not a restriction -- its torch 2.11.0+cu128 carries
# sm_80 kernels (verified via _cuda_getArchFlags) and nothing in the env is a
# Blackwell-only compiled extension.
#
# Weights must already be cached: compute nodes have no internet and this runs
# with HF_HUB_OFFLINE=1. Run scripts/prepare_traj3d_weights.sh on a login node
# first (once).

#SBATCH --job-name=d4rt-traj3d-a100
# Unlike the Blackwell launcher this file pins no partition.
#
# The typed GPU request is NOT sufficient on its own: the site submit filter
# flattens it, and job 11182996 asked for nvidia_a100_80gb_pcie and landed on a
# 40 GB A100-PCIE. So constrain the node too. The 80 GB A100 hosts are
# eu-a65-[05-07] (EPYC_7413) and eu-ts-[01-02] (EPYC_9354); the 40 GB ones are
# Intel XeonGold_6240R, so the CPU feature separates them cleanly.
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
# Capped at 4 by the es_schin share policy for BAUG-IGP-PRS-HPC-STUDENTS.
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/traj3d_a100_%j.out

set -euo pipefail

# Resolved before anything cds, and used only on the submit-time path below --
# inside a job SLURM runs a spool copy, so this would not point at the repo.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"

REPO_ROOT=/cluster/work/igp_psr/spanwar/Open-d4rt
D4RT_PYTHON=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
HF_HOME=/cluster/work/igp_psr/spanwar/hf_cache

cd "$REPO_ROOT"

STAGES="${STAGES:-segment query orient analyze}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "Unsupported FORCE=$FORCE (expected 0 or 1)" >&2
  exit 2
fi

# Minted here rather than inside the Python so that a resubmission of the same
# command continues the same run instead of silently starting a fresh one.
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="d4rt_agent/results/traj3d_a100_$(date -u +%Y%m%dT%H%M%SZ)"
fi

SAM3_CACHE="$HF_HOME/hub/models--facebook--sam3"
ORIENT_CACHE="$HF_HOME/hub/models--Viglong--OriAnyV2_ckpt"
for cache in "$SAM3_CACHE" "$ORIENT_CACHE"; do
  if [[ ! -d "$cache" ]]; then
    echo "Missing cached weights: $cache" >&2
    echo "Run ./scripts/prepare_traj3d_weights.sh on a login node first." >&2
    exit 2
  fi
done

echo "RUN_DIR=$RUN_DIR"
echo "STAGES=$STAGES"
echo "ONLY=${ONLY:-<all>}"
echo "FORCE=$FORCE"
echo "CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# Submit-time re-exec, matching run_traj3d_blackwell.slurm: executing this file
# resolves RUN_DIR and validates the weight cache on the login node, so a bad
# submit fails now rather than after queueing. Inside a job SLURM_JOB_ID is set
# and this is skipped, so plain `sbatch <this file>` still works.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$REPO_ROOT/logs/d4rt_agent"
  echo "Submitting from login node with RUN_DIR=$RUN_DIR"
  exec sbatch --export=ALL,RUN_DIR="$RUN_DIR",STAGES="$STAGES",FORCE="$FORCE",ONLY="$ONLY" "$SELF"
fi

mkdir -p logs/d4rt_agent "$RUN_DIR"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# SAM3 and the VGGT backbone both benefit, and neither needs full fp32 matmul.
export NVIDIA_TF32_OVERRIDE=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

EXTRA=()
if [[ "$FORCE" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ -n "$ONLY" ]]; then
  # shellcheck disable=SC2206
  ONLY_ARR=($ONLY)
  EXTRA+=(--only "${ONLY_ARR[@]}")
fi
# shellcheck disable=SC2206
STAGES_ARR=($STAGES)

"$D4RT_PYTHON" -m d4rt_agent.sam_orientany_d4rt_test.traj3d_run \
  --run-dir "$RUN_DIR" \
  --stages "${STAGES_ARR[@]}" \
  "${EXTRA[@]}"

echo "traj3d finished: $RUN_DIR"
