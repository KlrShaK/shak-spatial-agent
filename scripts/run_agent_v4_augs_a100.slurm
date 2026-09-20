#!/usr/bin/env bash
# Three splits at ~2.1h each, so this needs a partition that allows more than
# the 4h the single-split launcher uses. gpupr.24h carries the same A100 nodes.
#SBATCH --job-name=d4rt-v4augs-a100
#SBATCH --partition=gpupr.24h
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/v4augs_a100_%j.out
MODE=agent source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/run_augs_body.sh
