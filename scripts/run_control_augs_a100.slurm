#!/usr/bin/env bash
#SBATCH --job-name=d4rt-ctlaugs-a100
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/ctlaugs_a100_%j.out
MODE=control source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/run_augs_body.sh
