#!/usr/bin/env bash
#SBATCH --job-name=d4rt-v4augs-bw
#SBATCH --partition=cuda13pr.24h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/v4augs_bw_%j.out
MODE=agent source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/run_augs_body.sh
