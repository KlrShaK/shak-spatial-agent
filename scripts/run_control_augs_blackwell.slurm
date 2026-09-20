#!/usr/bin/env bash
#SBATCH --job-name=d4rt-ctlaugs-bw
#SBATCH --partition=cuda13pr.4h
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/ctlaugs_bw_%j.out
MODE=control source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/run_augs_body.sh
