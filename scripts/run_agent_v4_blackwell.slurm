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
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/agentv4_bw_%j.out

source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/agent_v4_body.sh
