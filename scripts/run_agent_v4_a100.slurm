#!/usr/bin/env bash
# Agent v4 on the A100 80GB nodes. Submitted as one half of a race by
# scripts/run_agent_v4_race.sh -- do not sbatch this directly unless you want
# only this partition.

#SBATCH --job-name=d4rt-agentv4-a100
# The typed GPU request is NOT sufficient on its own: the site submit filter
# flattens it, and job 11182996 asked for nvidia_a100_80gb_pcie and landed on a
# 40 GB A100-PCIE. So constrain the node too. The 80 GB A100 hosts are
# eu-a65-[05-07] (EPYC_7413) and eu-ts-[01-02] (EPYC_9354); the 40 GB ones are
# Intel XeonGold_6240R, so the CPU feature separates them cleanly. Four
# co-resident models need ~46 GB, so a 40 GB card would OOM at load.
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --constraint="EPYC_7413|EPYC_9354"
# Capped at 4 by the es_schin share policy for BAUG-IGP-PRS-HPC-STUDENTS.
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/agentv4_a100_%j.out

source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/agent_v4_body.sh
