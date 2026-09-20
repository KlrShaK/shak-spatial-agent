#!/usr/bin/env bash
#SBATCH --job-name=d4rt-census-blackwell
# No explicit partition. Pinning cuda13pr.24h made the site filter append
# gpupr.24h, and SLURM then rejected the job as BadConstraints because
# EPYC_9654 cannot be satisfied in that second partition -- it sat PENDING
# forever. Letting the filter route it works, as it does for the A100 file.
#SBATCH --gpus=nvidia_rtx_pro_6000:1
# Must stay typed and EPYC_9654: a generic --gpus=1 is rejected by the site
# ("node configuration is not available"), and a widened constraint only
# works with a generic GPU request, so the two cannot be relaxed together.
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
# 6G x 4 cpus = 24 GiB, which is the hard cap for this share
# (es_schin / BAUG-IGP-PRS-HPC-STUDENTS). There is no headroom to buy, so
# the bounded clip cache in QuestionPipeline is what keeps the run alive.
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=16G
# 12h, not 24h. The census chains itself, so walltime affects only how easily
# SLURM can schedule us -- and a 24h contiguous GPU window is hard to backfill:
# with the queue otherwise empty for us, a 24h request was planned for 23:15
# while any shorter one could start at 21:11. 12h keeps handovers (and their
# ~5 min of model loading) infrequent while still fitting backfill gaps.
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/Open-d4rt/logs/d4rt_agent/census_blackwell_%j.out
# Soft deadline 30 min inside the wall clock, so the last question completes.
DEADLINE_MIN=1410 source /cluster/work/igp_psr/spanwar/Open-d4rt/scripts/run_census_body.sh
