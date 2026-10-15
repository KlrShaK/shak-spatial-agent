#!/usr/bin/env bash
# A bounded CPU job for one command:   sbatch scripts/run_cpu_task.slurm <python -m module args...>
# (catalog builds, frame materialisation, evidence writing: anything that reads many clips belongs here,
# not on a login node).
#SBATCH --job-name=dyn-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/decomposition/cpu_%j.out

cd "${SLURM_SUBMIT_DIR:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
export PYTHONPATH="$PWD:/cluster/work/igp_psr/spanwar/Point4D${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/cluster/work/igp_psr/spanwar/envs/spatialstack-qwen35/bin/python}"
exec "$PY" -u "$@"
