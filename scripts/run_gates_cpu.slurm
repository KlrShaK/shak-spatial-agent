#!/usr/bin/env bash
# The D7 gate / fallback / fusion search (decomposition/run_gates.py) as a bounded CPU job: it reads
# every Point4D scene of the dev catalog, so it does not belong on a login node.
#SBATCH --job-name=dyn-gates
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/decomposition/gates_%j.out

cd "${SLURM_SUBMIT_DIR:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
export PYTHONPATH="$PWD:/cluster/work/igp_psr/spanwar/Point4D${PYTHONPATH:+:$PYTHONPATH}"
exec /cluster/work/igp_psr/spanwar/envs/point4d/bin/python -u -m dyn_dataset_exploration.decomposition.run_gates "$@"
