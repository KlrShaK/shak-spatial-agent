#!/usr/bin/env bash
# decomposition/policy_lab.py: fits and cross-validates D7 policies on the dumped per-question rows (no scenes).
#SBATCH --job-name=dyn-policylab
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/shak-spatial-agent/logs/decomposition/policylab_%j.out

cd "${SLURM_SUBMIT_DIR:-/cluster/work/igp_psr/spanwar/shak-spatial-agent}"
export PYTHONPATH="$PWD:/cluster/work/igp_psr/spanwar/Point4D${PYTHONPATH:+:$PYTHONPATH}"
exec /cluster/work/igp_psr/spanwar/envs/point4d/bin/python -u -m dyn_dataset_exploration.decomposition.policy_lab "$@"
