#!/bin/bash
#SBATCH --job-name=vessel-ms-prepare
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=2h
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-prepare-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-prepare-%j.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${INPUT_ZARR:?Missing INPUT_ZARR}"
: "${MID_FACTOR_ZYX:?Missing MID_FACTOR_ZYX}"
: "${COARSE_FACTOR_ZYX:?Missing COARSE_FACTOR_ZYX}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SCRIPT="${REPO_ROOT}/vessel_multiscale_ome_zarr.py"
VENV="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/prepare.log") 2>&1
cp "$0" "${RUN_ROOT}/submitted_prepare.slurm"

python -u "${SCRIPT}" prepare \
  --input "${INPUT_ZARR}" \
  --array-key "${ARRAY_KEY:-0}" \
  --run-root "${RUN_ROOT}" \
  --mid-factor "${MID_FACTOR_ZYX}" \
  --coarse-factor "${COARSE_FACTOR_ZYX}" \
  --chunks "${OUTPUT_CHUNKS:-8x256x256}" \
  --compressor-level "${COMPRESSOR_LEVEL:-5}"

test -f "${RUN_ROOT}/PREPARED"
