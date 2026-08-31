#!/bin/bash
#SBATCH --job-name=vessel-ms-union
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-union-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-union-%j.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${OUTPUT_ZARR:?Missing OUTPUT_ZARR}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SCRIPT="${REPO_ROOT}/vessel_multiscale_ome_zarr.py"
VENV="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec > >(tee -a "${RUN_ROOT}/finalize.log") 2>&1
cp "$0" "${RUN_ROOT}/submitted_finalize.slurm"

python -u "${SCRIPT}" finalize \
  --run-root "${RUN_ROOT}" \
  --output "${OUTPUT_ZARR}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --compressor-level "${COMPRESSOR_LEVEL:-5}"

test -f "${RUN_ROOT}/COMPLETED"
echo "Final multiscale vessel union: ${OUTPUT_ZARR}"
