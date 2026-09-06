#!/bin/bash
#SBATCH --job-name=cell-dist-finalize
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-finalize-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-finalize-%j.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${OUTPUT_ZARR:?Missing OUTPUT_ZARR}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
SCRIPT="${PROJECT_ROOT}/repo/nucseg_lubeck/cell_distributed_ome_zarr.py"
VENV="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec > >(tee -a "${RUN_ROOT}/finalize.log") 2>&1
cp "$0" "${RUN_ROOT}/submitted_finalize.slurm"

keep_args=()
if [[ "${KEEP_WORKING_LABELS:-0}" == 1 ]]; then
  keep_args+=(--keep-working-labels)
fi

python -u "${SCRIPT}" finalize \
  --run-root "${RUN_ROOT}" \
  --output "${OUTPUT_ZARR}" \
  "${keep_args[@]}"

test -f "${RUN_ROOT}/COMPLETED"
echo "Final distributed cell labels: ${OUTPUT_ZARR}"
