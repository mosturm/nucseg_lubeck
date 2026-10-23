#!/bin/bash
#SBATCH --job-name=detailed-voxels
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=detailed-voxels-%j.out
#SBATCH --error=detailed-voxels-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT is required}"
COORDINATE_TABLE="${COORDINATE_TABLE:-${REGISTER_ROOT}/detailed_questions/H2P7_Laesion_H2P2_Kontrolle_Koordinaten.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/run_detailed_collaborator_voxels_${SLURM_JOB_ID}}"

test -d "${SOURCE_RUN_ROOT}/analysis_cov_80/raw_voxel_tables"
test -d "${SOURCE_RUN_ROOT}/features/per_cell_radius_tables"
test -f "${COORDINATE_TABLE}"
mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/.cache/matplotlib"
exec > >(tee -a "${OUTPUT_ROOT}/workflow.log") 2>&1

finish() {
  status=$?
  trap - EXIT
  if [[ ${status} -eq 0 ]]; then
    touch "${OUTPUT_ROOT}/COMPLETED"
  else
    printf 'exit_code=%s\n' "${status}" > "${OUTPUT_ROOT}/FAILED"
  fi
  exit "${status}"
}
trap finish EXIT

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cp "${COORDINATE_TABLE}" "${OUTPUT_ROOT}/coordinate_requests_input.csv"
printf 'source_run=%s\noutput_root=%s\ncoordinate_table=%s\n' \
  "${SOURCE_RUN_ROOT}" "${OUTPUT_ROOT}" "${COORDINATE_TABLE}" \
  > "${OUTPUT_ROOT}/run_inputs.txt"

python -u "${REGISTER_ROOT}/plot_detailed_collaborator_voxels.py" \
  --source-run-root "${SOURCE_RUN_ROOT}" \
  --coordinate-table "${COORDINATE_TABLE}" \
  --output-dir "${OUTPUT_ROOT}" \
  --min-cells 20 \
  --examples-per-category 3 \
  --seed 46 \
  --radius-max-um 11

echo "Completed detailed collaborator plots: ${OUTPUT_ROOT}"
