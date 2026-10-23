#!/bin/bash
#SBATCH --job-name=reg-method-render
#SBATCH --partition=scc-cpu
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
: "${ANALYSIS_DIR:?ANALYSIS_DIR must point to the completed analysis_cov_80 directory}"
: "${VIS_RUN_ROOT:?VIS_RUN_ROOT is not set}"

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}" "${VIS_RUN_ROOT}/figures"

RANK=$((SLURM_ARRAY_TASK_ID + 1))
python -u "${REGISTER_ROOT}/render_registration_neighborhood.py" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --selection-dir "${VIS_RUN_ROOT}/selection" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${VIS_RUN_ROOT}/figures" \
  --rank "${RANK}" \
  --array-key 0 \
  --max-render-dim 180 \
  --cell-id-step 10 \
  --max-cell-points 250000

