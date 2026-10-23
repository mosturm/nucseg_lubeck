#!/bin/bash
#SBATCH --job-name=reg-method-select
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
: "${ANALYSIS_DIR:?ANALYSIS_DIR must point to the completed analysis_cov_80 directory}"
: "${VIS_RUN_ROOT:?VIS_RUN_ROOT is not set}"

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${VIS_RUN_ROOT}/selection"

python -u "${REGISTER_ROOT}/select_registration_neighborhoods.py" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --output-dir "${VIS_RUN_ROOT}/selection" \
  --top-k 8

