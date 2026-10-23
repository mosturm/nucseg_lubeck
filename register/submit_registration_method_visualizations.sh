#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
: "${ANALYSIS_DIR:?Set ANALYSIS_DIR to the newest completed analysis_cov_80 directory}"

test -s "${ANALYSIS_DIR}/all_univariate_correlations_and_grouped_cv.csv" || {
  echo "Missing correlation table below ANALYSIS_DIR=${ANALYSIS_DIR}" >&2
  exit 2
}
test -d "${ANALYSIS_DIR}/raw_voxel_tables" || {
  echo "Missing raw_voxel_tables below ANALYSIS_DIR=${ANALYSIS_DIR}" >&2
  exit 2
}

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
VIS_RUN_ROOT="${PROJECT_ROOT}/runs/run_registration_method_visualization_${RUN_TAG}"
mkdir -p "${VIS_RUN_ROOT}"
export PROJECT_ROOT REGISTER_ROOT DATA_ROOT ANALYSIS_DIR VIS_RUN_ROOT

SELECT_JOB=$(sbatch --parsable --export=ALL \
  --output="${VIS_RUN_ROOT}/selection-%j.out" \
  --error="${VIS_RUN_ROOT}/selection-%j.err" \
  "${REGISTER_ROOT}/select_registration_neighborhoods.slurm")
RENDER_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --export=ALL \
  --output="${VIS_RUN_ROOT}/render-%A_%a.out" \
  --error="${VIS_RUN_ROOT}/render-%A_%a.err" \
  "${REGISTER_ROOT}/render_registration_neighborhoods_array.slurm")
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${RENDER_JOB}" --export=ALL \
  --output="${VIS_RUN_ROOT}/finalize-%j.out" \
  --error="${VIS_RUN_ROOT}/finalize-%j.err" \
  "${REGISTER_ROOT}/finalize_registration_method_visualizations.slurm")

cat > "${VIS_RUN_ROOT}/submitted_pipeline.txt" <<EOF
analysis_dir=${ANALYSIS_DIR}
data_root=${DATA_ROOT}
selection_job=${SELECT_JOB}
render_array_job=${RENDER_JOB}
finalize_job=${FINAL_JOB}
run_root=${VIS_RUN_ROOT}
EOF

cat <<EOF
Registration-method visualization pipeline submitted
  selection:    ${SELECT_JOB}
  render array: ${RENDER_JOB}_[0-7%4]
  finalize:     ${FINAL_JOB}
  output:       ${VIS_RUN_ROOT}

Monitor:
  squeue -j ${SELECT_JOB},${RENDER_JOB},${FINAL_JOB} -o '%.18i %.24j %.10T %.12M %.40R'
EOF
