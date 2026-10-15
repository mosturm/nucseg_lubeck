#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
INPUT_ROOT="${INPUT_ROOT:-${REGISTER_ROOT}/register_full_input}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
export PROJECT_ROOT INPUT_ROOT DATA_ROOT

test -d "${INPUT_ROOT}/fuer_Mo" || { echo "Missing extracted input: ${INPUT_ROOT}/fuer_Mo" >&2; exit 2; }

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
RUN_ROOT="${PROJECT_ROOT}/runs/run_full_mri_segmentation_${RUN_TAG}"
mkdir -p "${RUN_ROOT}"
export RUN_ROOT

MERGE_JOB=$(sbatch --parsable --export=ALL "${REGISTER_ROOT}/merge_full_cell_instances_array.slurm")
EXTRACT_JOB=$(sbatch --parsable --dependency="afterok:${MERGE_JOB}" --export=ALL "${REGISTER_ROOT}/extract_full_features_array.slurm")
FULL_JOB=$(sbatch --parsable --dependency="afterok:${EXTRACT_JOB}" --export=ALL "${REGISTER_ROOT}/run_full_mri_segmentation.slurm")

cat > "${RUN_ROOT}/submitted_pipeline.txt" <<EOF
run_root=${RUN_ROOT}
input_root=${INPUT_ROOT}
data_root=${DATA_ROOT}
merge_job=${MERGE_JOB}
extract_job=${EXTRACT_JOB}
analysis_job=${FULL_JOB}
EOF

cat <<EOF
Full registration pipeline submitted
  instance merges: ${MERGE_JOB}_[0-11%4]
  extraction:      ${EXTRACT_JOB}_[0-11%4]
  full analysis:   ${FULL_JOB}
  final run root:  ${RUN_ROOT}

Monitor:
  squeue -j ${MERGE_JOB},${EXTRACT_JOB},${FULL_JOB} -o '%.18i %.24j %.10T %.12M %.40R'
EOF
