#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
RUNS_ROOT="${PROJECT_ROOT}/runs"
COORDINATE_TABLE="${COORDINATE_TABLE:-${REGISTER_ROOT}/detailed_questions/H2P7_Laesion_H2P2_Kontrolle_Koordinaten.csv}"

if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  while IFS= read -r candidate; do
    if [[ -d "${candidate}/analysis_cov_80/raw_voxel_tables" \
       && -d "${candidate}/features/per_cell_radius_tables" \
       && -f "${candidate}/analysis_cov_80/analysis_report.json" \
       && ( -f "${candidate}/COMPLETED" || -f "${candidate}/POSTHOC_PLOTS_COMPLETED" ) ]]; then
      SOURCE_RUN_ROOT="${candidate}"
      break
    fi
  done < <(find "${RUNS_ROOT}" -maxdepth 1 -mindepth 1 -type d \
    -name 'run_full_mri_segmentation_*' -printf '%T@ %p\n' \
    | sort -nr | cut -d' ' -f2-)
fi

test -n "${SOURCE_RUN_ROOT:-}" || {
  echo "No completed full MRI-segmentation run with raw and per-cell tables found" >&2
  exit 2
}
test -f "${COORDINATE_TABLE}" || {
  echo "Missing coordinate request table: ${COORDINATE_TABLE}" >&2
  exit 2
}

export PROJECT_ROOT SOURCE_RUN_ROOT COORDINATE_TABLE
JOB_ID=$(sbatch --parsable --export=ALL "${REGISTER_ROOT}/run_detailed_collaborator_voxels.slurm")
OUTPUT_ROOT="${RUNS_ROOT}/run_detailed_collaborator_voxels_${JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/submitted_job.txt" <<EOF
job_id=${JOB_ID}
source_run=${SOURCE_RUN_ROOT}
coordinate_table=${COORDINATE_TABLE}
output_root=${OUTPUT_ROOT}
EOF

cat <<EOF
Detailed collaborator plotting job submitted
  job:        ${JOB_ID}
  source run: ${SOURCE_RUN_ROOT}
  output:     ${OUTPUT_ROOT}

Monitor with:
  squeue -j ${JOB_ID} -o '%.18i %.24j %.10T %.12M %.40R'
  tail -f '${OUTPUT_ROOT}/workflow.log'
EOF
