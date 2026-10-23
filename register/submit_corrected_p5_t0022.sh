#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
REGISTER_ROOT="${REPO_ROOT}/register"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
INPUT_ROOT="${INPUT_ROOT:-${REGISTER_ROOT}/register_full_input}"
CORRECTED_RAW="${CORRECTED_RAW:-${DATA_ROOT}/tomo_reco_id0004_t0022_raw_correct_axrot.ome.zarr}"
SAMPLE_ID="tomo_reco_id0004_t0022"

test -d "${CORRECTED_RAW}" || { echo "Missing corrected raw OME-Zarr: ${CORRECTED_RAW}" >&2; exit 2; }
test -d "${INPUT_ROOT}/fuer_Mo" || { echo "Missing registration input: ${INPUT_ROOT}/fuer_Mo" >&2; exit 2; }

if [[ -z "${SOURCE_RUN_ROOT:-}" ]]; then
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/COMPLETED" && -d "${candidate}/features/partials" ]]; then
      completed="$(find "${candidate}/features/partials" -mindepth 2 -maxdepth 2 -name COMPLETED | wc -l)"
      if [[ "${completed}" -eq 12 ]]; then
        SOURCE_RUN_ROOT="${candidate}"
        break
      fi
    fi
  done < <(
    find "${PROJECT_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d \
      -name 'run_full_mri_segmentation_*' -printf '%T@ %p\n' \
      | sort -rn | cut -d' ' -f2-
  )
fi
test -n "${SOURCE_RUN_ROOT:-}" || { echo "No completed full-registration run with 12 partials found." >&2; exit 2; }

if [[ -z "${FINAL_RUN_ROOT:-}" ]]; then
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/COMPLETED" && -f "${candidate}/deployment_config.json" ]]; then
      FINAL_RUN_ROOT="${candidate}"
      break
    fi
  done < <(
    find "${PROJECT_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d \
      -name 'run_vessel_final_all_seed46_*' -printf '%T@ %p\n' \
      | sort -rn | cut -d' ' -f2-
  )
fi
test -n "${FINAL_RUN_ROOT:-}" || { echo "No completed final vessel-model run found." >&2; exit 2; }

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
RUN_ROOT="${PROJECT_ROOT}/runs/run_corrected_p5_t0022_${RUN_TAG}"
STAGING_DIR="${RUN_ROOT}/mask_staging"
mkdir -p "${STAGING_DIR}"

CELL_OUTPUT="${STAGING_DIR}/${SAMPLE_ID}_cells_cpneg2p25.ome.zarr"
VESSEL_OUTPUT="${STAGING_DIR}/${SAMPLE_ID}_vessels_multiscale_union.ome.zarr"

cell_submit="$(
  (
    # Never inherit a stale vessel/custom MODEL_PATH into cell inference.
    unset MODEL_PATH
    if [[ -n "${CELL_MODEL_PATH:-}" ]]; then
      export MODEL_PATH="${CELL_MODEL_PATH}"
    fi
    PROJECT_ROOT="${PROJECT_ROOT}" \
    INPUT_ZARR="${CORRECTED_RAW}" \
    SAMPLE_ID="${SAMPLE_ID}_correct_axrot" \
    OUTPUT_ZARR="${CELL_OUTPUT}" \
    CELLPROB_THRESHOLD=-2.25 \
    MIN_SIZE=15 \
    FLOW3D_SMOOTH=0 \
    bash "${REPO_ROOT}/submit_cell_distributed_inference.sh"
  ) 2>&1
)"
printf '%s\n' "${cell_submit}"
CELL_FINAL_JOB="$(printf '%s\n' "${cell_submit}" | awk '$1 == "finalize:" {print $2; exit}')"

vessel_submit="$(
  PROJECT_ROOT="${PROJECT_ROOT}" \
  INPUT_ZARR="${CORRECTED_RAW}" \
  SAMPLE_ID="${SAMPLE_ID}_correct_axrot" \
  FINAL_RUN_ROOT="${FINAL_RUN_ROOT}" \
  OUTPUT_ZARR="${VESSEL_OUTPUT}" \
  bash "${REPO_ROOT}/submit_vessel_multiscale_inference.sh" 2>&1
)"
printf '%s\n' "${vessel_submit}"
VESSEL_FINAL_JOB="$(printf '%s\n' "${vessel_submit}" | awk '$1 == "final" && $2 == "union:" {print $3; exit}')"

test -n "${CELL_FINAL_JOB}" || { echo "Could not parse cell finalize job ID" >&2; exit 2; }
test -n "${VESSEL_FINAL_JOB}" || { echo "Could not parse vessel finalize job ID" >&2; exit 2; }

common_export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},INPUT_ROOT=${INPUT_ROOT},CORRECTION_RUN_ROOT=${RUN_ROOT},STAGING_DIR=${STAGING_DIR}"
PROMOTE_JOB="$(sbatch --parsable \
  --dependency="afterok:${CELL_FINAL_JOB}:${VESSEL_FINAL_JOB}" \
  --export="${common_export}" \
  "${REGISTER_ROOT}/promote_corrected_p5_t0022.slurm")"

EXTRACT_JOB="$(sbatch --parsable \
  --dependency="afterok:${PROMOTE_JOB}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},INPUT_ROOT=${INPUT_ROOT},SOURCE_RUN_ROOT=${SOURCE_RUN_ROOT},RUN_ROOT=${RUN_ROOT}" \
  "${REGISTER_ROOT}/extract_corrected_p5_t0022.slurm")"

ANALYSIS_JOB="$(sbatch --parsable \
  --dependency="afterok:${EXTRACT_JOB}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},INPUT_ROOT=${INPUT_ROOT},RUN_ROOT=${RUN_ROOT}" \
  "${REGISTER_ROOT}/run_full_mri_segmentation.slurm")"

cat > "${RUN_ROOT}/submitted_pipeline.txt" <<EOF
run_root=${RUN_ROOT}
corrected_raw=${CORRECTED_RAW}
source_full_run=${SOURCE_RUN_ROOT}
final_vessel_model_run=${FINAL_RUN_ROOT}
staging_dir=${STAGING_DIR}
cell_output=${CELL_OUTPUT}
vessel_output=${VESSEL_OUTPUT}
cell_finalize_job=${CELL_FINAL_JOB}
vessel_finalize_job=${VESSEL_FINAL_JOB}
promote_job=${PROMOTE_JOB}
extract_corrected_job=${EXTRACT_JOB}
analysis_job=${ANALYSIS_JOB}
EOF

cat <<EOF
Corrected P5/0022 segmentation and registration pipeline submitted
  corrected raw: ${CORRECTED_RAW}
  source run:    ${SOURCE_RUN_ROOT}
  new run root:  ${RUN_ROOT}
  cell finalize: ${CELL_FINAL_JOB}
  vessel union:  ${VESSEL_FINAL_JOB}
  mask promote:  ${PROMOTE_JOB}
  P5/0022 scan:  ${EXTRACT_JOB}
  full analysis: ${ANALYSIS_JOB}

Monitor:
  squeue -j ${CELL_FINAL_JOB},${VESSEL_FINAL_JOB},${PROMOTE_JOB},${EXTRACT_JOB},${ANALYSIS_JOB} -o '%.18i %.24j %.10T %.12M %.40R'
EOF
