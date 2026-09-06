#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SLURM_ROOT="${REPO_ROOT}/slurmscripts"

: "${INPUT_ZARR:?Set INPUT_ZARR to the raw OME-Zarr}"
: "${SAMPLE_ID:?Set SAMPLE_ID}"

NUM_WORKERS=4
ARRAY_KEY="${ARRAY_KEY:-0}"
BLOCK_SHAPE="${BLOCK_SHAPE:-40x256x256}"
HALO="${HALO:-5x32x32}"
OUTPUT_CHUNKS="${OUTPUT_CHUNKS:-8x256x256}"
COMPRESSOR_LEVEL="${COMPRESSOR_LEVEL:-5}"
ANISOTROPY="${ANISOTROPY:-1.0}"
CELLPROB_THRESHOLD="${CELLPROB_THRESHOLD:--2.25}"
MIN_SIZE="${MIN_SIZE:-15}"
FLOW3D_SMOOTH="${FLOW3D_SMOOTH:-0}"
PROCESS_FRACTION="${PROCESS_FRACTION:-1.0}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
PREVIEW_MODE="${PREVIEW_MODE:-random}"
PREVIEW_SEED="${PREVIEW_SEED:-46}"

test -d "${INPUT_ZARR}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  while IFS= read -r run; do
    candidate="${run}/models/cell_final_all_120ep_epoch_0050"
    if [[ -f "${run}/COMPLETED" && -f "${candidate}" ]]; then
      MODEL_PATH="${candidate}"
      break
    fi
  done < <(
    find "${PROJECT_ROOT}/runs" \
      -mindepth 1 -maxdepth 1 -type d \
      -name 'run_cell_final_all_seed46_*' \
      -printf '%T@ %p\n' \
      | sort -rn \
      | cut -d' ' -f2-
  )
fi
if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH}" ]]; then
  echo "No completed final epoch-51 cell model was found." >&2
  echo "Set MODEL_PATH explicitly." >&2
  exit 1
fi

if [[ "${CELLPROB_THRESHOLD}" == -* ]]; then
  CELLPROB_TAG="neg${CELLPROB_THRESHOLD#-}"
else
  CELLPROB_TAG="${CELLPROB_THRESHOLD}"
fi
CELLPROB_TAG="${CELLPROB_TAG//./p}"

limited_run=0
if [[ "${PROCESS_FRACTION}" != "1" && "${PROCESS_FRACTION}" != "1.0" ]]; then
  limited_run=1
fi
if [[ "${MAX_BLOCKS}" -gt 0 ]]; then
  limited_run=1
fi
if [[ -z "${OUTPUT_ZARR:-}" ]]; then
  if [[ "${limited_run}" == 1 ]]; then
    OUTPUT_ZARR="$(dirname "${INPUT_ZARR}")/${SAMPLE_ID}_cells_cp${CELLPROB_TAG}_preview_$(date +%Y%m%d_%H%M%S).ome.zarr"
  else
    OUTPUT_ZARR="$(dirname "${INPUT_ZARR}")/${SAMPLE_ID}_cells_cp${CELLPROB_TAG}.ome.zarr"
  fi
fi
if [[ -e "${OUTPUT_ZARR}" && "${ALLOW_OVERWRITE:-0}" != 1 ]]; then
  echo "Output already exists: ${OUTPUT_ZARR}" >&2
  echo "Set OUTPUT_ZARR to a new path or explicitly set ALLOW_OVERWRITE=1." >&2
  exit 1
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_distributed_${SAMPLE_ID}_${RUN_TAG}"
mkdir -p "${RUN_ROOT}"

common_export="ALL,RUN_ROOT=${RUN_ROOT},NUM_WORKERS=${NUM_WORKERS}"
prepare_job="$(sbatch --parsable \
  --export="${common_export},INPUT_ZARR=${INPUT_ZARR},ARRAY_KEY=${ARRAY_KEY},MODEL_PATH=${MODEL_PATH},BLOCK_SHAPE=${BLOCK_SHAPE},HALO=${HALO},OUTPUT_CHUNKS=${OUTPUT_CHUNKS},COMPRESSOR_LEVEL=${COMPRESSOR_LEVEL},ANISOTROPY=${ANISOTROPY},CELLPROB_THRESHOLD=${CELLPROB_THRESHOLD},MIN_SIZE=${MIN_SIZE},FLOW3D_SMOOTH=${FLOW3D_SMOOTH},PROCESS_FRACTION=${PROCESS_FRACTION},MAX_BLOCKS=${MAX_BLOCKS},PREVIEW_MODE=${PREVIEW_MODE},PREVIEW_SEED=${PREVIEW_SEED}" \
  "${SLURM_ROOT}/prepare_cell_distributed_inference.slurm")"

gpu_job="$(sbatch --parsable \
  --dependency="afterok:${prepare_job}" \
  --array=0-3%4 \
  --export="${common_export}" \
  "${SLURM_ROOT}/infer_cell_distributed_array.slurm")"

finalize_job="$(sbatch --parsable \
  --dependency="afterok:${gpu_job}" \
  --export="${common_export},OUTPUT_ZARR=${OUTPUT_ZARR},KEEP_WORKING_LABELS=${KEEP_WORKING_LABELS:-0}" \
  "${SLURM_ROOT}/finalize_cell_distributed_inference.slurm")"

cat > "${RUN_ROOT}/submitted_pipeline.txt" <<EOF
input=${INPUT_ZARR}
sample_id=${SAMPLE_ID}
model=${MODEL_PATH}
cellprob_threshold=${CELLPROB_THRESHOLD}
min_size=${MIN_SIZE}
flow3D_smooth=${FLOW3D_SMOOTH}
block_shape=${BLOCK_SHAPE}
halo=${HALO}
output_chunks=${OUTPUT_CHUNKS}
num_workers=${NUM_WORKERS}
prepare_job=${prepare_job}
gpu_array_job=${gpu_job}
finalize_job=${finalize_job}
output=${OUTPUT_ZARR}
EOF

echo "Distributed cell pipeline submitted"
echo "  run root:    ${RUN_ROOT}"
echo "  prepare:     ${prepare_job}"
echo "  GPU array:   ${gpu_job}_[0-3] (four concurrent A100 workers)"
echo "  finalize:    ${finalize_job}"
echo "  output:      ${OUTPUT_ZARR}"
echo
echo "Monitor with:"
echo "  squeue -j ${prepare_job},${gpu_job},${finalize_job} -o '%.18i %.24j %.10T %.12M %.40R'"
