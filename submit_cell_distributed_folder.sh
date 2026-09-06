#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CELLPROB_THRESHOLD="${CELLPROB_THRESHOLD:--2.25}"

test -d "${INPUT_DIR}"
test -f "${REPO_ROOT}/submit_cell_distributed_inference.sh"

if [[ "${CELLPROB_THRESHOLD}" == -* ]]; then
  cellprob_tag="neg${CELLPROB_THRESHOLD#-}"
else
  cellprob_tag="${CELLPROB_THRESHOLD}"
fi
cellprob_tag="${cellprob_tag//./p}"

batch_tag="$(date +%Y%m%d_%H%M%S)_$$"
batch_log="${PROJECT_ROOT}/runs/cell_distributed_batch_${batch_tag}.log"
exec > >(tee -a "${batch_log}") 2>&1

shopt -s nullglob
inputs=("${INPUT_DIR}"/*_raw.ome.zarr)
if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "No *_raw.ome.zarr inputs found in ${INPUT_DIR}" >&2
  exit 1
fi

submitted=0
skipped=0
invalid=0
for input_zarr in "${inputs[@]}"; do
  if [[ ! -d "${input_zarr}" ]]; then
    continue
  fi
  if [[ ! -f "${input_zarr}/zarr.json" && ! -f "${input_zarr}/.zgroup" ]]; then
    echo "Skipping invalid OME-Zarr (group metadata missing): ${input_zarr}" >&2
    invalid=$((invalid + 1))
    continue
  fi
  basename="$(basename "${input_zarr}")"
  sample_id="${basename%_raw.ome.zarr}"
  output_zarr="${INPUT_DIR}/${sample_id}_cells_cp${cellprob_tag}.ome.zarr"
  if [[ -e "${output_zarr}" && "${SKIP_EXISTING}" == 1 ]]; then
    echo "Skipping existing output: ${output_zarr}"
    skipped=$((skipped + 1))
    continue
  fi

  echo
  echo "Submitting sample: ${sample_id}"
  INPUT_ZARR="${input_zarr}" \
  SAMPLE_ID="${sample_id}" \
  OUTPUT_ZARR="${output_zarr}" \
  CELLPROB_THRESHOLD="${CELLPROB_THRESHOLD}" \
    bash "${REPO_ROOT}/submit_cell_distributed_inference.sh"
  submitted=$((submitted + 1))
done

echo
echo "Folder submission complete"
echo "  input directory: ${INPUT_DIR}"
echo "  submitted:       ${submitted}"
echo "  skipped:         ${skipped}"
echo "  invalid:         ${invalid}"
echo "  batch log:       ${batch_log}"
