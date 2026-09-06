#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

: "${FINAL_RUN_ROOT:?Set FINAL_RUN_ROOT to the completed final vessel-model run}"
test -d "${INPUT_DIR}"
test -f "${REPO_ROOT}/submit_vessel_multiscale_inference.sh"
test -f "${FINAL_RUN_ROOT}/COMPLETED"
test -f "${FINAL_RUN_ROOT}/deployment_config.json"

batch_tag="$(date +%Y%m%d_%H%M%S)_$$"
batch_log="${PROJECT_ROOT}/runs/vessel_multiscale_batch_${batch_tag}.log"
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
    [[ -d "${input_zarr}" ]] || continue
    if [[ ! -f "${input_zarr}/zarr.json" && ! -f "${input_zarr}/.zgroup" ]]; then
        echo "Skipping invalid OME-Zarr (group metadata missing): ${input_zarr}" >&2
        invalid=$((invalid + 1))
        continue
    fi

    filename="$(basename "${input_zarr}")"
    sample_id="${filename%_raw.ome.zarr}"
    output_zarr="${INPUT_DIR}/${sample_id}_vessels_multiscale_union.ome.zarr"
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
    FINAL_RUN_ROOT="${FINAL_RUN_ROOT}" \
        bash "${REPO_ROOT}/submit_vessel_multiscale_inference.sh"
    submitted=$((submitted + 1))
done

echo
echo "Folder submission complete"
echo "  input directory: ${INPUT_DIR}"
echo "  final model run: ${FINAL_RUN_ROOT}"
echo "  submitted:       ${submitted}"
echo "  skipped:         ${skipped}"
echo "  invalid:         ${invalid}"
echo "  batch log:       ${batch_log}"
