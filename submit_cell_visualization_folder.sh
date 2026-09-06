#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SLURM_SCRIPT="${REPO_ROOT}/slurmscripts/visualize_cells_center_sections.slurm"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

test -d "${INPUT_DIR}"
test -f "${SLURM_SCRIPT}"

shopt -s nullglob
raw_inputs=("${INPUT_DIR}"/*_raw.ome.zarr)
if [[ ${#raw_inputs[@]} -eq 0 ]]; then
    echo "No *_raw.ome.zarr inputs found in ${INPUT_DIR}" >&2
    exit 1
fi

submitted=0
missing_labels=0
skipped=0
for raw_zarr in "${raw_inputs[@]}"; do
    [[ -d "${raw_zarr}" ]] || continue
    filename="$(basename "${raw_zarr}")"
    sample_id="${filename%_raw.ome.zarr}"
    merged_labels="${INPUT_DIR}/${sample_id}_cells_cpneg2p25_instances_merged.ome.zarr"
    distributed_labels="${INPUT_DIR}/${sample_id}_cells_cpneg2p25.ome.zarr"
    distributed_sidecar="${distributed_labels%.zarr}.json"

    if [[ -d "${merged_labels}" ]]; then
        label_zarr="${merged_labels}"
        variant="instances_merged"
    elif [[ -d "${distributed_labels}" && -f "${distributed_sidecar}" ]]; then
        label_zarr="${distributed_labels}"
        variant="distributed"
    elif [[ -d "${distributed_labels}" ]]; then
        echo "Skipping ${sample_id}: distributed output is not finalized yet" >&2
        missing_labels=$((missing_labels + 1))
        continue
    else
        echo "Skipping ${sample_id}: no completed cell label Zarr found" >&2
        missing_labels=$((missing_labels + 1))
        continue
    fi

    outdir="${INPUT_DIR}/${sample_id}_cell_visualization_${variant}"
    if [[ -f "${outdir}/visualization_metadata.json" && "${SKIP_EXISTING}" == 1 ]]; then
        echo "Skipping existing visualization: ${outdir}"
        skipped=$((skipped + 1))
        continue
    fi

    job_id="$(sbatch --parsable \
        --export="ALL,SAMPLE_ID=${sample_id},DATA_ROOT=${INPUT_DIR},RAW_ZARR=${raw_zarr},LABEL_ZARR=${label_zarr},OUTDIR=${outdir}" \
        "${SLURM_SCRIPT}")"
    echo "Submitted ${sample_id}: job=${job_id} labels=${variant} output=${outdir}"
    submitted=$((submitted + 1))
done

echo
echo "Cell visualization submission complete"
echo "  submitted:      ${submitted}"
echo "  skipped:        ${skipped}"
echo "  missing/incomplete labels: ${missing_labels}"
