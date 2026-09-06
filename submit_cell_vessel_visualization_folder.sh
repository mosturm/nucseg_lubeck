#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
SLURM_SCRIPT="${REPO_ROOT}/slurmscripts/visualize_cell_vessel_overlay.slurm"
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
skipped=0
incomplete=0
for raw_zarr in "${raw_inputs[@]}"; do
    [[ -d "${raw_zarr}" ]] || continue
    filename="$(basename "${raw_zarr}")"
    sample_id="${filename%_raw.ome.zarr}"

    merged_cells="${INPUT_DIR}/${sample_id}_cells_cpneg2p25_instances_merged.ome.zarr"
    distributed_cells="${INPUT_DIR}/${sample_id}_cells_cpneg2p25.ome.zarr"
    distributed_sidecar="${distributed_cells%.zarr}.json"
    if [[ -d "${merged_cells}" ]]; then
        cell_zarr="${merged_cells}"
        cell_variant="instances_merged"
    elif [[ -d "${distributed_cells}" && -f "${distributed_sidecar}" ]]; then
        cell_zarr="${distributed_cells}"
        cell_variant="distributed"
    else
        echo "Skipping ${sample_id}: completed cell segmentation not found" >&2
        incomplete=$((incomplete + 1))
        continue
    fi

    vessel_zarr="${INPUT_DIR}/${sample_id}_vessels_multiscale_union.ome.zarr"
    vessel_run_root="$(python3 - "${PROJECT_ROOT}/runs" "${sample_id}" "${vessel_zarr}" <<'PY'
import sys
from pathlib import Path

runs = Path(sys.argv[1])
sample_id = sys.argv[2]
expected_output = Path(sys.argv[3]).resolve()
candidates = sorted(
    runs.glob(f"run_vessel_multiscale_{sample_id}_*"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for path in candidates:
    config = path / "submitted_pipeline.txt"
    if not (path / "COMPLETED").is_file() or not config.is_file():
        continue
    values = dict(
        line.split("=", 1)
        for line in config.read_text().splitlines()
        if "=" in line
    )
    if Path(values.get("output", "")).resolve() == expected_output:
        print(path)
        break
PY
)"
    if [[ ! -d "${vessel_zarr}" || -z "${vessel_run_root}" ]]; then
        echo "Skipping ${sample_id}: completed multiscale vessel segmentation not found" >&2
        incomplete=$((incomplete + 1))
        continue
    fi

    outdir="${INPUT_DIR}/${sample_id}_cell_vessel_center_sections"
    if [[ -f "${outdir}/overlay_metadata.json" && "${SKIP_EXISTING}" == 1 ]]; then
        echo "Skipping existing visualization: ${outdir}"
        skipped=$((skipped + 1))
        continue
    fi

    job_id="$(sbatch --parsable \
        --export="ALL,SAMPLE_ID=${sample_id},DATA_ROOT=${INPUT_DIR},RAW_ZARR=${raw_zarr},CELL_ZARR=${cell_zarr},VESSEL_ZARR=${vessel_zarr},VESSEL_RUN_ROOT=${vessel_run_root},OUTDIR=${outdir}" \
        "${SLURM_SCRIPT}")"
    echo "Submitted ${sample_id}: job=${job_id} cells=${cell_variant} output=${outdir}"
    submitted=$((submitted + 1))
done

echo
echo "Combined cell/vessel visualization submission complete"
echo "  submitted: ${submitted}"
echo "  skipped:   ${skipped}"
echo "  incomplete: ${incomplete}"
