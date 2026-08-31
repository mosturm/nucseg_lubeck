#!/bin/bash
#SBATCH --job-name=cell-vessel-viz
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=2h
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-vessel-viz-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-vessel-viz-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr}"
RAW_ZARR="${RAW_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_raw.ome.zarr}"
CELL_ZARR="${CELL_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr}"
VESSEL_ZARR="${VESSEL_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_vessels_multiscale_union.ome.zarr}"
OUTDIR="${OUTDIR:-${DATA_ROOT}/${SAMPLE_ID}_cell_vessel_center_sections}"
VESSEL_RUN_ROOT="${VESSEL_RUN_ROOT:-}"
PLANE="${PLANE:-both}"
SCALE="${SCALE:-0.5}"
X_INDEX="${X_INDEX:-}"
Y_INDEX="${Y_INDEX:-}"
HTML_MAX_POINTS="${HTML_MAX_POINTS:-250000}"
POSTPROCESS_THRESHOLDS="${POSTPROCESS_THRESHOLDS:-}"
MID_POSTPROCESS_THRESHOLDS="${MID_POSTPROCESS_THRESHOLDS:-}"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
python -c 'import plotly, scipy; print("3D dependencies OK")'

for path in "${RAW_ZARR}" "${CELL_ZARR}" "${VESSEL_ZARR}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing OME-Zarr: ${path}" >&2
        exit 1
    fi
done

if [[ -z "${VESSEL_RUN_ROOT}" ]]; then
    VESSEL_RUN_ROOT="$(python - "${PROJECT_ROOT}/runs" "${SAMPLE_ID}" <<'PY'
import sys
from pathlib import Path

runs = Path(sys.argv[1])
sample_id = sys.argv[2]
candidates = sorted(
    runs.glob(f"run_vessel_multiscale_{sample_id}_*"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(next((path for path in candidates if (path / "COMPLETED").is_file()), ""))
PY
)"
fi
if [[ -z "${VESSEL_RUN_ROOT}" || ! -f "${VESSEL_RUN_ROOT}/COMPLETED" ]]; then
    echo "No completed multiscale vessel run found; set VESSEL_RUN_ROOT." >&2
    exit 1
fi
MID_VESSEL_ZARR="${MID_VESSEL_ZARR:-${VESSEL_RUN_ROOT}/scale_masks/mid_semantic.ome.zarr}"
COARSE_VESSEL_ZARR="${COARSE_VESSEL_ZARR:-${VESSEL_RUN_ROOT}/scale_masks/coarse_semantic.ome.zarr}"
for path in "${MID_VESSEL_ZARR}" "${COARSE_VESSEL_ZARR}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing scale mask OME-Zarr: ${path}" >&2
        exit 1
    fi
done

if [[ -z "${POSTPROCESS_THRESHOLDS}" ]]; then
    POSTPROCESS_THRESHOLDS="$(python - "${PROJECT_ROOT}/runs" <<'PY'
import sys
from pathlib import Path

runs = Path(sys.argv[1])
candidates = sorted(
    runs.glob("run_vessel_analytical_postprocess_*/calibration/coarse_deployment_thresholds.json"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(next((path for path in candidates if (path.parents[1] / "COMPLETED").is_file()), ""))
PY
)"
fi
if [[ -z "${POSTPROCESS_THRESHOLDS}" || ! -f "${POSTPROCESS_THRESHOLDS}" ]]; then
    echo "No calibrated coarse deployment rule found; set POSTPROCESS_THRESHOLDS." >&2
    exit 1
fi
if [[ -z "${MID_POSTPROCESS_THRESHOLDS}" ]]; then
    MID_POSTPROCESS_THRESHOLDS="$(dirname "${POSTPROCESS_THRESHOLDS}")/deployment_thresholds.json"
fi
if [[ ! -f "${MID_POSTPROCESS_THRESHOLDS}" ]]; then
    echo "No saved mid deployment rule found: ${MID_POSTPROCESS_THRESHOLDS}" >&2
    exit 1
fi
python - "${MID_POSTPROCESS_THRESHOLDS}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
if config.get("schema_version") != 2:
    raise SystemExit("Mid thresholds must use schema version 2")
rule = config.get("scale_specific_rules", {}).get("mid")
if not rule:
    raise SystemExit("No scale_specific_rules.mid entry in deployment_thresholds.json")
print(f"Mid rule: {rule}")
PY

EXTRA_ARGS=()
if [[ -n "${X_INDEX}" ]]; then
    EXTRA_ARGS+=(--x-index "${X_INDEX}")
fi
if [[ -n "${Y_INDEX}" ]]; then
    EXTRA_ARGS+=(--y-index "${Y_INDEX}")
fi

python -u "${REPO_ROOT}/visualize_cell_vessel_overlay_ome_zarr.py" \
    --raw "${RAW_ZARR}" \
    --cells "${CELL_ZARR}" \
    --vessels "${VESSEL_ZARR}" \
    --outdir "${OUTDIR}" \
    --plane "${PLANE}" \
    --scale "${SCALE}" \
    --low-percentile 0.5 \
    --high-percentile 99.5 \
    --cell-color 255,20,20 \
    --vessel-color 20,100,255 \
    --cell-alpha 0.62 \
    --vessel-alpha 0.72 \
    --mid-vessels "${MID_VESSEL_ZARR}" \
    --coarse-vessels "${COARSE_VESSEL_ZARR}" \
    --html-max-points "${HTML_MAX_POINTS}" \
    --cylinder-diameter-mm 1.2 \
    --cylinder-height-mm 1.2 \
    --coarse-postprocess-thresholds "${POSTPROCESS_THRESHOLDS}" \
    --mid-postprocess-thresholds "${MID_POSTPROCESS_THRESHOLDS}" \
    "${EXTRA_ARGS[@]}"

echo "Cell/vessel visualization output: ${OUTDIR}"
echo "Interactive mid+coarse vessel HTML: ${OUTDIR}/vessels_mid_coarse_3d.html"
echo "Coarse postprocessing rule: ${POSTPROCESS_THRESHOLDS}"
echo "Mid postprocessing rule: ${MID_POSTPROCESS_THRESHOLDS}"
