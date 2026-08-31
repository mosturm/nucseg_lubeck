#!/bin/bash
#SBATCH --job-name=all-scales-3d
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/all-scales-3d-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/all-scales-3d-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr}"
CELL_ZARR="${CELL_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr}"
VESSEL_RUN_ROOT="${VESSEL_RUN_ROOT:-}"
ANALYTICAL_ROOT="${ANALYTICAL_ROOT:-}"
OUTDIR="${OUTDIR:-${DATA_ROOT}/${SAMPLE_ID}_cells_vessels_all_scales_3d}"
MAX_VESSEL_POINTS="${MAX_VESSEL_POINTS:-120000}"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
python -c 'import matplotlib, pandas, plotly, scipy, zarr; print("3D dependencies OK")'

if [[ -z "${VESSEL_RUN_ROOT}" ]]; then
    VESSEL_RUN_ROOT="$(python - "${PROJECT_ROOT}/runs" "${SAMPLE_ID}" <<'PY'
import sys
from pathlib import Path
runs, sample = Path(sys.argv[1]), sys.argv[2]
items = sorted(runs.glob(f"run_vessel_multiscale_{sample}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
print(next((p for p in items if (p / "COMPLETED").is_file()), ""))
PY
)"
fi
if [[ -z "${ANALYTICAL_ROOT}" ]]; then
    ANALYTICAL_ROOT="$(python - "${PROJECT_ROOT}/runs" <<'PY'
import sys
from pathlib import Path
runs = Path(sys.argv[1])
items = sorted(runs.glob("run_vessel_analytical_postprocess_*"), key=lambda p: p.stat().st_mtime, reverse=True)
print(next((p for p in items if (p / "COMPLETED").is_file()), ""))
PY
)"
fi

test -f "${VESSEL_RUN_ROOT}/COMPLETED"
test -f "${ANALYTICAL_ROOT}/COMPLETED"
FINE_ZARR="${VESSEL_RUN_ROOT}/scale_masks/unscaled_semantic.ome.zarr"
MID_ZARR="${VESSEL_RUN_ROOT}/scale_masks/mid_semantic.ome.zarr"
COARSE_ZARR="${VESSEL_RUN_ROOT}/scale_masks/coarse_semantic.ome.zarr"
MID_THRESHOLDS="${ANALYTICAL_ROOT}/calibration/deployment_thresholds.json"
COARSE_THRESHOLDS="${ANALYTICAL_ROOT}/calibration/coarse_deployment_thresholds.json"
for path in "${CELL_ZARR}" "${FINE_ZARR}" "${MID_ZARR}" "${COARSE_ZARR}"; do
    test -d "${path}"
done
test -f "${MID_THRESHOLDS}"
test -f "${COARSE_THRESHOLDS}"

python -u "${REPO_ROOT}/visualize_cells_vessels_all_scales_3d.py" \
    --cells "${CELL_ZARR}" \
    --unscaled-vessels "${FINE_ZARR}" \
    --mid-vessels "${MID_ZARR}" \
    --coarse-vessels "${COARSE_ZARR}" \
    --mid-thresholds "${MID_THRESHOLDS}" \
    --coarse-thresholds "${COARSE_THRESHOLDS}" \
    --outdir "${OUTDIR}" \
    --physical-extent-um 1200,1200,1200 \
    --cell-scan-block 40,256,256 \
    --max-vessel-points "${MAX_VESSEL_POINTS}" \
    --histogram-bins 60 \
    --seed 46

echo "Interactive HTML: ${OUTDIR}/cells_vessels_all_scales_3d.html"
echo "5% cells + mid/coarse HTML: ${OUTDIR}/cells5pct_vessels_mid_coarse_3d.html"
echo "Log-radius histogram: ${OUTDIR}/cell_radius_histogram_log10.png"
