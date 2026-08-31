#!/bin/bash
#SBATCH --job-name=cell-zarr-viz
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-zarr-viz-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-zarr-viz-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
DATA_ROOT="${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr"
RAW_ZARR="${RAW_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_raw.ome.zarr}"
LABEL_ZARR="${LABEL_ZARR:-${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr}"
OUTDIR="${OUTDIR:-${DATA_ROOT}/${SAMPLE_ID}_cell_visualization_instances_merged}"
PLANE="${PLANE:-both}"
SCALE="${SCALE:-0.5}"
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.62}"
COLOR="${COLOR:-255,20,20}"
X_INDEX="${X_INDEX:-}"
Y_INDEX="${Y_INDEX:-}"
PHYSICAL_X_UM="${PHYSICAL_X_UM:-1200}"
PHYSICAL_Y_UM="${PHYSICAL_Y_UM:-1200}"
PHYSICAL_Z_UM="${PHYSICAL_Z_UM:-1200}"
VOXEL_Z_UM="${VOXEL_Z_UM:-}"
SPATIAL_BIN_UM="${SPATIAL_BIN_UM:-250}"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTDIR}"

EXTRA_ARGS=()
if [[ -n "${X_INDEX}" ]]; then
    EXTRA_ARGS+=(--x_index "${X_INDEX}")
fi
if [[ -n "${Y_INDEX}" ]]; then
    EXTRA_ARGS+=(--y_index "${Y_INDEX}")
fi
if [[ -n "${VOXEL_Z_UM}" ]]; then
    EXTRA_ARGS+=(--voxel_z_um "${VOXEL_Z_UM}")
fi

python -u "${REPO_ROOT}/visualize_semantic_ome_zarr.py" \
    --raw "${RAW_ZARR}" \
    --labels "${LABEL_ZARR}" \
    --raw_key 0 \
    --label_key 0 \
    --outdir "${OUTDIR}" \
    --plane "${PLANE}" \
    --scale "${SCALE}" \
    --low_percentile 0.5 \
    --high_percentile 99.5 \
    --overlay_alpha "${OVERLAY_ALPHA}" \
    --color "${COLOR}" \
    --physical_x_um "${PHYSICAL_X_UM}" \
    --physical_y_um "${PHYSICAL_Y_UM}" \
    --physical_z_um "${PHYSICAL_Z_UM}" \
    --spatial_bin_um "${SPATIAL_BIN_UM}" \
    --scan_block 40,256,256 \
    --histogram_bins 60 \
    "${EXTRA_ARGS[@]}"

echo "Visualization output: ${OUTDIR}"
