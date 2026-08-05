#!/bin/bash
#SBATCH --job-name=cell-zarr-infer
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-zarr-infer-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-zarr-infer-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
INPUT_ZARR="${INPUT_ZARR:-${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr/${SAMPLE_ID}_raw.ome.zarr}"
ARRAY_KEY="${ARRAY_KEY:-0}"
BLOCK_SHAPE="${BLOCK_SHAPE:-40,256,256}"
HALO="${HALO:-5,32,32}"
OUTPUT_CHUNKS="${OUTPUT_CHUNKS:-8,256,256}"
ANISOTROPY="${ANISOTROPY:-1.0}"
CELLPROB_THRESHOLD="${CELLPROB_THRESHOLD:--2.25}"
MIN_SIZE="${MIN_SIZE:-15}"
FLOW3D_SMOOTH="${FLOW3D_SMOOTH:-0}"
PROCESS_FRACTION="${PROCESS_FRACTION:-1.0}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
PREVIEW_MODE="${PREVIEW_MODE:-random}"
PREVIEW_SEED="${PREVIEW_SEED:-46}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
INPUT_PARENT="$(dirname "${INPUT_ZARR}")"
if [[ "${CELLPROB_THRESHOLD}" == -* ]]; then
    CELLPROB_TAG="neg${CELLPROB_THRESHOLD#-}"
else
    CELLPROB_TAG="${CELLPROB_THRESHOLD}"
fi
CELLPROB_TAG="${CELLPROB_TAG//./p}"

RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_inference_${SAMPLE_ID}_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${RUN_ROOT}/COMPLETED"
        echo "OME-Zarr cell inference completed successfully."
    else
        printf "exit_code=%s\n" "${status}" > "${RUN_ROOT}/FAILED"
        echo "OME-Zarr cell inference failed with exit code ${status}." >&2
    fi
    exit "${status}"
}
trap finish EXIT

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 1
    fi
}

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/.cellpose/models"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND="Agg"
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1

require_file "${VENV_ACTIVATE}"
require_dir "${INPUT_ZARR}"

# Prefer a copy in nucseg_lubeck, but also support the established organoid repo.
if [[ -z "${INFER_SCRIPT:-}" ]]; then
    for candidate in \
        "${REPO_ROOT}/infer_ome_zarr.py" \
        "${PROJECT_ROOT}/repo/organoid_seg/infer_ome_zarr.py"; do
        if [[ -f "${candidate}" ]]; then
            INFER_SCRIPT="${candidate}"
            break
        fi
    done
fi
if [[ -z "${INFER_SCRIPT:-}" ]]; then
    echo "infer_ome_zarr.py was not found in nucseg_lubeck or organoid_seg." >&2
    exit 1
fi
require_file "${INFER_SCRIPT}"

# Find the newest completed all-data run containing the selected epoch-51 model.
if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(
        python -c 'import sys; from pathlib import Path; root=Path(sys.argv[1]); runs=sorted(root.glob("run_cell_final_all_seed46_*"), key=lambda p:p.stat().st_mtime, reverse=True); models=[p/"models"/"cell_final_all_120ep_epoch_0050" for p in runs if (p/"COMPLETED").is_file() and (p/"models"/"cell_final_all_120ep_epoch_0050").is_file()]; print(models[0] if models else "")' \
            "${PROJECT_ROOT}/runs"
    )"
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "No completed final epoch-51 model was found." >&2
    echo "Set MODEL_PATH explicitly with sbatch --export=ALL,MODEL_PATH=/path/to/model ..." >&2
    exit 1
fi
require_file "${MODEL_PATH}"

limited_run="$(
    python -c 'import sys; print(int(float(sys.argv[1]) < 1.0 or int(sys.argv[2]) > 0))' \
        "${PROCESS_FRACTION}" "${MAX_BLOCKS}"
)"
if [[ -z "${OUTPUT_ZARR:-}" ]]; then
    if [[ "${limited_run}" == "1" ]]; then
        OUTPUT_ZARR="${INPUT_PARENT}/${SAMPLE_ID}_cells_cp${CELLPROB_TAG}_preview_${SLURM_JOB_ID}.ome.zarr"
    else
        OUTPUT_ZARR="${INPUT_PARENT}/${SAMPLE_ID}_cells_cp${CELLPROB_TAG}.ome.zarr"
    fi
fi

if [[ -e "${OUTPUT_ZARR}" ]]; then
    if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
        echo "Output already exists: ${OUTPUT_ZARR}" >&2
        echo "Choose OUTPUT_ZARR or set ALLOW_OVERWRITE=1 explicitly." >&2
        exit 1
    fi
    echo "WARNING: infer_ome_zarr.py will overwrite existing output: ${OUTPUT_ZARR}"
fi

python -c 'import cellpose, torch, zarr; print("Cellpose import: OK"); print("Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("Zarr:", zarr.__version__)'
nvidia-smi

cat > "${RUN_ROOT}/inference_config.txt" <<EOF
input=${INPUT_ZARR}
array_key=${ARRAY_KEY}
model=${MODEL_PATH}
output=${OUTPUT_ZARR}
block_shape=${BLOCK_SHAPE}
halo=${HALO}
output_chunks=${OUTPUT_CHUNKS}
anisotropy=${ANISOTROPY}
cellprob_threshold=${CELLPROB_THRESHOLD}
min_size=${MIN_SIZE}
flow3D_smooth=${FLOW3D_SMOOTH}
skip_policy=exactly_zero_core_only
process_fraction=${PROCESS_FRACTION}
max_blocks=${MAX_BLOCKS}
preview_mode=${PREVIEW_MODE}
preview_seed=${PREVIEW_SEED}
EOF
cp "$0" "${RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${RUN_ROOT}/python_packages.txt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Inference script: ${INFER_SCRIPT}"
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_ZARR}/${ARRAY_KEY}"
echo "Output: ${OUTPUT_ZARR}"
echo "Run metadata: ${RUN_ROOT}"

cd "${RUN_ROOT}"
srun python -u "${INFER_SCRIPT}" \
    --input "${INPUT_ZARR}" \
    --array_key "${ARRAY_KEY}" \
    --model "${MODEL_PATH}" \
    --output "${OUTPUT_ZARR}" \
    --block_shape "${BLOCK_SHAPE}" \
    --halo "${HALO}" \
    --output_chunks "${OUTPUT_CHUNKS}" \
    --anisotropy "${ANISOTROPY}" \
    --cellprob_threshold "${CELLPROB_THRESHOLD}" \
    --min_size "${MIN_SIZE}" \
    --flow3D_smooth "${FLOW3D_SMOOTH}" \
    --skip_empty \
    --min_core_nonzero_voxels 0 \
    --min_core_nonzero_fraction 0 \
    --process_fraction "${PROCESS_FRACTION}" \
    --max_blocks "${MAX_BLOCKS}" \
    --preview_mode "${PREVIEW_MODE}" \
    --preview_seed "${PREVIEW_SEED}" \
    --pass_channels \
    2>&1 | tee "${RUN_ROOT}/infer_ome_zarr.log"

# Count binary foreground blockwise without loading the full output into RAM.
python - "${OUTPUT_ZARR}" "${RUN_ROOT}/foreground_volume_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import zarr

zarr_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
root = zarr.open_group(str(zarr_path), mode="r")
arr = root["0"]
chunks = tuple(int(v) for v in arr.chunks)
foreground = 0
for z0 in range(0, arr.shape[0], chunks[0]):
    for y0 in range(0, arr.shape[1], chunks[1]):
        for x0 in range(0, arr.shape[2], chunks[2]):
            block = np.asarray(
                arr[
                    z0 : min(arr.shape[0], z0 + chunks[0]),
                    y0 : min(arr.shape[1], y0 + chunks[1]),
                    x0 : min(arr.shape[2], x0 + chunks[2]),
                ]
            )
            foreground += int(np.count_nonzero(block))

total = int(np.prod(arr.shape, dtype=np.int64))
summary = {
    "output": str(zarr_path.resolve()),
    "shape_zyx": [int(v) for v in arr.shape],
    "chunks_zyx": list(chunks),
    "foreground_voxels": foreground,
    "total_voxels": total,
    "foreground_fraction": foreground / total if total else 0.0,
    "note": "Foreground volume is the count of label voxels > 0. Instance IDs split across block boundaries do not affect this binary volume.",
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "Inference output: ${OUTPUT_ZARR}"
echo "Foreground summary: ${RUN_ROOT}/foreground_volume_summary.json"
