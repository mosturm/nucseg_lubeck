#!/bin/bash
#SBATCH --job-name=vessel-zarr-infer
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-zarr-infer-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-zarr-infer-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
SCALE="${SCALE:?Set SCALE to unscaled, mid, or coarse}"
case "${SCALE}" in unscaled|mid|coarse) ;; *) echo "SCALE must be unscaled, mid, or coarse" >&2; exit 2;; esac

SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
INPUT_ZARR="${INPUT_ZARR:-${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr/${SAMPLE_ID}_raw.ome.zarr}"
ARRAY_KEY="${ARRAY_KEY:-0}"
BLOCK_SHAPE="${BLOCK_SHAPE:-40,256,256}"
HALO="${HALO:-5,32,32}"
OUTPUT_CHUNKS="${OUTPUT_CHUNKS:-8,256,256}"
ANISOTROPY="${ANISOTROPY:-1.0}"
MIN_SIZE="${MIN_SIZE:-}"
FLOW3D_SMOOTH="${FLOW3D_SMOOTH:-}"
PROCESS_FRACTION="${PROCESS_FRACTION:-1.0}"
MAX_BLOCKS="${MAX_BLOCKS:-0}"
PREVIEW_MODE="${PREVIEW_MODE:-random}"
PREVIEW_SEED="${PREVIEW_SEED:-46}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_inference_${SCALE}_${SAMPLE_ID}_${SLURM_JOB_ID}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1
finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then touch "${RUN_ROOT}/COMPLETED"; else printf "exit_code=%s\n" "${status}" > "${RUN_ROOT}/FAILED"; fi
    exit "${status}"
}
trap finish EXIT
require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cellpose/models"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND=Agg
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1
require_file "${VENV_ACTIVATE}"
require_dir "${INPUT_ZARR}"

if [[ -z "${INFER_SCRIPT:-}" ]]; then
    for candidate in "${REPO_ROOT}/infer_ome_zarr.py" "${PROJECT_ROOT}/repo/organoid_seg/infer_ome_zarr.py"; do
        [[ -f "${candidate}" ]] && { INFER_SCRIPT="${candidate}"; break; }
    done
fi
[[ -n "${INFER_SCRIPT:-}" ]] || { echo "infer_ome_zarr.py not found" >&2; exit 1; }

if [[ -z "${FINAL_RUN_ROOT:-}" ]]; then
    FINAL_RUN_ROOT="$(python -c 'import sys; from pathlib import Path; r=Path(sys.argv[1]); xs=sorted(r.glob("run_vessel_final_all_seed46_*"),key=lambda p:p.stat().st_mtime,reverse=True); print(next((p for p in xs if (p/"COMPLETED").is_file() and (p/"deployment_config.json").is_file()),""))' "${PROJECT_ROOT}/runs")"
fi
[[ -n "${FINAL_RUN_ROOT}" ]] || { echo "No completed final vessel run found; set FINAL_RUN_ROOT" >&2; exit 1; }
DEPLOYMENT_CONFIG="${FINAL_RUN_ROOT}/deployment_config.json"
require_file "${DEPLOYMENT_CONFIG}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["model"])' "${DEPLOYMENT_CONFIG}" "${SCALE}")"
fi
if [[ -z "${CELLPROB_THRESHOLD:-}" ]]; then
    CELLPROB_THRESHOLD="$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["cellprob_threshold"]), "g"))' "${DEPLOYMENT_CONFIG}" "${SCALE}")"
fi
if [[ -z "${MIN_SIZE}" ]]; then
    MIN_SIZE="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["min_size"]))' "${DEPLOYMENT_CONFIG}" "${SCALE}")"
fi
if [[ -z "${FLOW3D_SMOOTH}" ]]; then
    FLOW3D_SMOOTH="$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["flow3D_smooth"]), "g"))' "${DEPLOYMENT_CONFIG}" "${SCALE}")"
fi
require_file "${MODEL_PATH}"

if [[ "${CELLPROB_THRESHOLD}" == -* ]]; then cp_tag="neg${CELLPROB_THRESHOLD#-}"; else cp_tag="${CELLPROB_THRESHOLD}"; fi
cp_tag="${cp_tag//./p}"
limited="$(python -c 'import sys; print(int(float(sys.argv[1]) < 1 or int(sys.argv[2]) > 0))' "${PROCESS_FRACTION}" "${MAX_BLOCKS}")"
if [[ -z "${OUTPUT_ZARR:-}" ]]; then
    suffix="vessels_${SCALE}_cp${cp_tag}"
    [[ "${limited}" == 1 ]] && suffix="${suffix}_preview_${SLURM_JOB_ID}"
    OUTPUT_ZARR="$(dirname "${INPUT_ZARR}")/${SAMPLE_ID}_${suffix}.ome.zarr"
fi
if [[ -e "${OUTPUT_ZARR}" && "${ALLOW_OVERWRITE}" != 1 ]]; then
    echo "Output exists: ${OUTPUT_ZARR}; set another OUTPUT_ZARR or ALLOW_OVERWRITE=1" >&2
    exit 1
fi

cat > "${RUN_ROOT}/inference_config.txt" <<EOF
scale=${SCALE}
deployment_config=${DEPLOYMENT_CONFIG}
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
process_fraction=${PROCESS_FRACTION}
max_blocks=${MAX_BLOCKS}
EOF
cp "$0" "${RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${RUN_ROOT}/python_packages.txt"
python -c 'import cellpose,torch,zarr; print("Torch",torch.__version__,"CUDA",torch.cuda.is_available(),"Zarr",zarr.__version__)'
nvidia-smi

echo "Scale=${SCALE} Model=${MODEL_PATH} Threshold=${CELLPROB_THRESHOLD}"
echo "Input=${INPUT_ZARR}/${ARRAY_KEY} Output=${OUTPUT_ZARR}"
cd "${RUN_ROOT}"
srun python -u "${INFER_SCRIPT}" \
    --input "${INPUT_ZARR}" --array_key "${ARRAY_KEY}" --model "${MODEL_PATH}" \
    --output "${OUTPUT_ZARR}" --block_shape "${BLOCK_SHAPE}" --halo "${HALO}" \
    --output_chunks "${OUTPUT_CHUNKS}" --anisotropy "${ANISOTROPY}" \
    --cellprob_threshold "${CELLPROB_THRESHOLD}" --min_size "${MIN_SIZE}" \
    --flow3D_smooth "${FLOW3D_SMOOTH}" --skip_empty --min_core_nonzero_voxels 0 \
    --min_core_nonzero_fraction 0 --process_fraction "${PROCESS_FRACTION}" \
    --max_blocks "${MAX_BLOCKS}" --preview_mode "${PREVIEW_MODE}" \
    --preview_seed "${PREVIEW_SEED}" --pass_channels 2>&1 | tee "${RUN_ROOT}/infer_ome_zarr.log"

# Count semantic foreground blockwise without loading the full output into RAM.
python - "${OUTPUT_ZARR}" "${RUN_ROOT}/foreground_volume_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import zarr

output = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
array = zarr.open_group(str(output), mode="r")["0"]
chunks = tuple(int(value) for value in array.chunks)
foreground = 0
for z0 in range(0, array.shape[0], chunks[0]):
    for y0 in range(0, array.shape[1], chunks[1]):
        for x0 in range(0, array.shape[2], chunks[2]):
            block = np.asarray(array[
                z0:min(array.shape[0], z0 + chunks[0]),
                y0:min(array.shape[1], y0 + chunks[1]),
                x0:min(array.shape[2], x0 + chunks[2]),
            ])
            foreground += int(np.count_nonzero(block))
total = int(np.prod(array.shape, dtype=np.int64))
summary = {
    "output": str(output.resolve()),
    "shape_zyx": [int(value) for value in array.shape],
    "chunks_zyx": list(chunks),
    "foreground_voxels": foreground,
    "total_voxels": total,
    "foreground_fraction": foreground / total if total else 0.0,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "Inference output: ${OUTPUT_ZARR}"
echo "Run metadata: ${RUN_ROOT}"
