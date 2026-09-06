#!/bin/bash
#SBATCH --job-name=cell-dist-prepare
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=2h
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-prepare-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-prepare-%j.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${INPUT_ZARR:?Missing INPUT_ZARR}"
: "${MODEL_PATH:?Missing MODEL_PATH}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
SCRIPT="${PROJECT_ROOT}/repo/nucseg_lubeck/cell_distributed_ome_zarr.py"
VENV="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/prepare.log") 2>&1
cp "$0" "${RUN_ROOT}/submitted_prepare.slurm"
python -m pip freeze > "${RUN_ROOT}/python_packages.txt"

python -u "${SCRIPT}" prepare \
  --input "${INPUT_ZARR}" \
  --array-key "${ARRAY_KEY:-0}" \
  --model "${MODEL_PATH}" \
  --run-root "${RUN_ROOT}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --block-shape "${BLOCK_SHAPE:-40x256x256}" \
  --halo "${HALO:-5x32x32}" \
  --output-chunks "${OUTPUT_CHUNKS:-8x256x256}" \
  --compressor-level "${COMPRESSOR_LEVEL:-5}" \
  --anisotropy "${ANISOTROPY:-1.0}" \
  --cellprob-threshold "${CELLPROB_THRESHOLD:--2.25}" \
  --min-size "${MIN_SIZE:-15}" \
  --flow3d-smooth "${FLOW3D_SMOOTH:-0}" \
  --skip-empty \
  --min-core-nonzero-voxels 0 \
  --min-core-nonzero-fraction 0 \
  --process-fraction "${PROCESS_FRACTION:-1.0}" \
  --max-blocks "${MAX_BLOCKS:-0}" \
  --preview-mode "${PREVIEW_MODE:-random}" \
  --preview-seed "${PREVIEW_SEED:-46}" \
  --pass-channels

test -f "${RUN_ROOT}/PREPARED"
