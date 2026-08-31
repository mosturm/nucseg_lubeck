#!/bin/bash
#SBATCH --job-name=vessel-ms-gpu
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-gpu-%A_%a.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-ms-gpu-%A_%a.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${DEPLOYMENT_CONFIG:?Missing DEPLOYMENT_CONFIG}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SCRIPT="${REPO_ROOT}/vessel_multiscale_ome_zarr.py"
VENV="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV}"
mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cellpose/models" "${RUN_ROOT}/workers"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND=Agg
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

exec > >(tee -a "${RUN_ROOT}/workers/worker_$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").log") 2>&1
test -f "${RUN_ROOT}/PREPARED"
nvidia-smi

python -u "${SCRIPT}" infer-worker \
  --run-root "${RUN_ROOT}" \
  --deployment-config "${DEPLOYMENT_CONFIG}" \
  --worker-index "${SLURM_ARRAY_TASK_ID}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --block-shape "${BLOCK_SHAPE:-40x256x256}" \
  --halo "${HALO:-5x32x32}" \
  --anisotropy "${ANISOTROPY:-1.0}"

test -f "${RUN_ROOT}/workers/worker_$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").DONE"
