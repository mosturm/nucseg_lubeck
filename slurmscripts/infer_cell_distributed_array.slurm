#!/bin/bash
#SBATCH --job-name=cell-dist-gpu
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-gpu-%A_%a.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-dist-gpu-%A_%a.err

set -euo pipefail
: "${RUN_ROOT:?Missing RUN_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
SCRIPT="${PROJECT_ROOT}/repo/nucseg_lubeck/cell_distributed_ome_zarr.py"
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

worker_tag="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
exec > >(tee -a "${RUN_ROOT}/workers/worker_${worker_tag}.log") 2>&1
test -f "${RUN_ROOT}/PREPARED"
nvidia-smi

python -u "${SCRIPT}" infer-worker \
  --run-root "${RUN_ROOT}" \
  --worker-index "${SLURM_ARRAY_TASK_ID}"

test -f "${RUN_ROOT}/workers/worker_${worker_tag}.DONE"
