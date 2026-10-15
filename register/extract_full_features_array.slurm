#!/bin/bash
#SBATCH --job-name=full-feature-extract
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=full-feature-extract-%A_%a.out
#SBATCH --error=full-feature-extract-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
INPUT_ROOT="${INPUT_ROOT:-${REGISTER_ROOT}/register_full_input}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is not set}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/extract_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -u "${REGISTER_ROOT}/extract_full_mri_segmentation_features.py" \
  --input-root "${INPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/features" \
  --array-key 0 \
  --voxel-size-um 0.65 \
  --scan-chunks 32,256,256 \
  --task-index "${SLURM_ARRAY_TASK_ID}"
