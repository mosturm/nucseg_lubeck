#!/bin/bash
#SBATCH --job-name=vessel-scale-ft
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-scale-ft-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-scale-ft-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
TRAIN_SCRIPT="${PROJECT_ROOT}/repo/nucseg_lubeck/segment_nuc.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
BASE_MODEL="${PROJECT_ROOT}/runs/run_vessel_seg2/models/best_model"
SPLIT_ROOT="${PROJECT_ROOT}/data_vessel_only/split_seed46"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/.cache/pip"
mkdir -p "${PROJECT_ROOT}/.cellpose/models"

export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1

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

require_file "${TRAIN_SCRIPT}"
require_file "${VENV_ACTIVATE}"
require_file "${BASE_MODEL}"

run_scale() {
    local scale="$1"
    local run_name="$2"
    local model_name="$3"
    local train_dir="${SPLIT_ROOT}/${scale}/Trainingsdaten"
    local val_dir="${SPLIT_ROOT}/${scale}/Valdaten"
    local run_dir="${PROJECT_ROOT}/runs/${run_name}"

    require_dir "${train_dir}"
    require_dir "${val_dir}"
    mkdir -p "${run_dir}"
    cd "${run_dir}"

    echo
    echo "Starting ${scale} fine-tuning"
    echo "Run directory: ${run_dir}"
    echo "Starting checkpoint: ${BASE_MODEL}"
    date

    srun python -u "${TRAIN_SCRIPT}" \
        --infer_3d \
        --rng_seed 46 \
        --data_dir "${train_dir}" \
        --val_data_dir "${val_dir}" \
        --pretrained_model "${BASE_MODEL}" \
        --dataset_dir "${run_dir}/dataset" \
        --model_name "${model_name}" \
        --mask_id 1 \
        --n_epochs 40 \
        --learning_rate 1e-5 \
        --weight_decay 0.05 \
        --batch_size 1 \
        2>&1 | tee "${run_dir}/run.log"

    require_file "${run_dir}/models/best_model"
    echo "Finished ${scale}: ${run_dir}/models/best_model"
    date
}

echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

run_scale "unscaled" "run_vessel_unscaled_ft" "vessel_unscaled_finetune"
run_scale "mid" "run_vessel_mid_ft" "vessel_mid_finetune"
run_scale "coarse" "run_vessel_coarse_ft" "vessel_coarse_finetune"

echo
echo "All scale-specific fine-tuning runs completed."
ls -lh \
    "${PROJECT_ROOT}/runs/run_vessel_unscaled_ft/models/best_model" \
    "${PROJECT_ROOT}/runs/run_vessel_mid_ft/models/best_model" \
    "${PROJECT_ROOT}/runs/run_vessel_coarse_ft/models/best_model"
