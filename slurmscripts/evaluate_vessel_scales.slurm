#!/bin/bash
#SBATCH --job-name=vessel-scale-eval
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-scale-eval-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-scale-eval-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SWEEP_SCRIPT="${REPO_ROOT}/sweep_thresholds_segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
SPLIT_ROOT="${PROJECT_ROOT}/data_vessel_only/split_seed46"
SUMMARY_FILE="${PROJECT_ROOT}/runs/vessel-scale-selected-settings-${SLURM_JOB_ID}.tsv"

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

require_file "${SWEEP_SCRIPT}"
require_file "${TEST_SCRIPT}"
require_file "${VENV_ACTIVATE}"

printf "scale\tmodel\tcoarse_cellprob_threshold\tcellprob_threshold\tvalidation_settings\n" > "${SUMMARY_FILE}"

evaluate_scale() {
    local scale="$1"
    local run_name="$2"
    local model_name="$3"
    local run_dir="${PROJECT_ROOT}/runs/${run_name}"
    local model_path="${run_dir}/models/${model_name}"
    local val_dir="${SPLIT_ROOT}/${scale}/Valdaten"
    local test_dir="${SPLIT_ROOT}/${scale}/Testdaten"
    local coarse_sweep_dir="${run_dir}/threshold_sweep_val_coarse"
    local fine_sweep_dir="${run_dir}/threshold_sweep_val_fine"
    local test_out="${run_dir}/inference_eval"
    local coarse_best_json="${coarse_sweep_dir}/best_settings_by_metric.json"
    local fine_best_json="${fine_sweep_dir}/best_settings_by_metric.json"
    local coarse_best_cp
    local -a fine_values
    local best_cp

    require_file "${model_path}"
    require_dir "${val_dir}"
    require_dir "${test_dir}"
    mkdir -p "${coarse_sweep_dir}"
    mkdir -p "${fine_sweep_dir}"
    mkdir -p "${test_out}"

    echo
    echo "Starting ${scale} coarse validation sweep"
    echo "Model: ${model_path}"
    date

    srun python -u "${SWEEP_SCRIPT}" \
        --data_dir "${val_dir}" \
        --model_path "${model_path}" \
        --output_dir "${coarse_sweep_dir}" \
        --infer_3d \
        --gt_foreground_mode label1 \
        --gt_foreground_label 1 \
        --cellprob_values -6 -4 -2 0 2 \
        --min_size_values 15 \
        --flow3D_smooth_values 0 \
        2>&1 | tee "${coarse_sweep_dir}/sweep.log"

    require_file "${coarse_best_json}"
    coarse_best_cp="$(
        python -c 'import json, sys; row=json.load(open(sys.argv[1]))["best_by_core_sort_key"]; print(row["cellprob_threshold"])' \
            "${coarse_best_json}"
    )"
    IFS=' ' read -r -a fine_values <<< "$(
        python -c 'import sys; c=float(sys.argv[1]); lo=max(-6.0,c-1.0); hi=min(2.0,c+1.0); n=round((hi-lo)/0.5); print(" ".join(f"{lo+0.5*i:g}" for i in range(n+1)))' \
            "${coarse_best_cp}"
    )"

    echo "Coarse winner=${coarse_best_cp}"
    echo "Fine validation thresholds: ${fine_values[*]}"
    date
    srun python -u "${SWEEP_SCRIPT}" \
        --data_dir "${val_dir}" \
        --model_path "${model_path}" \
        --output_dir "${fine_sweep_dir}" \
        --infer_3d \
        --gt_foreground_mode label1 \
        --gt_foreground_label 1 \
        --cellprob_values "${fine_values[@]}" \
        --min_size_values 15 \
        --flow3D_smooth_values 0 \
        2>&1 | tee "${fine_sweep_dir}/sweep.log"

    require_file "${fine_best_json}"
    best_cp="$(
        python -c 'import json, sys; row=json.load(open(sys.argv[1]))["best_by_core_sort_key"]; print(row["cellprob_threshold"])' \
            "${fine_best_json}"
    )"

    echo "Selected from validation:"
    echo "  coarse_cellprob_threshold=${coarse_best_cp}"
    echo "  cellprob_threshold=${best_cp}"

    printf "coarse_cellprob_threshold=%s\ncellprob_threshold=%s\ncoarse_source=%s\nsource=%s\n" \
        "${coarse_best_cp}" \
        "${best_cp}" \
        "${coarse_best_json}" \
        "${fine_best_json}" \
        > "${test_out}/selected_validation_settings.txt"

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${scale}" \
        "${model_path}" \
        "${coarse_best_cp}" \
        "${best_cp}" \
        "${fine_best_json}" \
        >> "${SUMMARY_FILE}"

    echo
    echo "Starting ${scale} test inference with frozen validation settings"
    date

    srun python -u "${TEST_SCRIPT}" \
        --model "${model_path}" \
        --test_data_dir "${test_dir}" \
        --outdir "${test_out}" \
        --label_extension .seg.nrrd \
        --cellprob_threshold "${best_cp}" \
        --mask_id 1 \
        --infer_3d \
        2>&1 | tee "${test_out}/test.log"

    echo "Finished ${scale} evaluation: ${test_out}"
    date
}

echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

evaluate_scale "unscaled" "run_vessel_unscaled_ft" "vessel_unscaled_finetune"
evaluate_scale "mid" "run_vessel_mid_ft" "vessel_mid_finetune"
evaluate_scale "coarse" "run_vessel_coarse_ft" "vessel_coarse_finetune"

echo
echo "All validation sweeps and test-set inference runs completed."
echo "Selected settings summary: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"
