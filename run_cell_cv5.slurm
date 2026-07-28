#!/bin/bash
#SBATCH --job-name=cell-cv5
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-cv5-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-cv5-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_cell_only/cv5_seed46"
CV_RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_cv5_seed46_${SLURM_JOB_ID}"
TRAIN_SCRIPT="${REPO_ROOT}/segment_nuc.py"
SWEEP_SCRIPT="${REPO_ROOT}/sweep_thresholds_segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_cell_cv.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
THRESHOLD_SUMMARY="${CV_RUN_ROOT}/selected_thresholds.tsv"

mkdir -p "${CV_RUN_ROOT}"
exec > >(tee -a "${CV_RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${CV_RUN_ROOT}/COMPLETED"
        echo "Workflow completed successfully."
    else
        printf "exit_code=%s\n" "${status}" > "${CV_RUN_ROOT}/FAILED"
        echo "Workflow failed with exit code ${status}." >&2
    fi
    exit "${status}"
}
trap finish EXIT

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/.cache/pip"
mkdir -p "${PROJECT_ROOT}/.cellpose/models"

export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND="Agg"
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
require_file "${SWEEP_SCRIPT}"
require_file "${TEST_SCRIPT}"
require_file "${AGGREGATE_SCRIPT}"
require_file "${VENV_ACTIVATE}"
require_dir "${CV_DATA_ROOT}"

cp "$0" "${CV_RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${CV_RUN_ROOT}/python_packages.txt"
printf "fold\tmodel\tcoarse_cellprob_threshold\tcellprob_threshold\tvalidation_settings\n" > "${THRESHOLD_SUMMARY}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CV data: ${CV_DATA_ROOT}"
echo "CV output: ${CV_RUN_ROOT}"
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
nvidia-smi

for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    fold_data="${CV_DATA_ROOT}/fold_${fold_tag}/combined"
    fold_run="${CV_RUN_ROOT}/fold_${fold_tag}"
    model_name="cell_fold_${fold_tag}_final"
    model_path="${fold_run}/models/best_model"
    coarse_sweep_dir="${fold_run}/threshold_sweep_val_coarse"
    fine_sweep_dir="${fold_run}/threshold_sweep_val_fine"
    inference_dir="${fold_run}/inference_eval"
    coarse_best_json="${coarse_sweep_dir}/best_settings_by_metric.json"
    fine_best_json="${fine_sweep_dir}/best_settings_by_metric.json"
    fold_seed=$((46 + fold))

    require_dir "${fold_data}/Trainingsdaten"
    require_dir "${fold_data}/Valdaten"
    require_dir "${fold_data}/Testdaten"
    mkdir -p "${fold_run}" "${coarse_sweep_dir}" "${fine_sweep_dir}" "${inference_dir}"

    echo
    echo "============================================================"
    echo "Starting cell outer fold ${fold_tag}"
    echo "============================================================"
    date

    cd "${fold_run}"
    srun python -u "${TRAIN_SCRIPT}" \
        --infer_3d \
        --rng_seed "${fold_seed}" \
        --data_dir "${fold_data}/Trainingsdaten" \
        --val_data_dir "${fold_data}/Valdaten" \
        --dataset_dir "${fold_run}/dataset" \
        --model_name "${model_name}" \
        --mask_id 1 \
        --n_epochs 120 \
        --learning_rate 3e-5 \
        --weight_decay 0.05 \
        --batch_size 1 \
        2>&1 | tee "${fold_run}/train.log"

    require_file "${fold_run}/models/${model_name}"
    require_file "${model_path}"

    echo
    echo "Coarse validation threshold sweep: fold ${fold_tag}"
    date
    srun python -u "${SWEEP_SCRIPT}" \
        --data_dir "${fold_data}/Valdaten" \
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
        --data_dir "${fold_data}/Valdaten" \
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

    printf "coarse_cellprob_threshold=%s\ncellprob_threshold=%s\ncoarse_source=%s\nsource=%s\n" \
        "${coarse_best_cp}" \
        "${best_cp}" \
        "${coarse_best_json}" \
        "${fine_best_json}" \
        > "${inference_dir}/selected_validation_settings.txt"
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${fold_tag}" \
        "${model_path}" \
        "${coarse_best_cp}" \
        "${best_cp}" \
        "${fine_best_json}" \
        >> "${THRESHOLD_SUMMARY}"

    echo "Selected fine cellprob_threshold=${best_cp}"
    echo "Outer test inference: fold ${fold_tag}"
    date
    srun python -u "${TEST_SCRIPT}" \
        --model "${model_path}" \
        --test_data_dir "${fold_data}/Testdaten" \
        --outdir "${inference_dir}" \
        --label_extension .seg.nrrd \
        --cellprob_threshold "${best_cp}" \
        --min_size 15 \
        --mask_id 1 \
        --infer_3d \
        2>&1 | tee "${inference_dir}/test.log"

    require_file "${inference_dir}/metrics_per_sample.csv"
    require_file "${inference_dir}/metrics_summary.csv"
    touch "${fold_run}/COMPLETED"
    echo "Completed cell outer fold ${fold_tag}"
done

echo
echo "Aggregating out-of-fold test metrics"
srun python -u "${AGGREGATE_SCRIPT}" \
    --cv_run_dir "${CV_RUN_ROOT}" \
    --n_folds 5 \
    2>&1 | tee "${CV_RUN_ROOT}/aggregation.log"

require_file "${CV_RUN_ROOT}/test_metrics_all.csv"
require_file "${CV_RUN_ROOT}/test_metrics_summary.csv"
require_file "${CV_RUN_ROOT}/cv_report.txt"
require_file "${CV_RUN_ROOT}/test_dice_violin.png"
require_file "${CV_RUN_ROOT}/test_dice_violin.pdf"

echo
echo "Cross-validation outputs:"
echo "  ${CV_RUN_ROOT}/test_metrics_all.csv"
echo "  ${CV_RUN_ROOT}/test_metrics_summary.csv"
echo "  ${CV_RUN_ROOT}/cv_report.txt"
echo "  ${CV_RUN_ROOT}/test_dice_violin.png"
echo "  ${CV_RUN_ROOT}/test_dice_violin.pdf"
cat "${CV_RUN_ROOT}/cv_report.txt"
