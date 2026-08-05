#!/bin/bash
#SBATCH --job-name=cell-eval-cal
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-eval-cal-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-eval-cal-%j.err

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_cell_only/cv5_seed46"
CV_RUN_ROOT="${CV_RUN_ROOT:-${PROJECT_ROOT}/runs/run_cell_cv5_seed46_15083187}"
EVAL_RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_fixed_eval_calibration_${SLURM_JOB_ID}"
REPORT_ROOT="${EVAL_RUN_ROOT}/fixed_reporting"
CALIBRATION_ROOT="${EVAL_RUN_ROOT}/deployment_calibration"
TRAIN_SCRIPT="${REPO_ROOT}/segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
SWEEP_SCRIPT="${REPO_ROOT}/sweep_thresholds_segment_nuc.py"
AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_cell_cv.py"
CALIBRATION_AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_cell_calibration.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

REPORT_PREFIX="fixed_recipe_epoch51_cpneg3p5"
CALIBRATION_PREFIX="deployment_calibration_epoch51"
CALIBRATION_STAGE="volume_balance"
FIXED_CELLPROB="-3.5"
FIXED_MIN_SIZE="15"
SELECTED_EPOCH="51"
TRAINING_EPOCHS="120"
MAX_DICE_DROP="0.01"

mkdir -p "${EVAL_RUN_ROOT}" "${REPORT_ROOT}" "${CALIBRATION_ROOT}"
exec > >(tee -a "${EVAL_RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${EVAL_RUN_ROOT}/COMPLETED"
        echo "Fixed reporting and deployment calibration completed successfully."
    else
        printf "exit_code=%s\n" "${status}" > "${EVAL_RUN_ROOT}/FAILED"
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

link_or_copy() {
    local source_path="$1"
    local destination_path="$2"
    ln "${source_path}" "${destination_path}" 2>/dev/null || \
        cp "${source_path}" "${destination_path}"
}

require_file "${TRAIN_SCRIPT}"
require_file "${TEST_SCRIPT}"
require_file "${SWEEP_SCRIPT}"
require_file "${AGGREGATE_SCRIPT}"
require_file "${CALIBRATION_AGGREGATE_SCRIPT}"
require_file "${VENV_ACTIVATE}"
require_dir "${CV_DATA_ROOT}"
require_dir "${CV_RUN_ROOT}"

cp "$0" "${EVAL_RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${EVAL_RUN_ROOT}/python_packages.txt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Source CV run: ${CV_RUN_ROOT}"
echo "Output run: ${EVAL_RUN_ROOT}"
echo "Reporting recipe: epoch=${SELECTED_EPOCH}, cellprob=${FIXED_CELLPROB}, min_size=${FIXED_MIN_SIZE}"
echo "Calibration training: ${TRAINING_EPOCHS} total epochs, checkpoint epoch ${SELECTED_EPOCH}"
echo "Optimizer: learning_rate=3e-5, weight_decay=0.05, batch_size=1"
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
nvidia-smi

###############################################################################
# Phase 1: frozen, unbiased reporting estimate. No fitting or threshold tuning.
###############################################################################

echo
echo "PHASE 1: frozen out-of-fold reporting estimate"
for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    model_path="${CV_RUN_ROOT}/fold_${fold_tag}/models/cell_fold_${fold_tag}_final_epoch_0050"
    test_dir="${CV_DATA_ROOT}/fold_${fold_tag}/combined/Testdaten"
    eval_dir="${REPORT_ROOT}/fold_${fold_tag}/inference_eval"

    require_file "${model_path}"
    require_dir "${test_dir}"
    mkdir -p "${eval_dir}"

    echo
    echo "Frozen reporting inference: fold ${fold_tag}"
    date
    srun python -u "${TEST_SCRIPT}" \
        --model "${model_path}" \
        --test_data_dir "${test_dir}" \
        --outdir "${eval_dir}" \
        --label_extension .seg.nrrd \
        --cellprob_threshold "${FIXED_CELLPROB}" \
        --min_size "${FIXED_MIN_SIZE}" \
        --mask_id 1 \
        --infer_3d \
        2>&1 | tee "${eval_dir}/test.log"

    require_file "${eval_dir}/metrics_per_sample.csv"
    require_file "${eval_dir}/metrics_summary.csv"
    require_file "${eval_dir}/run_meta.json"
done

srun python -u "${AGGREGATE_SCRIPT}" \
    --cv_run_dir "${REPORT_ROOT}" \
    --n_folds 5 \
    2>&1 | tee "${REPORT_ROOT}/aggregation.log"

for artifact in \
    test_metrics_all.csv \
    test_metrics_summary.csv \
    cv_report.txt \
    test_dice_violin.png \
    test_dice_violin.pdf; do
    require_file "${REPORT_ROOT}/${artifact}"
    cp "${REPORT_ROOT}/${artifact}" "${CV_RUN_ROOT}/${REPORT_PREFIX}_${artifact}"
done

###############################################################################
# Phase 2: cross-fitted calibration. Former test folds are calibration-only.
###############################################################################

echo
echo "PHASE 2: train+validation cross-fitted deployment calibration"
for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    fold_seed=$((46 + fold))
    fold_data="${CV_DATA_ROOT}/fold_${fold_tag}/combined"
    fold_run="${CALIBRATION_ROOT}/fold_${fold_tag}"
    train_plus_val="${fold_run}/train_plus_val"
    model_name="cell_calibration_fold_${fold_tag}_120ep"
    selected_model="${fold_run}/models/${model_name}_epoch_0050"

    require_dir "${fold_data}/Trainingsdaten"
    require_dir "${fold_data}/Valdaten"
    require_dir "${fold_data}/Testdaten"
    mkdir -p "${train_plus_val}"

    declare -A fold_samples=()
    for split in Trainingsdaten Valdaten; do
        for image_path in "${fold_data}/${split}"/*_img.tif; do
            sample_id="$(basename "${image_path}" _img.tif)"
            label_path="${fold_data}/${split}/${sample_id}_label.seg.nrrd"
            require_file "${label_path}"
            if [[ -v "fold_samples[${sample_id}]" ]]; then
                echo "Fold ${fold_tag}: duplicate train+validation sample ${sample_id}" >&2
                exit 1
            fi
            fold_samples["${sample_id}"]=1
            link_or_copy "${image_path}" "${train_plus_val}/${sample_id}_img.tif"
            link_or_copy "${label_path}" "${train_plus_val}/${sample_id}_label.seg.nrrd"
        done
    done

    test_images=("${fold_data}/Testdaten"/*_img.tif)
    expected_train_count=$((12 - ${#test_images[@]}))
    if [[ ${#fold_samples[@]} -ne ${expected_train_count} ]]; then
        echo "Fold ${fold_tag}: expected ${expected_train_count} train+validation samples, got ${#fold_samples[@]}" >&2
        exit 1
    fi
    for test_image in "${test_images[@]}"; do
        test_sample_id="$(basename "${test_image}" _img.tif)"
        if [[ -v "fold_samples[${test_sample_id}]" ]]; then
            echo "Fold ${fold_tag}: held-out calibration sample leaked into training: ${test_sample_id}" >&2
            exit 1
        fi
    done

    echo
    echo "Training calibration fold ${fold_tag}: ${#fold_samples[@]} samples"
    echo "Former test fold remains excluded from gradient training"
    date
    cd "${fold_run}"
    srun python -u "${TRAIN_SCRIPT}" \
        --infer_3d \
        --rng_seed "${fold_seed}" \
        --data_dir "${train_plus_val}" \
        --val_data_dir "${fold_data}/Valdaten" \
        --dataset_dir "${fold_run}/dataset" \
        --model_name "${model_name}" \
        --mask_id 1 \
        --n_epochs "${TRAINING_EPOCHS}" \
        --learning_rate 3e-5 \
        --weight_decay 0.05 \
        --batch_size 1 \
        2>&1 | tee "${fold_run}/train.log"

    # Valdaten is now part of gradient training and is reused only for Cellpose's
    # required diagnostic test-loss input. Ignore best_model: epoch 51 is fixed.
    require_file "${selected_model}"

done

echo
echo "Full shared threshold scan for Dice-constrained volume calibration"
IFS=' ' read -r -a calibration_values <<< "$(
    python -c 'print(" ".join(f"{-6.0+0.25*i:g}" for i in range(33)))'
)"
echo "Thresholds: ${calibration_values[*]}"

for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    fold_data="${CV_DATA_ROOT}/fold_${fold_tag}/combined"
    fold_run="${CALIBRATION_ROOT}/fold_${fold_tag}"
    selected_model="${fold_run}/models/cell_calibration_fold_${fold_tag}_120ep_epoch_0050"
    calibration_sweep_dir="${fold_run}/threshold_sweep_${CALIBRATION_STAGE}"

    require_file "${selected_model}"
    mkdir -p "${calibration_sweep_dir}"
    echo
    echo "Full held-out volume-calibration sweep: fold ${fold_tag}"
    date
    srun python -u "${SWEEP_SCRIPT}" \
        --data_dir "${fold_data}/Testdaten" \
        --model_path "${selected_model}" \
        --output_dir "${calibration_sweep_dir}" \
        --infer_3d \
        --gt_foreground_mode label1 \
        --gt_foreground_label 1 \
        --cellprob_values "${calibration_values[@]}" \
        --min_size_values "${FIXED_MIN_SIZE}" \
        --flow3D_smooth_values 0 \
        2>&1 | tee "${calibration_sweep_dir}/sweep.log"
done

echo
echo "Selecting a volume-balanced threshold within ${MAX_DICE_DROP} Dice of the optimum"
srun python -u "${CALIBRATION_AGGREGATE_SCRIPT}" \
    --calibration_root "${CALIBRATION_ROOT}" \
    --stage "${CALIBRATION_STAGE}" \
    --n_folds 5 \
    --selection_mode volume_balanced \
    --max_dice_drop "${MAX_DICE_DROP}" \
    2>&1 | tee "${CALIBRATION_ROOT}/pooled_${CALIBRATION_STAGE}_aggregation.log"

for artifact in \
    "pooled_${CALIBRATION_STAGE}_per_sample_all_thresholds.csv" \
    "pooled_${CALIBRATION_STAGE}_threshold_summary.csv" \
    "pooled_${CALIBRATION_STAGE}_best.json" \
    "pooled_${CALIBRATION_STAGE}_report.txt"; do
    require_file "${CALIBRATION_ROOT}/${artifact}"
    cp "${CALIBRATION_ROOT}/${artifact}" \
        "${CV_RUN_ROOT}/${CALIBRATION_PREFIX}_${artifact#pooled_}"
done

echo
echo "Frozen reporting estimate:"
cat "${CV_RUN_ROOT}/${REPORT_PREFIX}_cv_report.txt"
echo
echo "Final deployment calibration:"
cat "${CV_RUN_ROOT}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_report.txt"
echo "Full reproducibility run: ${EVAL_RUN_ROOT}"
