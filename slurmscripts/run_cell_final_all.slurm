#!/bin/bash
#SBATCH --job-name=cell-final-all
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-final-all-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-final-all-%j.err

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_cell_only/cv5_seed46"
CV_SOURCE_RUN="${CV_SOURCE_RUN:-${PROJECT_ROOT}/runs/run_cell_cv5_seed46_15083187}"
FINAL_RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_final_all_seed46_${SLURM_JOB_ID}"
ALL_DATA_DIR="${FINAL_RUN_ROOT}/all_accepted_data"
TRAIN_SCRIPT="${REPO_ROOT}/segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
MODEL_NAME="cell_final_all_120ep"
FINAL_TRAINING_MODEL="${FINAL_RUN_ROOT}/models/${MODEL_NAME}"
MODEL_PATH="${FINAL_RUN_ROOT}/models/${MODEL_NAME}_epoch_0050"
QC_DIR="${FINAL_RUN_ROOT}/training_data_qc_not_validation"
FIXED_RECIPE_PREFIX="fixed_recipe_epoch51_cpneg3p5"
CALIBRATION_PREFIX="deployment_calibration_epoch51"
CALIBRATION_STAGE="volume_balance"
CALIBRATION_BEST="${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_best.json"
FIXED_RECIPE_SUMMARY="${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_metrics_summary.csv"

mkdir -p "${FINAL_RUN_ROOT}"
exec > >(tee -a "${FINAL_RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${FINAL_RUN_ROOT}/COMPLETED"
        echo "Final cell-model workflow completed successfully."
    else
        printf "exit_code=%s\n" "${status}" > "${FINAL_RUN_ROOT}/FAILED"
        echo "Final cell-model workflow failed with exit code ${status}." >&2
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
require_file "${TEST_SCRIPT}"
require_file "${VENV_ACTIVATE}"
require_file "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_metrics_all.csv"
require_file "${FIXED_RECIPE_SUMMARY}"
require_file "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_cv_report.txt"
require_file "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_dice_violin.png"
require_file "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_dice_violin.pdf"
require_file "${CALIBRATION_BEST}"
require_file "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_per_sample_all_thresholds.csv"
require_file "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_threshold_summary.csv"
require_file "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_report.txt"
require_dir "${CV_DATA_ROOT}/fold_01/combined"

cp "$0" "${FINAL_RUN_ROOT}/submitted_job.slurm"
cp "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_metrics_all.csv" \
    "${FINAL_RUN_ROOT}/fixed_recipe_cv_test_metrics_all.csv"
cp "${FIXED_RECIPE_SUMMARY}" \
    "${FINAL_RUN_ROOT}/fixed_recipe_cv_test_metrics_summary.csv"
cp "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_cv_report.txt" \
    "${FINAL_RUN_ROOT}/fixed_recipe_cv_report.txt"
cp "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_dice_violin.png" \
    "${FINAL_RUN_ROOT}/fixed_recipe_cv_test_dice_violin.png"
cp "${CV_SOURCE_RUN}/${FIXED_RECIPE_PREFIX}_test_dice_violin.pdf" \
    "${FINAL_RUN_ROOT}/fixed_recipe_cv_test_dice_violin.pdf"
cp "${CALIBRATION_BEST}" "${FINAL_RUN_ROOT}/deployment_calibration_best.json"
cp "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_per_sample_all_thresholds.csv" \
    "${FINAL_RUN_ROOT}/deployment_calibration_per_sample_all_thresholds.csv"
cp "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_threshold_summary.csv" \
    "${FINAL_RUN_ROOT}/deployment_calibration_threshold_summary.csv"
cp "${CV_SOURCE_RUN}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_report.txt" \
    "${FINAL_RUN_ROOT}/deployment_calibration_report.txt"
python -m pip freeze > "${FINAL_RUN_ROOT}/python_packages.txt"

# Fold 01 contains all 12 accepted samples exactly once across its three splits.
# Reassemble those splits without using the excluded source dataset root.
mkdir -p "${ALL_DATA_DIR}"
declare -A seen_samples=()
for split in Trainingsdaten Valdaten Testdaten; do
    split_dir="${CV_DATA_ROOT}/fold_01/combined/${split}"
    require_dir "${split_dir}"
    for image_path in "${split_dir}"/*_img.tif; do
        sample_id="$(basename "${image_path}" _img.tif)"
        label_path="${split_dir}/${sample_id}_label.seg.nrrd"
        require_file "${label_path}"
        if [[ -v "seen_samples[${sample_id}]" ]]; then
            echo "Duplicate sample while assembling final data: ${sample_id}" >&2
            exit 1
        fi
        seen_samples["${sample_id}"]=1
        ln "${image_path}" "${ALL_DATA_DIR}/${sample_id}_img.tif" 2>/dev/null || \
            cp "${image_path}" "${ALL_DATA_DIR}/${sample_id}_img.tif"
        ln "${label_path}" "${ALL_DATA_DIR}/${sample_id}_label.seg.nrrd" 2>/dev/null || \
            cp "${label_path}" "${ALL_DATA_DIR}/${sample_id}_label.seg.nrrd"
    done
done

if [[ ${#seen_samples[@]} -ne 12 ]]; then
    echo "Expected 12 accepted samples, assembled ${#seen_samples[@]}." >&2
    exit 1
fi
if [[ -v "seen_samples[tomo_reco_id0004_t0007vn]" ]]; then
    echo "Excluded sample unexpectedly entered final training data." >&2
    exit 1
fi

DEPLOYMENT_THRESHOLD="$(
    python -c 'import json, sys; value=float(json.load(open(sys.argv[1]))["best"]["cellprob_threshold"]); print(f"{value:g}")' \
        "${CALIBRATION_BEST}"
)"

cat > "${FINAL_RUN_ROOT}/deployment_config.txt" <<EOF
model=${MODEL_PATH}
cellprob_threshold=${DEPLOYMENT_THRESHOLD}
min_size=15
infer_3d=1
mask_id=1
training_samples=12
training_epochs=120
selected_checkpoint_epoch=51
rng_seed=46
learning_rate=3e-5
weight_decay=0.05
batch_size=1
source_cv_run=${CV_SOURCE_RUN}
performance_estimate=${FIXED_RECIPE_SUMMARY}
EOF

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Final run: ${FINAL_RUN_ROOT}"
echo "Accepted training samples: ${#seen_samples[@]}"
echo "Frozen deployment threshold: ${DEPLOYMENT_THRESHOLD}"
python --version
python -c 'import torch; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
nvidia-smi

cd "${FINAL_RUN_ROOT}"
echo
echo "Training final deployment model for 120 epochs on all accepted labeled volumes"
echo "The deployed checkpoint is fixed at epoch 51 (filename index 0050)."
date
srun python -u "${TRAIN_SCRIPT}" \
    --infer_3d \
    --rng_seed 46 \
    --data_dir "${ALL_DATA_DIR}" \
    --val_data_dir "${ALL_DATA_DIR}" \
    --dataset_dir "${FINAL_RUN_ROOT}/dataset" \
    --model_name "${MODEL_NAME}" \
    --mask_id 1 \
    --n_epochs 120 \
    --learning_rate 3e-5 \
    --weight_decay 0.05 \
    --batch_size 1 \
    2>&1 | tee "${FINAL_RUN_ROOT}/train.log"

# Run all 120 epochs to reproduce the original training schedule, but deploy the
# preselected epoch-51 checkpoint. best_model and the epoch-120 model are not used.
require_file "${FINAL_TRAINING_MODEL}"
require_file "${MODEL_PATH}"

echo
echo "Running in-sample training-data QC (not an unbiased validation result)"
date
mkdir -p "${QC_DIR}"
srun python -u "${TEST_SCRIPT}" \
    --model "${MODEL_PATH}" \
    --test_data_dir "${ALL_DATA_DIR}" \
    --outdir "${QC_DIR}" \
    --label_extension .seg.nrrd \
    --cellprob_threshold "${DEPLOYMENT_THRESHOLD}" \
    --min_size 15 \
    --mask_id 1 \
    --infer_3d \
    2>&1 | tee "${QC_DIR}/qc.log"

require_file "${QC_DIR}/metrics_per_sample.csv"
require_file "${QC_DIR}/metrics_summary.csv"

echo
echo "Deployment model: ${MODEL_PATH}"
echo "Deployment threshold: ${DEPLOYMENT_THRESHOLD}"
echo "Fixed-recipe CV performance estimate: ${FINAL_RUN_ROOT}/fixed_recipe_cv_test_metrics_summary.csv"
echo "Training-data QC only: ${QC_DIR}/metrics_summary.csv"
cat "${FINAL_RUN_ROOT}/fixed_recipe_cv_report.txt"
