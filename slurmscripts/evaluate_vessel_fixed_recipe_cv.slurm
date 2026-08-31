#!/bin/bash
#SBATCH --job-name=vessel-eval-cal
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-eval-cal-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-eval-cal-%j.err

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_vessel_only/cv5_seed46"
CV_RUN_ROOT="${CV_RUN_ROOT:-${PROJECT_ROOT}/runs/run_vessel_cv5_seed46_15083193}"
EVAL_RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_fixed_eval_calibration_${SLURM_JOB_ID}"
REPORT_ROOT="${EVAL_RUN_ROOT}/fixed_reporting"
CALIBRATION_ROOT="${EVAL_RUN_ROOT}/deployment_calibration"
RECIPE_SCRIPT="${REPO_ROOT}/derive_vessel_fixed_recipe.py"
TRAIN_SCRIPT="${REPO_ROOT}/segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
SWEEP_SCRIPT="${REPO_ROOT}/sweep_thresholds_segment_nuc.py"
AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_vessel_cv.py"
CALIBRATION_AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_vessel_calibration.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
FIXED_RECIPE="${EVAL_RUN_ROOT}/fixed_vessel_recipe.json"
REPORT_PREFIX="fixed_recipe_vessel"
CALIBRATION_PREFIX="deployment_calibration_vessel"
CALIBRATION_STAGE="volume_balance"
MIN_SIZE=15
MAX_DICE_DROP=0.01

mkdir -p "${EVAL_RUN_ROOT}" "${REPORT_ROOT}" "${CALIBRATION_ROOT}"
exec > >(tee -a "${EVAL_RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${EVAL_RUN_ROOT}/COMPLETED"
    else
        printf "exit_code=%s\n" "${status}" > "${EVAL_RUN_ROOT}/FAILED"
    fi
    exit "${status}"
}
trap finish EXIT

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }
link_or_copy() { ln "$1" "$2" 2>/dev/null || cp "$1" "$2"; }
recipe_value() {
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["scales"][sys.argv[2]][sys.argv[3]])' \
        "${FIXED_RECIPE}" "$1" "$2"
}

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cache/pip" "${PROJECT_ROOT}/.cellpose/models"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND=Agg
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1

for path in "${RECIPE_SCRIPT}" "${TRAIN_SCRIPT}" "${TEST_SCRIPT}" "${SWEEP_SCRIPT}" \
    "${AGGREGATE_SCRIPT}" "${CALIBRATION_AGGREGATE_SCRIPT}" "${VENV_ACTIVATE}"; do
    require_file "${path}"
done
require_dir "${CV_DATA_ROOT}"
require_dir "${CV_RUN_ROOT}"
cp "$0" "${EVAL_RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${EVAL_RUN_ROOT}/python_packages.txt"
python -u "${RECIPE_SCRIPT}" --cv_run_dir "${CV_RUN_ROOT}" --output "${FIXED_RECIPE}"
cp "${FIXED_RECIPE}" "${CV_RUN_ROOT}/${REPORT_PREFIX}.json"

echo "Source CV: ${CV_RUN_ROOT}"
echo "Output: ${EVAL_RUN_ROOT}"
cat "${FIXED_RECIPE}"
nvidia-smi

###############################################################################
# Phase 1: frozen out-of-fold reporting. No retraining or test-set tuning.
###############################################################################
for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    for scale in unscaled mid coarse; do
        epoch_index="$(recipe_value "${scale}" checkpoint_index)"
        threshold="$(recipe_value "${scale}" reporting_cellprob_threshold)"
        model="${CV_RUN_ROOT}/fold_${fold_tag}/${scale}/models/vessel_${scale}_finetune_final_epoch_$(printf "%04d" "${epoch_index}")"
        test_dir="${CV_DATA_ROOT}/fold_${fold_tag}/${scale}/Testdaten"
        eval_dir="${REPORT_ROOT}/fold_${fold_tag}/${scale}/inference_eval"
        require_file "${model}"
        require_dir "${test_dir}"
        mkdir -p "${eval_dir}"
        echo "Frozen reporting: fold=${fold_tag} scale=${scale} model=${model} cp=${threshold}"
        srun python -u "${TEST_SCRIPT}" \
            --model "${model}" --test_data_dir "${test_dir}" --outdir "${eval_dir}" \
            --label_extension .seg.nrrd --cellprob_threshold "${threshold}" \
            --min_size "${MIN_SIZE}" --mask_id 1 --infer_3d \
            2>&1 | tee "${eval_dir}/test.log"
    done
done

srun python -u "${AGGREGATE_SCRIPT}" --cv_run_dir "${REPORT_ROOT}" --n_folds 5 \
    2>&1 | tee "${REPORT_ROOT}/aggregation.log"
for artifact in test_metrics_all_scales.csv test_metrics_summary_by_scale.csv cv_report.txt \
    test_dice_violin_by_scale.png test_dice_violin_by_scale.pdf; do
    require_file "${REPORT_ROOT}/${artifact}"
    cp "${REPORT_ROOT}/${artifact}" "${CV_RUN_ROOT}/${REPORT_PREFIX}_${artifact}"
done

###############################################################################
# Phase 2: retrain mixed-scale precursor and scale models on train+validation.
# Former outer-test samples are used only to calibrate deployment thresholds.
###############################################################################
for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    fold_data="${CV_DATA_ROOT}/fold_${fold_tag}"
    fold_run="${CALIBRATION_ROOT}/fold_${fold_tag}"
    combined_data="${fold_run}/combined_train_plus_val"
    combined_run="${fold_run}/combined"
    combined_name="vessel_calibration_fold_${fold_tag}_combined_120ep"
    combined_model="${combined_run}/models/${combined_name}"
    mkdir -p "${combined_data}" "${combined_run}"

    declare -A combined_samples=()
    for split in Trainingsdaten Valdaten; do
        for image in "${fold_data}/combined/${split}"/*_img.tif; do
            sample="$(basename "${image}" _img.tif)"
            label="${fold_data}/combined/${split}/${sample}_label.seg.nrrd"
            require_file "${label}"
            [[ ! -v "combined_samples[${sample}]" ]] || { echo "Duplicate combined sample: ${sample}" >&2; exit 1; }
            combined_samples["${sample}"]=1
            link_or_copy "${image}" "${combined_data}/${sample}_img.tif"
            link_or_copy "${label}" "${combined_data}/${sample}_label.seg.nrrd"
        done
    done
    test_images=("${fold_data}/combined/Testdaten"/*_img.tif)
    [[ ${#combined_samples[@]} -eq $((20 - ${#test_images[@]})) ]] || {
        echo "Fold ${fold_tag}: wrong combined train+val count ${#combined_samples[@]}" >&2; exit 1;
    }
    for image in "${test_images[@]}"; do
        sample="$(basename "${image}" _img.tif)"
        [[ ! -v "combined_samples[${sample}]" ]] || { echo "Combined test leakage: ${sample}" >&2; exit 1; }
    done

    cd "${combined_run}"
    srun python -u "${TRAIN_SCRIPT}" --infer_3d --rng_seed "$((46 + fold))" \
        --data_dir "${combined_data}" --val_data_dir "${fold_data}/combined/Valdaten" \
        --dataset_dir "${combined_run}/dataset" --model_name "${combined_name}" \
        --mask_id 1 --n_epochs 120 --learning_rate 3e-5 --weight_decay 0.05 --batch_size 1 \
        2>&1 | tee "${combined_run}/train.log"
    require_file "${combined_model}"

    scale_offset=0
    for scale in unscaled mid coarse; do
        scale_data="${fold_run}/${scale}/train_plus_val"
        scale_run="${fold_run}/${scale}"
        model_name="vessel_calibration_fold_${fold_tag}_${scale}_80ep"
        epoch_index="$(recipe_value "${scale}" checkpoint_index)"
        selected_model="${scale_run}/models/${model_name}_epoch_$(printf "%04d" "${epoch_index}")"
        mkdir -p "${scale_data}"
        declare -A scale_samples=()
        for split in Trainingsdaten Valdaten; do
            for image in "${fold_data}/${scale}/${split}"/*_img.tif; do
                sample="$(basename "${image}" _img.tif)"
                label="${fold_data}/${scale}/${split}/${sample}_label.seg.nrrd"
                require_file "${label}"
                [[ ! -v "scale_samples[${sample}]" ]] || { echo "Duplicate ${scale} sample: ${sample}" >&2; exit 1; }
                scale_samples["${sample}"]=1
                link_or_copy "${image}" "${scale_data}/${sample}_img.tif"
                link_or_copy "${label}" "${scale_data}/${sample}_label.seg.nrrd"
            done
        done
        scale_tests=("${fold_data}/${scale}/Testdaten"/*_img.tif)
        expected_total=5
        [[ "${scale}" == unscaled ]] && expected_total=10
        [[ ${#scale_samples[@]} -eq $((expected_total - 1)) ]] || {
            echo "Fold ${fold_tag} ${scale}: wrong train+val count ${#scale_samples[@]}" >&2; exit 1;
        }
        [[ ${#scale_tests[@]} -eq 1 ]] || { echo "Expected one ${scale} test sample" >&2; exit 1; }
        test_sample="$(basename "${scale_tests[0]}" _img.tif)"
        [[ ! -v "scale_samples[${test_sample}]" ]] || { echo "${scale} test leakage: ${test_sample}" >&2; exit 1; }

        cd "${scale_run}"
        srun python -u "${TRAIN_SCRIPT}" --infer_3d \
            --rng_seed "$((4600 + fold * 10 + scale_offset))" \
            --data_dir "${scale_data}" --val_data_dir "${fold_data}/${scale}/Valdaten" \
            --dataset_dir "${scale_run}/dataset" --model_name "${model_name}" --mask_id 1 \
            --n_epochs 80 --learning_rate 1e-5 --weight_decay 0.05 --batch_size 1 \
            --pretrained_model "${combined_model}" 2>&1 | tee "${scale_run}/train.log"
        require_file "${selected_model}"
        scale_offset=$((scale_offset + 1))
    done
done

IFS=' ' read -r -a thresholds <<< "$(python -c 'print(" ".join(f"{-6+0.25*i:g}" for i in range(33)))')"
for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    for scale in unscaled mid coarse; do
        epoch_index="$(recipe_value "${scale}" checkpoint_index)"
        model="${CALIBRATION_ROOT}/fold_${fold_tag}/${scale}/models/vessel_calibration_fold_${fold_tag}_${scale}_80ep_epoch_$(printf "%04d" "${epoch_index}")"
        sweep_dir="${CALIBRATION_ROOT}/fold_${fold_tag}/${scale}/threshold_sweep_${CALIBRATION_STAGE}"
        mkdir -p "${sweep_dir}"
        srun python -u "${SWEEP_SCRIPT}" \
            --data_dir "${CV_DATA_ROOT}/fold_${fold_tag}/${scale}/Testdaten" \
            --model_path "${model}" --output_dir "${sweep_dir}" --infer_3d \
            --gt_foreground_mode label1 --gt_foreground_label 1 \
            --cellprob_values "${thresholds[@]}" --min_size_values "${MIN_SIZE}" \
            --flow3D_smooth_values 0 2>&1 | tee "${sweep_dir}/sweep.log"
    done
done

srun python -u "${CALIBRATION_AGGREGATE_SCRIPT}" \
    --calibration_root "${CALIBRATION_ROOT}" --stage "${CALIBRATION_STAGE}" \
    --n_folds 5 --max_dice_drop "${MAX_DICE_DROP}" \
    2>&1 | tee "${CALIBRATION_ROOT}/aggregation.log"
for artifact in \
    "pooled_${CALIBRATION_STAGE}_per_sample_all_thresholds.csv" \
    "pooled_${CALIBRATION_STAGE}_threshold_summary_by_scale.csv" \
    "pooled_${CALIBRATION_STAGE}_best_by_scale.json" \
    "pooled_${CALIBRATION_STAGE}_report.txt"; do
    require_file "${CALIBRATION_ROOT}/${artifact}"
    cp "${CALIBRATION_ROOT}/${artifact}" "${CV_RUN_ROOT}/${CALIBRATION_PREFIX}_${artifact#pooled_}"
done

echo "Fixed reporting estimate:"
cat "${CV_RUN_ROOT}/${REPORT_PREFIX}_cv_report.txt"
echo "Deployment calibration:"
cat "${CV_RUN_ROOT}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_report.txt"
echo "Reproducibility output: ${EVAL_RUN_ROOT}"
