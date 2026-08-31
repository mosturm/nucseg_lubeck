#!/bin/bash
#SBATCH --job-name=vessel-final-all
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-final-all-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-final-all-%j.err

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_vessel_only/cv5_seed46"
CV_SOURCE_RUN="${CV_SOURCE_RUN:-${PROJECT_ROOT}/runs/run_vessel_cv5_seed46_15083193}"
FINAL_RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_final_all_seed46_${SLURM_JOB_ID}"
TRAIN_SCRIPT="${REPO_ROOT}/segment_nuc.py"
TEST_SCRIPT="${REPO_ROOT}/test_infer.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
FIXED_RECIPE="${CV_SOURCE_RUN}/fixed_recipe_vessel.json"
CALIBRATION="${CV_SOURCE_RUN}/deployment_calibration_vessel_volume_balance_best_by_scale.json"
COMBINED_DATA="${FINAL_RUN_ROOT}/all_prepared/combined"
COMBINED_NAME="vessel_combined_final_all_120ep"
COMBINED_MODEL="${FINAL_RUN_ROOT}/combined/models/${COMBINED_NAME}"

mkdir -p "${FINAL_RUN_ROOT}"
exec > >(tee -a "${FINAL_RUN_ROOT}/workflow.log") 2>&1
finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then touch "${FINAL_RUN_ROOT}/COMPLETED"; else printf "exit_code=%s\n" "${status}" > "${FINAL_RUN_ROOT}/FAILED"; fi
    exit "${status}"
}
trap finish EXIT

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }
link_or_copy() { ln "$1" "$2" 2>/dev/null || cp "$1" "$2"; }

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cache/pip" "${PROJECT_ROOT}/.cellpose/models"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND=Agg
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export CELLPOSE_LOCAL_MODELS_PATH="${PROJECT_ROOT}/.cellpose/models"
export PYTHONUNBUFFERED=1

for path in "${TRAIN_SCRIPT}" "${TEST_SCRIPT}" "${VENV_ACTIVATE}" "${FIXED_RECIPE}" "${CALIBRATION}"; do require_file "${path}"; done
require_dir "${CV_DATA_ROOT}/fold_01"
cp "$0" "${FINAL_RUN_ROOT}/submitted_job.slurm"
cp "${FIXED_RECIPE}" "${FINAL_RUN_ROOT}/fixed_recipe_vessel.json"
cp "${CALIBRATION}" "${FINAL_RUN_ROOT}/deployment_calibration_best_by_scale.json"
for artifact in fixed_recipe_vessel_test_metrics_all_scales.csv \
    fixed_recipe_vessel_test_metrics_summary_by_scale.csv fixed_recipe_vessel_cv_report.txt \
    deployment_calibration_vessel_volume_balance_per_sample_all_thresholds.csv \
    deployment_calibration_vessel_volume_balance_threshold_summary_by_scale.csv \
    deployment_calibration_vessel_volume_balance_report.txt; do
    require_file "${CV_SOURCE_RUN}/${artifact}"
    cp "${CV_SOURCE_RUN}/${artifact}" "${FINAL_RUN_ROOT}/${artifact}"
done
python -m pip freeze > "${FINAL_RUN_ROOT}/python_packages.txt"

# Fold 01 contains every accepted sample exactly once across its three splits.
for scope in combined unscaled mid coarse; do
    target="${FINAL_RUN_ROOT}/all_prepared/${scope}"
    mkdir -p "${target}"
    declare -A seen=()
    for split in Trainingsdaten Valdaten Testdaten; do
        source_dir="${CV_DATA_ROOT}/fold_01/${scope}/${split}"
        require_dir "${source_dir}"
        for image in "${source_dir}"/*_img.tif; do
            sample="$(basename "${image}" _img.tif)"
            label="${source_dir}/${sample}_label.seg.nrrd"
            require_file "${label}"
            [[ ! -v "seen[${sample}]" ]] || { echo "Duplicate ${scope} sample: ${sample}" >&2; exit 1; }
            seen["${sample}"]=1
            link_or_copy "${image}" "${target}/${sample}_img.tif"
            link_or_copy "${label}" "${target}/${sample}_label.seg.nrrd"
        done
    done
    expected=5
    [[ "${scope}" == combined ]] && expected=20
    [[ "${scope}" == unscaled ]] && expected=10
    [[ ${#seen[@]} -eq ${expected} ]] || { echo "Expected ${expected} ${scope} samples, found ${#seen[@]}" >&2; exit 1; }
    [[ ! -v "seen[tomo_reco_id0004_t0007vn]" ]] || { echo "Excluded sample entered ${scope}" >&2; exit 1; }
done

echo "Training mixed-scale precursor on all 20 accepted vessel volumes"
mkdir -p "${FINAL_RUN_ROOT}/combined"
cd "${FINAL_RUN_ROOT}/combined"
srun python -u "${TRAIN_SCRIPT}" --infer_3d --rng_seed 46 \
    --data_dir "${COMBINED_DATA}" --val_data_dir "${COMBINED_DATA}" \
    --dataset_dir "${FINAL_RUN_ROOT}/combined/dataset" --model_name "${COMBINED_NAME}" \
    --mask_id 1 --n_epochs 120 --learning_rate 3e-5 --weight_decay 0.05 --batch_size 1 \
    2>&1 | tee "${FINAL_RUN_ROOT}/combined/train.log"
require_file "${COMBINED_MODEL}"

scales=(unscaled mid coarse)
for scale_index in 0 1 2; do
    scale="${scales[${scale_index}]}"
    scale_run="${FINAL_RUN_ROOT}/${scale}"
    model_name="vessel_${scale}_final_all_80ep"
    epoch_index="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["checkpoint_index"])' "${FIXED_RECIPE}" "${scale}")"
    selected_epoch="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["selected_completed_epoch"])' "${FIXED_RECIPE}" "${scale}")"
    threshold="$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["scales"][sys.argv[2]]["best"]["cellprob_threshold"]), "g"))' "${CALIBRATION}" "${scale}")"
    selected_model="${scale_run}/models/${model_name}_epoch_$(printf "%04d" "${epoch_index}")"
    deployment_model="${scale_run}/models/vessel_${scale}_deployment_model"
    mkdir -p "${scale_run}"
    cd "${scale_run}"
    echo "Fine-tuning ${scale}: fixed epoch=${selected_epoch}, deployment cp=${threshold}"
    srun python -u "${TRAIN_SCRIPT}" --infer_3d --rng_seed "$((4600 + scale_index))" \
        --data_dir "${FINAL_RUN_ROOT}/all_prepared/${scale}" \
        --val_data_dir "${FINAL_RUN_ROOT}/all_prepared/${scale}" \
        --dataset_dir "${scale_run}/dataset" --model_name "${model_name}" --mask_id 1 \
        --n_epochs 80 --learning_rate 1e-5 --weight_decay 0.05 --batch_size 1 \
        --pretrained_model "${COMBINED_MODEL}" 2>&1 | tee "${scale_run}/train.log"
    require_file "${selected_model}"
    link_or_copy "${selected_model}" "${deployment_model}"

    qc_dir="${scale_run}/training_data_qc_not_validation"
    mkdir -p "${qc_dir}"
    srun python -u "${TEST_SCRIPT}" --model "${deployment_model}" \
        --test_data_dir "${FINAL_RUN_ROOT}/all_prepared/${scale}" --outdir "${qc_dir}" \
        --label_extension .seg.nrrd --cellprob_threshold "${threshold}" --min_size 15 \
        --mask_id 1 --infer_3d 2>&1 | tee "${qc_dir}/qc.log"
done

python - "${FINAL_RUN_ROOT}" "${FIXED_RECIPE}" "${CALIBRATION}" "${CV_SOURCE_RUN}" <<'PY'
import json
import sys
from pathlib import Path

root, recipe_path, calibration_path, source = map(Path, sys.argv[1:])
recipe = json.loads(recipe_path.read_text())
calibration = json.loads(calibration_path.read_text())
counts = {"unscaled": 10, "mid": 5, "coarse": 5}
config = {
    "schema_version": 1,
    "source_cv_run": str(source),
    "combined_pretraining": {
        **recipe["combined_pretraining"],
        "training_samples": 20,
        "model": str(root / "combined/models/vessel_combined_final_all_120ep"),
        "role": "shared precursor; not the preferred inference model",
    },
    "scales": {},
}
for scale in ("unscaled", "mid", "coarse"):
    fixed = recipe["scales"][scale]
    selected = calibration["scales"][scale]["best"]
    config["scales"][scale] = {
        "model": str(root / scale / "models" / f"vessel_{scale}_deployment_model"),
        "training_samples": counts[scale],
        "finetuning_epochs_run": 80,
        "selected_completed_epoch": fixed["selected_completed_epoch"],
        "cellprob_threshold": selected["cellprob_threshold"],
        "min_size": int(selected["min_size"]),
        "flow3D_smooth": float(selected["flow3D_smooth_scalar"]),
        "learning_rate": recipe["scale_finetuning"]["learning_rate"],
        "weight_decay": recipe["scale_finetuning"]["weight_decay"],
        "batch_size": recipe["scale_finetuning"]["batch_size"],
    }
(root / "deployment_config.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(config, indent=2))
PY

require_file "${FINAL_RUN_ROOT}/deployment_config.json"
echo "Final vessel deployment bundle: ${FINAL_RUN_ROOT}"
echo "Use the model and threshold under the matching scale in deployment_config.json."
