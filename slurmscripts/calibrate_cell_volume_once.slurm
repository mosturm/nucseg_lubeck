#!/bin/bash
#SBATCH --job-name=cell-volume-cal
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-volume-cal-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-volume-cal-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
CV_DATA_ROOT="${PROJECT_ROOT}/data_cell_only/cv5_seed46"
CV_RUN_ROOT="${CV_RUN_ROOT:-${PROJECT_ROOT}/runs/run_cell_cv5_seed46_15083187}"
SWEEP_SCRIPT="${REPO_ROOT}/sweep_thresholds_segment_nuc.py"
AGGREGATE_SCRIPT="${REPO_ROOT}/aggregate_cell_calibration.py"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
CALIBRATION_PREFIX="deployment_calibration_epoch51"
CALIBRATION_STAGE="volume_balance"
MAX_DICE_DROP="${MAX_DICE_DROP:-0.01}"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/.cellpose/models"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MPLBACKEND="Agg"
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
require_file "${AGGREGATE_SCRIPT}"
require_file "${VENV_ACTIVATE}"
require_dir "${CV_DATA_ROOT}"
require_dir "${CV_RUN_ROOT}"

# By default, reuse the newest evaluation/calibration run containing all five
# cross-fitted epoch-51 models. Override with CALIBRATION_SOURCE_RUN if needed.
if [[ -z "${CALIBRATION_SOURCE_RUN:-}" ]]; then
    CALIBRATION_SOURCE_RUN="$(
        python -c 'import sys; from pathlib import Path; root=Path(sys.argv[1]); candidates=sorted(root.glob("run_cell_fixed_eval_calibration_*"), key=lambda p: p.stat().st_mtime, reverse=True); valid=[p for p in candidates if all((p / "deployment_calibration" / f"fold_{fold:02d}" / "models" / f"cell_calibration_fold_{fold:02d}_120ep_epoch_0050").is_file() for fold in range(1,6))]; print(valid[0] if valid else "")' \
            "${PROJECT_ROOT}/runs"
    )"
fi
if [[ -z "${CALIBRATION_SOURCE_RUN}" ]]; then
    echo "No completed calibration-model run found." >&2
    echo "Submit with: sbatch --export=ALL,CALIBRATION_SOURCE_RUN=/path/to/run $0" >&2
    exit 1
fi

CALIBRATION_ROOT="${CALIBRATION_SOURCE_RUN}/deployment_calibration"
require_dir "${CALIBRATION_ROOT}"

IFS=' ' read -r -a calibration_values <<< "$(
    python -c 'print(" ".join(f"{-6.0+0.25*i:g}" for i in range(33)))'
)"

echo "Source calibration models: ${CALIBRATION_SOURCE_RUN}"
echo "Maximum allowed macro Dice drop: ${MAX_DICE_DROP}"
echo "Thresholds: ${calibration_values[*]}"

for fold in 1 2 3 4 5; do
    fold_tag="$(printf "%02d" "${fold}")"
    model_path="${CALIBRATION_ROOT}/fold_${fold_tag}/models/cell_calibration_fold_${fold_tag}_120ep_epoch_0050"
    calibration_dir="${CALIBRATION_ROOT}/fold_${fold_tag}/threshold_sweep_${CALIBRATION_STAGE}"
    heldout_dir="${CV_DATA_ROOT}/fold_${fold_tag}/combined/Testdaten"

    require_file "${model_path}"
    require_dir "${heldout_dir}"
    mkdir -p "${calibration_dir}"

    echo
    echo "Volume-calibration sweep: fold ${fold_tag}"
    date
    srun python -u "${SWEEP_SCRIPT}" \
        --data_dir "${heldout_dir}" \
        --model_path "${model_path}" \
        --output_dir "${calibration_dir}" \
        --infer_3d \
        --gt_foreground_mode label1 \
        --gt_foreground_label 1 \
        --cellprob_values "${calibration_values[@]}" \
        --min_size_values 15 \
        --flow3D_smooth_values 0 \
        2>&1 | tee "${calibration_dir}/sweep.log"
done

srun python -u "${AGGREGATE_SCRIPT}" \
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
echo "Selected volume-balanced deployment threshold:"
cat "${CV_RUN_ROOT}/${CALIBRATION_PREFIX}_${CALIBRATION_STAGE}_report.txt"
