#!/bin/bash
#SBATCH --job-name=vessel-shape-cal
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-shape-cal-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/vessel-shape-cal-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
POSTPROCESS_SCRIPT="${REPO_ROOT}/vessel_analytical_postprocess.py"

# Prefer an explicit path:
# sbatch --export=ALL,CV_RUN_ROOT=/.../run_vessel_cv5_seed46_JOBID this_script.slurm
if [[ -z "${CV_RUN_ROOT:-}" ]]; then
    CV_RUN_ROOT=""
    for candidate in "${PROJECT_ROOT}"/runs/run_vessel_cv5_seed46_*; do
        [[ -d "${candidate}" && -f "${candidate}/COMPLETED" ]] || continue
        if [[ -z "${CV_RUN_ROOT}" || "${candidate}/COMPLETED" -nt "${CV_RUN_ROOT}/COMPLETED" ]]; then
            CV_RUN_ROOT="${candidate}"
        fi
    done
fi

if [[ -z "${CV_RUN_ROOT}" || ! -d "${CV_RUN_ROOT}" ]]; then
    echo "Set CV_RUN_ROOT to the completed run_vessel_cv5_seed46_* directory." >&2
    exit 1
fi
if [[ ! -f "${CV_RUN_ROOT}/COMPLETED" ]]; then
    echo "CV run has no COMPLETED marker: ${CV_RUN_ROOT}" >&2
    echo "Pass a completed vessel CV run via CV_RUN_ROOT." >&2
    exit 1
fi

RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_analytical_postprocess_${SLURM_JOB_ID}"
OUTPUT_DIR="${RUN_ROOT}/calibration"

WINDOW_SIZES="${WINDOW_SIZES:-10 12 15 20 25 30 35 40}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1

finish() {
    local status=$?
    trap - EXIT
    if [[ ${status} -eq 0 ]]; then
        touch "${RUN_ROOT}/COMPLETED"
    else
        printf 'exit_code=%s\n' "${status}" > "${RUN_ROOT}/FAILED"
    fi
    exit "${status}"
}
trap finish EXIT

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test -f "${POSTPROCESS_SCRIPT}"
test -f "${VENV_ACTIVATE}"
cp "$0" "${RUN_ROOT}/submitted_job.slurm"
python -m pip freeze > "${RUN_ROOT}/python_packages.txt"

echo "Source CV run: ${CV_RUN_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Fixed local window sizes: ${WINDOW_SIZES}"
date

read -r -a WINDOW_SIZE_ARGS <<< "${WINDOW_SIZES}"

srun python -u "${POSTPROCESS_SCRIPT}" calibrate \
    --cv_run_dir "${CV_RUN_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_folds 5 \
    --positive_overlap 0.5 \
    --negative_overlap 0.1 \
    --max_recall_loss 0.02 \
    --threshold_step 0.025 \
    --coarse_max_pooled_recall_loss 0.01 \
    --coarse_max_mean_recall_loss 0.01 \
    --coarse_max_sample_recall_loss 0.02 \
    --window_sizes "${WINDOW_SIZE_ARGS[@]}"

test -f "${OUTPUT_DIR}/deployment_thresholds.json"
test -f "${OUTPUT_DIR}/meta_cv_metrics_per_sample.csv"
test -f "${OUTPUT_DIR}/meta_cv_summary_by_scale.csv"
test -f "${OUTPUT_DIR}/meta_cv_summary_by_scale_scale_specific.csv"
test -f "${OUTPUT_DIR}/coarse_stage2_meta_cv_summary.csv"
test -f "${OUTPUT_DIR}/coarse_deployment_thresholds.json"

echo
echo "Main outputs:"
echo "  ${OUTPUT_DIR}/report.txt"
echo "  ${OUTPUT_DIR}/meta_cv_metrics_per_sample.csv"
echo "  ${OUTPUT_DIR}/meta_cv_summary_by_scale.csv"
echo "  ${OUTPUT_DIR}/meta_cv_summary_by_scale_scale_specific.csv"
echo "  ${OUTPUT_DIR}/coarse_stage2_meta_cv_summary.csv"
echo "  ${OUTPUT_DIR}/deployment_thresholds.json"
echo "  ${OUTPUT_DIR}/coarse_deployment_thresholds.json"
cat "${OUTPUT_DIR}/report.txt"
date
