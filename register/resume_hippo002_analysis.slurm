#!/bin/bash
#SBATCH --job-name=hippo002-resume
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=hippo002-resume-%j.out
#SBATCH --error=hippo002-resume-%j.err

set -euo pipefail
: "${SOURCE_RUN_ROOT:?Export SOURCE_RUN_ROOT from the completed extraction job}"

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/envs/cellpose-clean/bin/activate}"

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

for modality in sandi mpm r2s; do
    feature="${SOURCE_RUN_ROOT}/features/hippo002_${modality}_segmentation_features.csv.gz"
    test -s "${feature}" || { echo "Missing feature table: ${feature}"; exit 2; }
done

COVERAGE_MIN="${COVERAGE_MIN:-80}"
ANALYSIS_NAME="${ANALYSIS_NAME:-analysis_cov_80}"
ANALYSIS_DIR="${SOURCE_RUN_ROOT}/${ANALYSIS_NAME}"

python -u "${REGISTER_ROOT}/analyze_mri_segmentation_correlations.py" \
    --features-dir "${SOURCE_RUN_ROOT}/features" \
    --output-dir "${ANALYSIS_DIR}" \
    --coverage-min "${COVERAGE_MIN}" \
    --min-cells-radius "${MIN_CELLS_RADIUS:-5}" \
    --spatial-block-size "${SPATIAL_BLOCK_SIZE:-3}" \
    --folds 5 \
    --seed 46 \
    --near-zero-abs-spearman "${NEAR_ZERO_ABS_SPEARMAN:-0.10}"

python -u "${REGISTER_ROOT}/plot_all_association_heatmaps.py" \
    --results "${ANALYSIS_DIR}/all_univariate_correlations_and_spatial_cv.csv" \
    --output-dir "${ANALYSIS_DIR}/plots"

cat > "${SOURCE_RUN_ROOT}/ANALYSIS_COV80_NOTE.txt" <<EOF
Existing extracted feature tables were reused; OME-Zarr extraction was not repeated.
Analysis SLURM job: ${SLURM_JOB_ID}
Coverage threshold: ${COVERAGE_MIN}%
Output directory: ${ANALYSIS_DIR}
EOF
touch "${SOURCE_RUN_ROOT}/ANALYSIS_COV80_COMPLETED"
echo "Coverage-${COVERAGE_MIN} analysis completed: ${ANALYSIS_DIR}"
