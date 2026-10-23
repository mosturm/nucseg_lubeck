#!/bin/bash
#SBATCH --job-name=full-mri-posthoc
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=full-mri-posthoc-%j.out
#SBATCH --error=full-mri-posthoc-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
RUN_ROOT="${RUN_ROOT:?Export RUN_ROOT for the completed full-registration run}"
COVERAGE_MIN="${COVERAGE_MIN:-80}"
MIN_CELLS_RADIUS="${MIN_CELLS_RADIUS:-5}"
ANALYSIS_ROOT="${RUN_ROOT}/analysis_cov_80"

test -s "${RUN_ROOT}/features/feature_manifest.csv"
mkdir -p "${ANALYSIS_ROOT}" "${PROJECT_ROOT}/.cache/matplotlib"
exec > >(tee -a "${RUN_ROOT}/posthoc_plots.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Reanalyzing existing features only: ${RUN_ROOT}/features"
echo "No segmentation masks will be opened and feature extraction will not rerun."

python -u "${REGISTER_ROOT}/analyze_full_mri_segmentation_correlations.py" \
  --features-dir "${RUN_ROOT}/features" \
  --output-dir "${ANALYSIS_ROOT}" \
  --coverage-min "${COVERAGE_MIN}" \
  --min-cells-radius "${MIN_CELLS_RADIUS}" \
  --spatial-block-size 3 \
  --folds 5 \
  --seed 46

python -u "${REGISTER_ROOT}/plot_full_association_heatmaps.py" \
  --results "${ANALYSIS_ROOT}/all_univariate_correlations_and_grouped_cv.csv" \
  --output-dir "${ANALYSIS_ROOT}/heatmaps" \
  --include-exploratory

python -u "${REGISTER_ROOT}/plot_full_r23_theory.py" \
  --features-dir "${RUN_ROOT}/features" \
  --theory-results "${ANALYSIS_ROOT}/r1_vs_r23_theory_fits.csv" \
  --output-dir "${ANALYSIS_ROOT}/r1_r23_plots" \
  --coverage-min "${COVERAGE_MIN}" \
  --min-cells "${MIN_CELLS_RADIUS}"

python -u "${REGISTER_ROOT}/plot_full_generalization_diagnostics.py" \
  --results "${ANALYSIS_ROOT}/all_univariate_correlations_and_grouped_cv.csv" \
  --raw-dir "${ANALYSIS_ROOT}/raw_voxel_tables" \
  --output-dir "${ANALYSIS_ROOT}/generalization_diagnostics"

touch "${RUN_ROOT}/POSTHOC_PLOTS_COMPLETED"
echo "Completed post-hoc analysis and plots: ${ANALYSIS_ROOT}"
