#!/bin/bash
#SBATCH --job-name=full-mri-xpct
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=full-mri-xpct-%j.out
#SBATCH --error=full-mri-xpct-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
INPUT_ROOT="${INPUT_ROOT:-${REGISTER_ROOT}/register_full_input}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/run_full_mri_segmentation_${SLURM_JOB_ID}}"
COVERAGE_MIN="${COVERAGE_MIN:-80}"
MIN_CELLS_RADIUS="${MIN_CELLS_RADIUS:-5}"
mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/.cache/matplotlib"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1

finish() {
  status=$?
  trap - EXIT
  if [[ ${status} -eq 0 ]]; then touch "${RUN_ROOT}/COMPLETED"; else echo "exit_code=${status}" > "${RUN_ROOT}/FAILED"; fi
  exit "${status}"
}
trap finish EXIT

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Run root: ${RUN_ROOT}"
echo "Input root: ${INPUT_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Extraction is CPU/I/O bound; the GPU allocation is a scheduling target."

python -u "${REGISTER_ROOT}/audit_full_registration_inputs.py" \
  --input-root "${INPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/preflight"

python -u "${REGISTER_ROOT}/extract_full_mri_segmentation_features.py" \
  --input-root "${INPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/features" \
  --array-key 0 \
  --voxel-size-um 0.65 \
  --scan-chunks 32,256,256 \
  --finalize-only

python -u "${REGISTER_ROOT}/analyze_full_mri_segmentation_correlations.py" \
  --features-dir "${RUN_ROOT}/features" \
  --output-dir "${RUN_ROOT}/analysis_cov_80" \
  --coverage-min "${COVERAGE_MIN}" \
  --min-cells-radius "${MIN_CELLS_RADIUS}" \
  --spatial-block-size 3 \
  --folds 5 \
  --seed 46

python -u "${REGISTER_ROOT}/plot_full_association_heatmaps.py" \
  --results "${RUN_ROOT}/analysis_cov_80/all_univariate_correlations_and_grouped_cv.csv" \
  --output-dir "${RUN_ROOT}/analysis_cov_80/heatmaps" \
  --include-exploratory

python -u "${REGISTER_ROOT}/plot_full_r23_theory.py" \
  --features-dir "${RUN_ROOT}/features" \
  --theory-results "${RUN_ROOT}/analysis_cov_80/r1_vs_r23_theory_fits.csv" \
  --output-dir "${RUN_ROOT}/analysis_cov_80/r1_r23_plots" \
  --coverage-min "${COVERAGE_MIN}" \
  --min-cells "${MIN_CELLS_RADIUS}"

cat > "${RUN_ROOT}/README_RESULTS.txt" <<EOF
Full MRI-XPCT registration analysis

preflight/                 Input-table and segmentation-mask audit
features/tables/           MRI voxel tables with segmentation features
features/per_cell_radius_tables/  Every assigned cell radius and provenance
analysis_cov_80/raw_voxel_tables/ All MRI voxels, plus analysis inclusion flags
analysis_cov_80/heatmaps/  Pooled-protocol Spearman and held-out R2 heatmaps
analysis_cov_80/r1_r23_plots/ R1 = intercept + slope * r23 theory plots

Primary pooled validation is leave-one-specimen-out across P2/P5/P7.
Portugal DKI is extracted but excluded from primary heatmaps by default.
Only r23=sum(r^2)/sum(r^3) is analyzed; reciprocal r3/r2 is not produced.
Missing vessel masks yield NA vessel metrics without removing cell/radius results.
EOF

echo "Completed: ${RUN_ROOT}"
