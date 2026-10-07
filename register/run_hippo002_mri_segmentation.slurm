#!/bin/bash
#SBATCH --job-name=hippo002-mri-seg
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=hippo002-mri-seg-%j.out
#SBATCH --error=hippo002-mri-seg-%j.err

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
REGISTER_ROOT="${REPO_ROOT}/register"
TABLES_DIR="${TABLES_DIR:-${REGISTER_ROOT}/Hippo002_tables_in_tomos}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
CONTROL_DIR="${CONTROL_DIR:-${PROJECT_ROOT}/data_inference/Hippo002_registration_controls}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_ROOT}/envs/cellpose-clean/bin/activate}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/run_hippo002_mri_segmentation_${SLURM_JOB_ID}}"

CELL_0022_MERGED="${DATA_ROOT}/tomo_reco_id0004_t0022_cells_cpneg2p25_instances_merged.ome.zarr"
CELL_0029_MERGED="${DATA_ROOT}/tomo_reco_id0004_t0029_cells_cpneg2p25_instances_merged.ome.zarr"
CELL_0022_UNMERGED="${DATA_ROOT}/tomo_reco_id0004_t0022_cells_cpneg2p25.ome.zarr"
CELL_0029_UNMERGED="${DATA_ROOT}/tomo_reco_id0004_t0029_cells_cpneg2p25.ome.zarr"
VESSEL_0022="${DATA_ROOT}/tomo_reco_id0004_t0022_vessels_multiscale_union.ome.zarr"
VESSEL_0029="${DATA_ROOT}/tomo_reco_id0004_t0029_vessels_multiscale_union.ome.zarr"

ALLOW_UNMERGED="${ALLOW_UNMERGED:-0}"
REQUIRE_CONTROLS="${REQUIRE_CONTROLS:-1}"
COVERAGE_MIN="${COVERAGE_MIN:-25}"
MIN_CELLS_RADIUS="${MIN_CELLS_RADIUS:-5}"
SPATIAL_BLOCK_SIZE="${SPATIAL_BLOCK_SIZE:-3}"

mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/.cache/matplotlib"
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
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Run root: ${RUN_ROOT}"
echo "Project root: ${PROJECT_ROOT}"
echo "GPU allocation is used as a scheduling target; extraction itself is CPU/I/O bound."
nvidia-smi || true

python - <<'PY'
import importlib.util
required = ["numpy", "pandas", "scipy", "matplotlib", "zarr", "nibabel"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("Python dependency check passed")
PY

for required in \
    "${TABLES_DIR}" \
    "${VESSEL_0022}" \
    "${VESSEL_0029}"; do
    test -e "${required}" || { echo "Missing required input: ${required}"; exit 2; }
done

choose_cell_labels() {
    local merged=$1
    local unmerged=$2
    if [[ -d "${merged}" ]]; then
        printf '%s\n' "${merged}"
    elif [[ "${ALLOW_UNMERGED}" == "1" && -d "${unmerged}" ]]; then
        echo "WARNING: using unreconciled cell instances; radius estimates may be biased downward" >&2
        printf '%s\n' "${unmerged}"
    else
        echo "Missing merged cell labels: ${merged}" >&2
        echo "Run instance reconciliation, or explicitly set ALLOW_UNMERGED=1 for exploratory output." >&2
        return 1
    fi
}

CELL_0022=$(choose_cell_labels "${CELL_0022_MERGED}" "${CELL_0022_UNMERGED}")
CELL_0029=$(choose_cell_labels "${CELL_0029_MERGED}" "${CELL_0029_UNMERGED}")

SANDI_TABLE=$(find "${TABLES_DIR}" -maxdepth 1 -name 'Hippo002_table_sandi*_in_tomos.csv' -print -quit)
MPM_TABLE=$(find "${TABLES_DIR}" -maxdepth 1 -name 'Hippo002_table_mpm*_in_tomos.csv' -print -quit)
R2S_TABLE=$(find "${TABLES_DIR}" -maxdepth 1 -name 'Hippo002_table_r2s*_in_tomos.csv' -print -quit)
DKI_TABLE=$(find "${TABLES_DIR}" -maxdepth 1 -name 'Hippo002_table_dki*_in_tomos.csv' -print -quit)
for table in "${SANDI_TABLE}" "${MPM_TABLE}" "${R2S_TABLE}" "${DKI_TABLE}"; do
    test -f "${table}" || { echo "Missing modality table"; exit 2; }
done

python -u "${REGISTER_ROOT}/audit_hippo002_tomogram_tables.py" \
    --tables-dir "${TABLES_DIR}" \
    --original-tables-dir "${REGISTER_ROOT}/Hippo002_table_correlation" \
    --outdir "${RUN_ROOT}/audit"

DKI_ARGS=()
if [[ -n "${DKI_SOURCE_NIFTI:-}" ]]; then
    DKI_ARGS+=(--source-nifti "${DKI_SOURCE_NIFTI}" --contrast "${DKI_CONTRAST:-FA}")
fi
python -u "${REGISTER_ROOT}/investigate_dki_registration.py" \
    --table "${DKI_TABLE}" \
    --output-dir "${RUN_ROOT}/dki_investigation" \
    "${DKI_ARGS[@]}"

python -u "${REGISTER_ROOT}/extract_mri_segmentation_features.py" \
    --tables-dir "${TABLES_DIR}" \
    --cell-0022 "${CELL_0022}" \
    --cell-0029 "${CELL_0029}" \
    --vessel-0022 "${VESSEL_0022}" \
    --vessel-0029 "${VESSEL_0029}" \
    --modalities sandi,mpm,r2s \
    --voxel-size-um 0.65 \
    --scan-chunks 32,256,256 \
    --coverage-sample-step-um 8 \
    --output-dir "${RUN_ROOT}/features"

declare -A TABLE_BY_MODALITY=(
    [sandi]="${SANDI_TABLE}"
    [mpm]="${MPM_TABLE}"
    [r2s]="${R2S_TABLE}"
)
declare -A SOURCE_COLUMN=(
    [sandi]="Rsoma"
    [mpm]="R1"
    [r2s]="R2s"
)

for modality in sandi mpm r2s; do
    upper=$(printf '%s' "${modality}" | tr '[:lower:]' '[:upper:]')
    control_variable="${upper}_CELL_CENTER_CONTROL"
    control="${CONTROL_DIR}/Hippo002_cells_center_nn_in_${modality}.nii.gz"
    if [[ -n "${!control_variable-}" ]]; then
        control="${!control_variable}"
    fi
    feature_table="${RUN_ROOT}/features/hippo002_${modality}_segmentation_features.csv.gz"
    if [[ -f "${control}" ]]; then
        python -u "${REGISTER_ROOT}/validate_segmentation_orientation.py" \
            --features "${feature_table}" \
            --control-nifti "${control}" \
            --output "${RUN_ROOT}/audit/${modality}_cell_center_orientation.json" \
            --require-identity-best
    elif [[ "${REQUIRE_CONTROLS}" == "1" ]]; then
        echo "Missing required MRI-space cell-center control: ${control}"
        exit 3
    else
        echo "WARNING: no ${modality} cell-center control; segmentation orientation was not independently checked"
    fi

    source_variable="${upper}_SOURCE_NIFTI"
    source_nifti=""
    if [[ -n "${!source_variable-}" ]]; then
        source_nifti="${!source_variable}"
    fi
    if [[ -n "${source_nifti}" ]]; then
        test -f "${source_nifti}" || { echo "Missing source NIfTI: ${source_nifti}"; exit 3; }
        python -u "${REGISTER_ROOT}/validate_table_against_nifti.py" \
            --table "${TABLE_BY_MODALITY[$modality]}" \
            --column "${SOURCE_COLUMN[${modality}]}" \
            --nifti "${source_nifti}" \
            --output "${RUN_ROOT}/audit/${modality}_source_nifti_orientation.json" \
            --require-identity-best
    fi
done

python -u "${REGISTER_ROOT}/analyze_mri_segmentation_correlations.py" \
    --features-dir "${RUN_ROOT}/features" \
    --output-dir "${RUN_ROOT}/analysis" \
    --coverage-min "${COVERAGE_MIN}" \
    --min-cells-radius "${MIN_CELLS_RADIUS}" \
    --spatial-block-size "${SPATIAL_BLOCK_SIZE}" \
    --folds 5 \
    --seed 46

cat > "${RUN_ROOT}/README_RESULTS.txt" <<EOF
Hippo002 MRI-segmentation exploratory workflow

features/   Joined MRI and segmentation feature tables
analysis/   Correlation tables, spatial-CV results, and figures
audit/      Coordinate, coverage, and 48-hypothesis orientation checks
dki_investigation/  DKI-only diagnostics; DKI was not included in correlations

Cell labels:
  tomo0022: ${CELL_0022}
  tomo0029: ${CELL_0029}

Interpretation remains exploratory because local-affine voxel footprints were
reconstructed from neighbouring registered centres. An independent specimen is
still required for a biological generalization estimate.
EOF

echo "Completed: ${RUN_ROOT}"
