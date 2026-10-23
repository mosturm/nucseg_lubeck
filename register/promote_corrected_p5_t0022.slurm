#!/bin/bash
#SBATCH --job-name=p5t22-promote
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=p5t22-promote-%j.out
#SBATCH --error=p5t22-promote-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
CORRECTION_RUN_ROOT="${CORRECTION_RUN_ROOT:?CORRECTION_RUN_ROOT is not set}"
STAGING_DIR="${STAGING_DIR:?STAGING_DIR is not set}"
SAMPLE_ID="tomo_reco_id0004_t0022"

STAGED_CELL="${STAGING_DIR}/${SAMPLE_ID}_cells_cpneg2p25.ome.zarr"
STAGED_CELL_JSON="${STAGING_DIR}/${SAMPLE_ID}_cells_cpneg2p25.ome.json"
STAGED_MERGED="${STAGING_DIR}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr"
STAGED_VESSEL="${STAGING_DIR}/${SAMPLE_ID}_vessels_multiscale_union.ome.zarr"

CANONICAL_CELL="${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25.ome.zarr"
CANONICAL_CELL_JSON="${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25.ome.json"
CANONICAL_MERGED="${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr"
CANONICAL_VESSEL="${DATA_ROOT}/${SAMPLE_ID}_vessels_multiscale_union.ome.zarr"
BACKUP_DIR="${CORRECTION_RUN_ROOT}/replaced_mask_backup"
promotion_started=0

rollback() {
  status=$?
  trap - EXIT
  if [[ ${status} -ne 0 && ${promotion_started} -eq 1 ]]; then
    echo "Mask promotion failed; restoring the previous canonical masks." >&2
    pairs=(
      "${CANONICAL_CELL}|${BACKUP_DIR}/$(basename "${CANONICAL_CELL}")"
      "${CANONICAL_MERGED}|${BACKUP_DIR}/$(basename "${CANONICAL_MERGED}")"
      "${CANONICAL_VESSEL}|${BACKUP_DIR}/$(basename "${CANONICAL_VESSEL}")"
      "${CANONICAL_CELL_JSON}|${BACKUP_DIR}/$(basename "${CANONICAL_CELL_JSON}")"
    )
    for pair in "${pairs[@]}"; do
      canonical="${pair%%|*}"
      backup="${pair#*|}"
      if [[ -e "${backup}" ]]; then
        if [[ -e "${canonical}" ]]; then
          mv "${canonical}" "${STAGING_DIR}/promotion_failed_${SLURM_JOB_ID}_$(basename "${canonical}")"
        fi
        mv "${backup}" "${canonical}"
      fi
    done
  fi
  exit "${status}"
}
trap rollback EXIT

mkdir -p "${CORRECTION_RUN_ROOT}" "${BACKUP_DIR}"
exec > >(tee -a "${CORRECTION_RUN_ROOT}/promote_masks.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1

test -d "${STAGED_CELL}"
test -d "${STAGED_VESSEL}"

python - "${STAGED_CELL}" "${STAGED_VESSEL}" <<'PY'
import sys
import zarr

expected = (2156, 2560, 2560)
for path in sys.argv[1:]:
    array = zarr.open_group(path, mode="r")["0"]
    if tuple(array.shape) != expected:
        raise ValueError(f"{path}: shape {array.shape}, expected {expected}")
    if array.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D array")
    print(f"Validated {path}: shape={array.shape}, dtype={array.dtype}")
PY

python -u "${REPO_ROOT}/merge_block_instances_ome_zarr.py" \
  --input "${STAGED_CELL}" \
  --output "${STAGED_MERGED}" \
  --array_key 0 \
  --min_contacts 3 \
  --min_reciprocal_overlap 0.25 \
  --compressor_level 5 \
  --audit_csv "${CORRECTION_RUN_ROOT}/boundary_pair_audit.csv" \
  --summary_json "${CORRECTION_RUN_ROOT}/instance_merge_summary.json"

python - "${STAGED_MERGED}" <<'PY'
import sys
import zarr

array = zarr.open_group(sys.argv[1], mode="r")["0"]
if tuple(array.shape) != (2156, 2560, 2560) or str(array.dtype) != "uint32":
    raise ValueError(f"Invalid merged cell mask: shape={array.shape}, dtype={array.dtype}")
print(f"Validated merged mask: shape={array.shape}, dtype={array.dtype}")
PY

for path in "${CANONICAL_CELL}" "${CANONICAL_MERGED}" "${CANONICAL_VESSEL}"; do
  test -d "${path}" || { echo "Missing canonical mask to back up: ${path}" >&2; exit 2; }
done

promotion_started=1
mv "${CANONICAL_CELL}" "${BACKUP_DIR}/"
mv "${CANONICAL_MERGED}" "${BACKUP_DIR}/"
mv "${CANONICAL_VESSEL}" "${BACKUP_DIR}/"
if [[ -f "${CANONICAL_CELL_JSON}" ]]; then
  mv "${CANONICAL_CELL_JSON}" "${BACKUP_DIR}/"
fi

mv "${STAGED_CELL}" "${CANONICAL_CELL}"
mv "${STAGED_MERGED}" "${CANONICAL_MERGED}"
mv "${STAGED_VESSEL}" "${CANONICAL_VESSEL}"
if [[ -f "${STAGED_CELL_JSON}" ]]; then
  mv "${STAGED_CELL_JSON}" "${CANONICAL_CELL_JSON}"
fi
promotion_started=0

cat > "${CORRECTION_RUN_ROOT}/promoted_masks.txt" <<EOF
sample_id=${SAMPLE_ID}
canonical_cell=${CANONICAL_CELL}
canonical_merged_cell=${CANONICAL_MERGED}
canonical_vessel=${CANONICAL_VESSEL}
old_masks_backup=${BACKUP_DIR}
EOF
touch "${CORRECTION_RUN_ROOT}/MASKS_PROMOTED"
trap - EXIT
echo "Corrected P5/0022 masks promoted; old masks preserved in ${BACKUP_DIR}"
