#!/bin/bash
#SBATCH --job-name=full-cell-merge
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=full-cell-merge-%A_%a.out
#SBATCH --error=full-cell-merge-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"

SAMPLES=(
  tomo_reco_id0014_t0001 tomo_reco_id0014_t0008 tomo_reco_id0014_t0015
  tomo_reco_id0014_t0022 tomo_reco_id0014_t0029 tomo_reco_id0014_t0036
  tomo_reco_id0004_t0022 tomo_reco_id0004_t0029
  tomo_reco_id0011_t0001 tomo_reco_id0011_t0002 tomo_reco_id0011_t0003
  tomo_reco_id0011_t0004
)
SAMPLE_ID="${SAMPLES[${SLURM_ARRAY_TASK_ID}]}"
INPUT="${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25.ome.zarr"
OUTPUT="${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr"
RUN_ROOT="${PROJECT_ROOT}/runs/run_full_cell_merge_${SAMPLE_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1

if [[ -d "${OUTPUT}" ]] && python - "${OUTPUT}" <<'PY'
import sys
import zarr
root = zarr.open_group(sys.argv[1], mode="r")
array = root["0"]
assert tuple(array.shape) == (2156, 2560, 2560)
assert str(array.dtype) == "uint32"
PY
then
  echo "Already merged and structurally valid: ${OUTPUT}"
  touch "${RUN_ROOT}/COMPLETED"
  exit 0
fi
test -d "${INPUT}" || { echo "Missing cell mask: ${INPUT}" >&2; exit 2; }

python -u "${REPO_ROOT}/merge_block_instances_ome_zarr.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --array_key 0 \
  --min_contacts 3 \
  --min_reciprocal_overlap 0.25 \
  --compressor_level 5 \
  --audit_csv "${RUN_ROOT}/boundary_pair_audit.csv" \
  --summary_json "${RUN_ROOT}/instance_merge_summary.json"

touch "${RUN_ROOT}/COMPLETED"
echo "Merged: ${OUTPUT}"
