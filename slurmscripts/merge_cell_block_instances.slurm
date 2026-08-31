#!/bin/bash
#SBATCH --job-name=cell-merge-ids
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-merge-ids-%j.out
#SBATCH --error=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/runs/cell-merge-ids-%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
VENV_ACTIVATE="${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
SAMPLE_ID="${SAMPLE_ID:-tomo_reco_id0004_t0015}"
DATA_ROOT="${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr"
INPUT_LABELS="${INPUT_LABELS:-${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25.ome.zarr}"
OUTPUT_LABELS="${OUTPUT_LABELS:-${DATA_ROOT}/${SAMPLE_ID}_cells_cpneg2p25_instances_merged.ome.zarr}"
RUN_ROOT="${PROJECT_ROOT}/runs/run_cell_instance_merge_${SAMPLE_ID}_${SLURM_JOB_ID}"
MIN_CONTACTS="${MIN_CONTACTS:-3}"
MIN_RECIPROCAL_OVERLAP="${MIN_RECIPROCAL_OVERLAP:-0.25}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/workflow.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1

if [[ -e "${OUTPUT_LABELS}" ]]; then
    echo "Output already exists: ${OUTPUT_LABELS}" >&2
    exit 1
fi

python -u "${REPO_ROOT}/merge_block_instances_ome_zarr.py" \
    --input "${INPUT_LABELS}" \
    --output "${OUTPUT_LABELS}" \
    --array_key 0 \
    --min_contacts "${MIN_CONTACTS}" \
    --min_reciprocal_overlap "${MIN_RECIPROCAL_OVERLAP}" \
    --compressor_level 5 \
    --audit_csv "${RUN_ROOT}/boundary_pair_audit.csv" \
    --summary_json "${RUN_ROOT}/instance_merge_summary.json"

cp "$0" "${RUN_ROOT}/submitted_job.slurm"
touch "${RUN_ROOT}/COMPLETED"
echo "Corrected labels: ${OUTPUT_LABELS}"
echo "Merge summary: ${RUN_ROOT}/instance_merge_summary.json"
