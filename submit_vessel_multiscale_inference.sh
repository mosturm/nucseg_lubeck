#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg}"
REPO_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck"
SLURM_ROOT="${REPO_ROOT}/slurmscripts"

: "${INPUT_ZARR:?Set INPUT_ZARR to the unscaled raw OME-Zarr}"
: "${SAMPLE_ID:?Set SAMPLE_ID}"
: "${FINAL_RUN_ROOT:?Set FINAL_RUN_ROOT to the completed final vessel-model run}"
MID_FACTOR_ZYX="${MID_FACTOR_ZYX:-4.269230769230769x4.266666666666667x4.266666666666667}"
COARSE_FACTOR_ZYX="${COARSE_FACTOR_ZYX:-14.223076923076924x14.222222222222221x14.222222222222221}"

DEPLOYMENT_CONFIG="${FINAL_RUN_ROOT}/deployment_config.json"
test -d "${INPUT_ZARR}"
test -f "${DEPLOYMENT_CONFIG}"
test -f "${FINAL_RUN_ROOT}/COMPLETED"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_multiscale_${SAMPLE_ID}_${RUN_TAG}"
OUTPUT_ZARR="${OUTPUT_ZARR:-$(dirname "${INPUT_ZARR}")/${SAMPLE_ID}_vessels_multiscale_union.ome.zarr}"
if [[ -e "${OUTPUT_ZARR}" && "${ALLOW_OVERWRITE:-0}" != 1 ]]; then
  echo "Output already exists: ${OUTPUT_ZARR}" >&2
  echo "Set OUTPUT_ZARR to a new path or explicitly set ALLOW_OVERWRITE=1." >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

common_export="ALL,RUN_ROOT=${RUN_ROOT},NUM_WORKERS=8"
prepare_job="$(sbatch --parsable \
  --export="${common_export},INPUT_ZARR=${INPUT_ZARR},MID_FACTOR_ZYX=${MID_FACTOR_ZYX},COARSE_FACTOR_ZYX=${COARSE_FACTOR_ZYX}" \
  "${SLURM_ROOT}/prepare_vessel_multiscale_inference.slurm")"

gpu_job="$(sbatch --parsable \
  --dependency="afterok:${prepare_job}" \
  --array=0-7%4 \
  --export="${common_export},DEPLOYMENT_CONFIG=${DEPLOYMENT_CONFIG}" \
  "${SLURM_ROOT}/infer_vessel_multiscale_array.slurm")"

finalize_job="$(sbatch --parsable \
  --dependency="afterok:${gpu_job}" \
  --export="${common_export},OUTPUT_ZARR=${OUTPUT_ZARR}" \
  "${SLURM_ROOT}/finalize_vessel_multiscale_inference.slurm")"

cat > "${RUN_ROOT}/submitted_pipeline.txt" <<EOF
input=${INPUT_ZARR}
sample_id=${SAMPLE_ID}
final_run_root=${FINAL_RUN_ROOT}
deployment_config=${DEPLOYMENT_CONFIG}
mid_factor_zyx=${MID_FACTOR_ZYX}
coarse_factor_zyx=${COARSE_FACTOR_ZYX}
prepare_job=${prepare_job}
gpu_array_job=${gpu_job}
finalize_job=${finalize_job}
output=${OUTPUT_ZARR}
EOF

echo "Multiscale vessel pipeline submitted"
echo "  run root:     ${RUN_ROOT}"
echo "  prepare:      ${prepare_job}"
echo "  GPU array:    ${gpu_job}_[0-7]"
echo "  final union:  ${finalize_job}"
echo "  output:       ${OUTPUT_ZARR}"
echo
echo "Monitor with:"
echo "  squeue -j ${prepare_job},${gpu_job},${finalize_job} -o '%.18i %.24j %.10T %.12M %.40R'"
