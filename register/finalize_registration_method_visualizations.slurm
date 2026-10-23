#!/bin/bash
#SBATCH --job-name=reg-method-final
#SBATCH --partition=scc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

set -euo pipefail
: "${VIS_RUN_ROOT:?VIS_RUN_ROOT is not set}"

for rank_number in $(seq 1 8); do
  rank=$(printf '%02d' "${rank_number}")
  for panel in overview mri_voxels xpct_volume cell_instances correlation; do
    test -s "${VIS_RUN_ROOT}/figures/neighborhood_${rank}_${panel}.png"
    test -s "${VIS_RUN_ROOT}/figures/neighborhood_${rank}_${panel}.pdf"
  done
  test -s "${VIS_RUN_ROOT}/figures/neighborhood_${rank}_render_manifest.json"
done

find "${VIS_RUN_ROOT}/figures" -maxdepth 1 -type f -printf '%f\n' \
  | sort > "${VIS_RUN_ROOT}/output_inventory.txt"
touch "${VIS_RUN_ROOT}/COMPLETED"
echo "Verified eight neighborhoods and five PNG/PDF panel types per neighborhood."
echo "Output: ${VIS_RUN_ROOT}"

