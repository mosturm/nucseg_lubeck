#!/bin/bash
#SBATCH --job-name=p5t22-extract
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=p5t22-extract-%j.out
#SBATCH --error=p5t22-extract-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR:?PROJECT_DIR is not set}/nucseg}"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
INPUT_ROOT="${INPUT_ROOT:-${REGISTER_ROOT}/register_full_input}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT is not set}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is not set}"
SOURCE_PARTIALS="${SOURCE_RUN_ROOT}/features/partials"
TARGET_PARTIALS="${RUN_ROOT}/features/partials"

test -f "${SOURCE_RUN_ROOT}/COMPLETED"
test -d "${SOURCE_PARTIALS}"
test ! -e "${TARGET_PARTIALS}" || {
  echo "Target partial directory already exists: ${TARGET_PARTIALS}" >&2
  exit 2
}

mkdir -p "${TARGET_PARTIALS}" "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/extract_corrected_p5_t0022.log") 2>&1

module load gcc/13.2.0
module load python/3.11.9
source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Reused partial arrays are row-aligned to the registration tables. Refuse the
# incremental route when datasets or row counts changed; use the full pipeline
# in that case so every affected partial is regenerated.
python - "${REGISTER_ROOT}" "${INPUT_ROOT}" "${SOURCE_RUN_ROOT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

register_root, input_root, source_root = map(Path, sys.argv[1:])
sys.path.insert(0, str(register_root))
from full_registration_config import discover_tables

manifest_path = source_root / "features" / "feature_manifest.csv"
manifest = pd.read_csv(manifest_path).set_index("dataset_id")
specs = discover_tables(input_root)
current_ids = {spec.dataset_id for spec in specs}
source_ids = set(manifest.index.astype(str))
if current_ids != source_ids:
    raise RuntimeError(
        "Registration datasets changed since the source extraction. "
        "Run submit_full_registration.sh instead of reusing partials. "
        f"Added={sorted(current_ids - source_ids)}, removed={sorted(source_ids - current_ids)}"
    )
for spec in specs:
    current_rows = sum(1 for _ in spec.path.open("rb")) - 1
    source_rows = int(manifest.loc[spec.dataset_id, "input_rows"])
    if current_rows != source_rows:
        raise RuntimeError(
            f"{spec.dataset_id}: row count changed from {source_rows} to {current_rows}; "
            "run the full extraction pipeline"
        )
print(f"Registration table inventory unchanged: {len(specs)} datasets")
PY

# Reuse immutable partials for the other 11 tomograms without copying their data.
for source in "${SOURCE_PARTIALS}"/*; do
  name="$(basename "${source}")"
  if [[ "${name}" != "P5_0022" ]]; then
    cp -al "${source}" "${TARGET_PARTIALS}/${name}"
  fi
done

reused_count="$(find "${TARGET_PARTIALS}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${reused_count}" -ne 11 ]]; then
  echo "Expected 11 reusable partial directories, found ${reused_count}" >&2
  exit 2
fi

python -u "${REGISTER_ROOT}/extract_full_mri_segmentation_features.py" \
  --input-root "${INPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${RUN_ROOT}/features" \
  --array-key 0 \
  --voxel-size-um 0.65 \
  --scan-chunks 32,256,256 \
  --task-index 6

touch "${RUN_ROOT}/CORRECTED_PARTIAL_COMPLETED"
echo "Recomputed P5/0022 and linked the other 11 extraction partials from ${SOURCE_RUN_ROOT}"
