# MRI-XPCT registration method visualization

This workflow selects eight compact six-voxel MRI neighborhoods and renders the
same registered volume in five overview representations:

1. MRI voxels colored by the measured predictor.
2. The corresponding XPCT subvolume with the exact MRI footprints in red.
3. Cellpose instances inside those footprints, colored by instance ID (every
   tenth ID is shown to keep the volume readable).
4. The same registered MRI footprints colored by mean sphere-equivalent cell
   radius in micrometres.
5. The observed MRI value versus mean sphere-equivalent cell radius, including
   all radius-eligible voxels and highlighting the six illustrated voxels.

Four standalone high-resolution panels are also saved for MRI, XPCT, cell
instances, and the correlation. Selection requires at least
80% coverage, a compact neighborhood, a visible local cell-density gradient,
and agreement between the local and dataset-wide Spearman direction. The same
fixed neighborhoods are subsequently displayed using mean cell radius. Positive held-out R2 is
not required and is not used for ranking. The correlation panel deliberately
omits Spearman and grouped held-out annotations. These neighborhoods are method
illustrations selected for visual concordance, not independent validation
examples.

The local layout preview uses orthographic unit cubes. The cluster renderer
draws the eight mapped XPCT corners and rigidly reorients the complete scene
into an MRI-axis-aligned orthographic display frame. This matches the preview
camera while preserving real anisotropy, shear, and relative geometry.
For the XPCT panel, only the exposed faces of the six-voxel union are sampled
(48 x 48 raw samples per face); the enclosed XPCT interior is not loaded.
The instance panel must retain interior information, so it reads a strided mask
subvolume capped at 180 samples per axis and then keeps points inside the exact
registered footprints.

## Files

- `select_registration_neighborhoods.py`: table-based selection and local preview.
- `render_registration_neighborhood.py`: real XPCT and merged-instance renderer.
- `select_registration_neighborhoods.slurm`: selection job.
- `render_registration_neighborhoods_array.slurm`: eight parallel render tasks.
- `finalize_registration_method_visualizations.slurm`: completeness check.
- `submit_registration_method_visualizations.sh`: submits the complete pipeline.

## Cluster run

Set `ANALYSIS_DIR` explicitly to the newest completed `analysis_cov_80`
directory. The following command finds the correlation table modified most
recently and prints the selected directory before submission:

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
DATA_ROOT="${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr"

ANALYSIS_DIR=$(find "${PROJECT_ROOT}/runs" -mindepth 3 -maxdepth 3 \
  -type f -name all_univariate_correlations_and_grouped_cv.csv \
  -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)

echo "ANALYSIS_DIR=${ANALYSIS_DIR}"
test -s "${ANALYSIS_DIR}/all_univariate_correlations_and_grouped_cv.csv"
test -d "${ANALYSIS_DIR}/raw_voxel_tables"

export PROJECT_ROOT REGISTER_ROOT DATA_ROOT ANALYSIS_DIR
cd "${REGISTER_ROOT}"
bash submit_registration_method_visualizations.sh
```

The submitter prints the new run directory. It has this structure:

```text
runs/run_registration_method_visualization_<timestamp>/
  submitted_pipeline.txt
  selection/
    selected_neighborhoods.csv
    selected_neighborhood_voxels.csv.gz
    selected_neighborhoods_overview.png
    candidate_inventory.csv
  figures/
    neighborhood_01_overview.png
    neighborhood_01_mri_voxels.png
    neighborhood_01_xpct_volume.png
    neighborhood_01_cell_instances.png
    neighborhood_01_correlation.png
    ...matching PDFs and ranks 02-08...
  output_inventory.txt
  COMPLETED
```

Each `neighborhood_XX_render_manifest.json` records the exact input raw volume,
merged mask, XPCT bounding box, rendering stride, and cell sampling settings.
