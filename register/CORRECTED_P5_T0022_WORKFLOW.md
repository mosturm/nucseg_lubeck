# Corrected P5 Tomogram 0022

This workflow replaces only the masks derived from the incorrectly scaled
`tomo_reco_id0004_t0022` volume. It segments the corrected raw OME-Zarr,
merges its cell instances, backs up the old canonical cell and vessel masks,
recomputes the P5/0022 extraction partial, and then rebuilds the full
correlation analysis from all 12 tomograms.

## Input

`data_inference/tomos_to_segment/fullzarr/tomo_reco_id0004_t0022_raw_correct_axrot.ome.zarr`

The registration tables are not renamed or edited. Their P5/0022 coordinates
already target the canonical 0.65-um tomogram grid.

The incremental path requires the same registration-table inventory and row
counts as the source run. It intentionally stops if tables were added or rows
changed; submit the complete registration workflow after such a change.

## Submit

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"

export PROJECT_ROOT
cd "${REGISTER_ROOT}"
bash submit_corrected_p5_t0022.sh
```

The submitter automatically selects the newest completed cell model, vessel
deployment run, and full-registration run with all 12 extraction partials.
Set `SOURCE_RUN_ROOT` or `FINAL_RUN_ROOT` before submission to override either
selection.

## Output

The command prints a new run directory named:

`runs/run_corrected_p5_t0022_<timestamp>_<random>/`

Important files are:

- `submitted_pipeline.txt`: all job IDs and selected source/model runs.
- `replaced_mask_backup/`: the previous canonical P5/0022 masks.
- `promoted_masks.txt`: canonical corrected-mask paths.
- `features/`: the corrected P5/0022 extraction plus the 11 reused partials.
- `analysis_cov_80/`: rebuilt correlations, raw voxel tables, and plots.
- `analysis_cov_80/heatmaps/all_primary_associations__spearman.png`: primary result.
- `analysis_cov_80/heatmaps/all_associations_including_exploratory__spearman.png`:
  primary rows plus registration-flagged Portugal DKI `FA`, `MD`, and `MK`.

The old masks are retained. Portugal DKI remains excluded from the primary
heatmap and appears only in the explicitly inclusive exploratory overview.

## MRI Value Updates

Segmentation features can be reused when collaborators replace only MRI
predictor values and preserve every table row, MRI index, tomogram assignment,
corner coordinate, and coverage value. New SANDI maps require mapped tables
with the same `*_in_tomos.csv` geometry columns. The current full input has
Portugal DKI (`FA`, `MD`, `MK`) mappings but no full-run SANDI predictor table.
