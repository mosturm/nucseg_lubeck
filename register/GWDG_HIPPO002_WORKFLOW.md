# Hippo002 MRI-segmentation workflow on GWDG

## Runtime inputs

The workflow uses these existing segmentation outputs for tomo0022 and tomo0029:

```text
<sample>_cells_cpneg2p25_instances_merged.ome.zarr
<sample>_vessels_multiscale_union.ome.zarr
```

Merged cell instances are required by default because unreconciled inference
blocks bias cell radii downward. Vessel labels are treated semantically.

The collaborator's MRI-space control NIfTIs were mentioned in `info.txt` but
were not included in the supplied data. This version of the run therefore uses
`REQUIRE_CONTROLS=0`. Feature extraction, coverage QC, correlations, spatial
validation, and plots still run. The independent 48-hypothesis MRI axis/flip
test is skipped and this limitation is recorded in the log.

The MRI contrast values needed for regression are already contained in the
CSV tables. Original contrast NIfTIs are not required for this exploratory run.

## Files to upload

Upload these files to:

```text
${PROJECT_DIR}/nucseg/repo/nucseg_lubeck/register/
```

```text
extract_mri_segmentation_features.py
analyze_mri_segmentation_correlations.py
validate_segmentation_orientation.py
validate_table_against_nifti.py
investigate_dki_registration.py
audit_hippo002_tomogram_tables.py
run_hippo002_mri_segmentation.slurm
```

Upload both table directories there as well:

```text
Hippo002_tables_in_tomos/
Hippo002_table_correlation/
```

## Local Windows upload

Adjust only `$LocalRegister` if the repository is stored elsewhere:

```powershell
$Key = "$env:USERPROFILE\.ssh\id_ed25519_gwdg"
$HpcUser = "u30847"
$HpcHost = "glogin-gpu.hpc.gwdg.de"
$LocalRegister = "C:\Users\Moritz_Sturm\Desktop\Code\Lubeck\nucseg_lubeck\register"
$RemoteRegister = "/projects/scc/UGOE/UPFB/UPRP/scc_uprp_salditt/dir.project/nucseg/repo/nucseg_lubeck/register"

ssh -i "$Key" "${HpcUser}@${HpcHost}" "mkdir -p '$RemoteRegister'"

scp -i "$Key" `
  "$LocalRegister\extract_mri_segmentation_features.py" `
  "$LocalRegister\analyze_mri_segmentation_correlations.py" `
  "$LocalRegister\validate_segmentation_orientation.py" `
  "$LocalRegister\validate_table_against_nifti.py" `
  "$LocalRegister\investigate_dki_registration.py" `
  "$LocalRegister\audit_hippo002_tomogram_tables.py" `
  "$LocalRegister\run_hippo002_mri_segmentation.slurm" `
  "${HpcUser}@${HpcHost}:${RemoteRegister}/"

scp -i "$Key" -r `
  "$LocalRegister\Hippo002_tables_in_tomos" `
  "$LocalRegister\Hippo002_table_correlation" `
  "${HpcUser}@${HpcHost}:${RemoteRegister}/"
```

## Cluster preparation

Log in and set the project paths:

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
DATA_ROOT="${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr"

source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
# The current workflow dependency check includes nibabel even though the
# optional NIfTI validation is disabled for this run.
python -m pip install nibabel
```

Confirm that the four required segmentation outputs exist:

```bash
ls -ld \
  "${DATA_ROOT}/tomo_reco_id0004_t0022_cells_cpneg2p25_instances_merged.ome.zarr" \
  "${DATA_ROOT}/tomo_reco_id0004_t0029_cells_cpneg2p25_instances_merged.ome.zarr" \
  "${DATA_ROOT}/tomo_reco_id0004_t0022_vessels_multiscale_union.ome.zarr" \
  "${DATA_ROOT}/tomo_reco_id0004_t0029_vessels_multiscale_union.ome.zarr"
```

If either merged cell output is missing, submit the two reconciliation jobs:

```bash
MERGE_JOB_22=$(sbatch --parsable \
  --export="ALL,SAMPLE_ID=tomo_reco_id0004_t0022,INPUT_LABELS=${DATA_ROOT}/tomo_reco_id0004_t0022_cells_cpneg2p25.ome.zarr,OUTPUT_LABELS=${DATA_ROOT}/tomo_reco_id0004_t0022_cells_cpneg2p25_instances_merged.ome.zarr" \
  "${PROJECT_ROOT}/repo/nucseg_lubeck/slurmscripts/merge_cell_block_instances.slurm")

MERGE_JOB_29=$(sbatch --parsable \
  --export="ALL,SAMPLE_ID=tomo_reco_id0004_t0029,INPUT_LABELS=${DATA_ROOT}/tomo_reco_id0004_t0029_cells_cpneg2p25.ome.zarr,OUTPUT_LABELS=${DATA_ROOT}/tomo_reco_id0004_t0029_cells_cpneg2p25_instances_merged.ome.zarr" \
  "${PROJECT_ROOT}/repo/nucseg_lubeck/slurmscripts/merge_cell_block_instances.slurm")

echo "Merge jobs: ${MERGE_JOB_22}, ${MERGE_JOB_29}"
```

Wait until both are `COMPLETED` before submitting the MRI workflow:

```bash
sacct -j "${MERGE_JOB_22},${MERGE_JOB_29}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed
```

## Submit

```bash
export PROJECT_ROOT="${PROJECT_DIR}/nucseg"
export DATA_ROOT="${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr"
export REQUIRE_CONTROLS=0
export ALLOW_UNMERGED=0
export COVERAGE_MIN=80
export MIN_CELLS_RADIUS=5
export SPATIAL_BLOCK_SIZE=3

cd "${PROJECT_ROOT}/repo/nucseg_lubeck/register"
JOB_ID=$(sbatch --parsable --export=ALL run_hippo002_mri_segmentation.slurm)
echo "Submitted ${JOB_ID}"
```

Monitor it with:

```bash
squeue -j "${JOB_ID}" -o "%.18i %.24j %.10T %.12M %.40R"
tail -f "hippo002-mri-seg-${JOB_ID}.out"
```

The results are written to:

```text
${PROJECT_ROOT}/runs/run_hippo002_mri_segmentation_<JOB_ID>/
```

Important outputs:

```text
features/hippo002_<modality>_segmentation_features.csv.gz
analysis_cov_80/all_univariate_correlations_and_spatial_cv.csv
analysis_cov_80/sandi_rsoma_radius_definition_comparison.csv
analysis_cov_80/surface_volume_theory_fits.csv
analysis_cov_80/plots/*.png and *.pdf
audit/hippo002_tomogram_table_audit.*
dki_investigation/dki_registration_report.json
```

`COMPLETED` means the extraction, table/geometry checks, regressions, and plots
finished. `FAILED` contains the workflow exit code.

## Interpretation limits

- SANDI, MPM, and R2* are analysed; DKI is deliberately excluded.
- MRI footprints are locally affine reconstructions from neighbouring centres.
- Supplied `coverage_percent` is used for valid-volume denominators.
- The bounds-based coverage reconstruction cannot reproduce the collaborator's
  circular stitch mask exactly, so its difference is a QC diagnostic.
- Spatial CV is within Hippo002 and is not an independent-specimen test.
- High-order radius moments are reported only for MRI voxels containing at
  least five assigned cells by default.
- Independent validation against MRI-space control masks and source NIfTIs is
  not performed because those files were not supplied. Axis order follows the
  table legend: CSV `(i,j,k)=(x,y,z)` maps to Zarr `(z,y,x)=(k,j,i)`.

## Reanalyse extracted features with 80% coverage

This does not repeat OME-Zarr extraction. Upload these current files to
`${PROJECT_ROOT}/repo/nucseg_lubeck/register/`:

```text
analyze_mri_segmentation_correlations.py
plot_all_association_heatmaps.py
resume_hippo002_analysis.slurm
```

Then submit against the existing extraction run:

```bash
export PROJECT_ROOT="${PROJECT_DIR}/nucseg"
export SOURCE_RUN_ROOT="${PROJECT_ROOT}/runs/run_hippo002_mri_segmentation_16531563"
export COVERAGE_MIN=80
export MIN_CELLS_RADIUS=5
export SPATIAL_BLOCK_SIZE=3
export NEAR_ZERO_ABS_SPEARMAN=0.10

cd "${PROJECT_ROOT}/repo/nucseg_lubeck/register"
COV80_JOB=$(sbatch --parsable --export=ALL resume_hippo002_analysis.slurm)
echo "Submitted ${COV80_JOB}"
```

All new results are isolated from the 25% analysis:

```text
${SOURCE_RUN_ROOT}/analysis_cov_80/
  all_univariate_correlations_and_spatial_cv.csv
  near_zero_associations_abs_spearman_le_0p10.csv
  non_significant_associations_spearman_fdr_ge_0p05.csv
  plots/all_100_associations_spearman_heatmap.{png,pdf}
  plots/all_100_associations_spatial_cv_r2_heatmap.{png,pdf}
  plots/near_zero_associations/*.png
  plots/near_zero_associations/*.pdf
```

`near_zero_associations` uses the effect-size rule `abs(Spearman rho) <= 0.10`.
Its scatterplots include the fitted line, which should be approximately
horizontal. Statistical non-significance is reported separately because the
large R2* sample can make very small effects significant.

## Surface-volume radius moments

The current extraction additionally computes, per MRI footprint:

```text
cell_surface_to_volume_radius_inverse_um_inv = sum(R^2) / sum(R^3)
cell_volume_to_surface_radius_um             = sum(R^3) / sum(R^2)
```

For literal spheres, area/volume has an additional factor of 3 and volume/area
an additional factor of 1/3. The requested moment ratios are retained because
these constants only alter the fitted slope, not correlation, significance, or
R2.

These require a new scan of the merged cell-instance Zarrs because older
feature tables did not save `sum(R^2)`. They cannot be reconstructed exactly
from the previous aggregate columns. Submit the complete workflow with
`COVERAGE_MIN=80`; its analysis is written under `analysis_cov_80/` in the new
run directory.

For every MRI predictor, `surface_volume_theory_fits.csv` and
`plots/surface_volume_theory/` compare:

- a straight-line fit against `sum(R^2)/sum(R^3)`;
- the algebraically equivalent inverse fit against `sum(R^3)/sum(R^2)`;
- a naive straight-line fit against `sum(R^3)/sum(R^2)`.

The first two must have identical fit quality because the markers are exact
reciprocals. The informative comparison is the theoretical inverse curve versus
the naive straight line in volume/surface-radius space.

### Raw voxel-level analysis tables

The analysis also writes one compressed table per MRI modality:

```text
analysis_cov_80/raw_voxel_tables/hippo002_sandi_raw_voxel_values.csv.gz
analysis_cov_80/raw_voxel_tables/hippo002_mpm_raw_voxel_values.csv.gz
analysis_cov_80/raw_voxel_tables/hippo002_r2s_raw_voxel_values.csv.gz
```

These tables contain every row from the corresponding collaborator MRI table,
without any coverage filtering. This includes zero-coverage, low-coverage, and
invalid-geometry rows. They include MRI `(i,j,k)` coordinates, owning tomogram,
tomogram-space centers, the complete local 3x3 transform, MRI submodality values,
coverage, reconstructed footprint edge lengths and maximum body diagonal in
tomography voxels, equivalent total and covered tomography-voxel counts,
foreground soma/vessel voxel counts, cell count, `sum(R^p)` for
`p = 1,2,3,4,5,7`, fractions, density, and all derived radius markers.

`local_footprint_voxels_equivalent` is a continuous geometric volume in
tomography-voxel units (`abs(det(local basis))`), not an integer raster count.
This is appropriate because reconstructed MRI footprints are rotated/skewed
parallelepipeds and generally do not align with tomography voxel boundaries.
`covered_footprint_voxels_equivalent` additionally applies the collaborator's
coverage fraction. Indicator columns record whether a row entered the shared
coverage/mapping analysis, radius regressions, or soma/vessel-fraction
regressions. Therefore the raw tables preserve all MRI voxels while the
statistical analysis itself remains restricted to coverage >=80%.
