# Hippo002 MRI to XPCT registration and correlation

This directory contains the first audit and modelling tools for relating MRI
contrasts to cell and vessel measurements derived from the XPCT segmentations.

## Current status

The four supplied correlation tables are internally well formed and suitable
for inspection. They contain one row per MRI voxel, MRI contrast values, the
mapped MRI-voxel centre in a claimed isotropic 650 nm segmentation frame, and
an XPCT coverage estimate.

They are **not yet sufficient for exact MRI-voxel-to-segmentation aggregation**.
The package does not contain the NIfTI images and headers, the complete ordered
transform chain, the registered control images, or the exact stitched XPCT
reference and stitching provenance. A centre coordinate alone does not describe
the rotated or deformed footprint of an MRI voxel in segmentation space.

The supplied coordinates also refer to a stitched volume made from
`tomo_reco_id0004_t0022` and `tomo_reco_id0004_t0029`. They must not be used as
indices into either individual production OME-Zarr until the stitch transform,
overlap removal, crop offsets, axis order, origin, and spacing have been
reproduced.

### Update: direct tomogram coordinates

The archive `Hippo002_table_sandi_portugal_0p25mm_in_tomos.zip` was added on
2026-10-06. Despite its name, it contains updated DKI, MPM, R2*, and SANDI
tables. The extracted files are in `Hippo002_tables_in_tomos/`.

These tables preserve every row and value from the original tables and add the
same MRI-voxel centre in the original tomo0022 and tomo0029 arrays. They define
CSV axes `(i,j,k)=(x,y,z)` and OME-Zarr access order `(z,y,x)=(k,j,i)` for an
individual tomogram shape of `(z,y,x)=(2156,2560,2560)`. All in-volume flags
agree with these dimensions in the audit.

This is enough to spot-check registered centres directly in the production
OME-Zarr masks and removes the need to reproduce the stitch merely to locate a
centre. It is not yet an exact MRI-voxel footprint: density and volume-fraction
targets require the transformed corners/dense sampling lattice or the complete
transform. A locally affine footprint can be approximated from neighbouring
centre coordinates for an exploratory analysis, but that approximation must be
labelled and tested against collaborator-derived coverage.

## Files

- `Hippo002_table_correlation/`: extracted collaborator tables and legend.
- `audit_hippo002_registration.py`: structural and numerical table audit.
- `audit_hippo002_tomogram_tables.py`: verifies the new direct-to-tomogram
  coordinates, bounds flags, centre lattice, and equality with the old tables.
- `validate_table_against_nifti.py`: checks table values against a source NIfTI
  and ranks all 48 axis permutation/flip hypotheses.
- `run_spatial_elastic_net.py`: nested, spatially grouped Elastic Net baseline
  for a final joined MRI/segmentation table.
- `extract_mri_segmentation_features.py`: streams the two cell/vessel OME-Zarr
  pairs, reconstructs local MRI footprints, and extracts soma/vessel features.
- `analyze_mri_segmentation_correlations.py`: univariate Pearson/Spearman,
  linear regression, spatially blocked validation, FDR tables, and plots.
- `validate_segmentation_orientation.py`: tests all 48 MRI index axis/flip
  hypotheses against the collaborator's MRI-space cell-centre masks.
- `investigate_dki_registration.py`: separate DKI geometry and source-NIfTI
  diagnostics; it does not clear DKI for the main correlations.
- `run_hippo002_mri_segmentation.slurm`: one GPU-node workflow for extraction,
  orientation gates, regressions, plots, and the separate DKI audit.
- `GWDG_HIPPO002_WORKFLOW.md`: upload, input, submission, and output guide.
- `CORRECTED_P5_T0022_WORKFLOW.md`: staged replacement and incremental
  reanalysis guide for the corrected P5 tomogram 0022.
- `registration_requirements.json`: machine-readable delivery and acceptance
  checklist.
- `audit_output/`: generated audit reports.

## Audit findings

| MRI family | Resolution | Rows | Contrasts | Median XPCT coverage | Rows at coverage >=25% |
| --- | ---: | ---: | --- | ---: | ---: |
| DKI | 0.10 mm | 11,678 | FA, MD, AD, AK, MK, MSD, MSK, RD, RK, S0, b0 | 100.00% | 10,381 |
| MPM | 0.30 mm | 625 | R1, R2s, A | 71.69% | 436 |
| R2* | 0.05 mm | 88,221 | R2s | 100.00% | 83,340 |
| SANDI | 0.25 mm | 999 | De, Din, Rsoma, fextra, fneurite, fsoma | 86.90% | 735 |

No duplicate MRI indices or non-finite table values were found. Fourteen R2*
rows differ from the documented `floor(x + 0.5)` nearest-index rule at displayed
half-integer ties. This is compatible with round-to-even or hidden source
precision, but the producer should clarify it before exact indexing.

The mapped centres span approximately 0.87-2.81 mm, 0.73-2.71 mm, and
2.86-5.57 mm on the three claimed 650 nm axes. This is further evidence that
the coordinates belong to a stitched/cropped frame rather than either original
tomogram by itself.

## Required inputs

Obtain the following from the collaborator before producing biological results:

1. The exact stitched XPCT reference used to calculate the tables, plus its
   shape, axis names, direction matrix, origin, and voxel spacing.
2. The stitched cell and vessel segmentations in exactly that frame.
3. The transform and crop/overlap record used to stitch tomo0022 and tomo0029.
4. Every source and reference NIfTI with intact qform and sform.
5. The complete ordered affine/warp chain and inverse transforms listed in
   `registration_requirements.json`.
6. The registered mask, cell-centre, and coverage control NIfTIs named in the
   collaborator legend.
7. Several manually identifiable landmarks in XPCT and MRI space for an
   independent target-registration-error measurement.

## Registration workflow

1. Preserve each image in its native index space and retain the physical-space
   affine. Record array order separately from anatomical orientation.
2. Rebuild the stitched XPCT reference and segmentation masks from the two
   tomograms. Use nearest-neighbour interpolation for labels and masks.
3. Reproduce the full MRI-to-XPCT transform chain in physical coordinates.
   ANTs point transforms have the opposite mapping convention from image
   resampling, so transform order and inversion flags must be recorded.
4. Compare table contrast values to their source NIfTI at `(mri_i,mri_j,mri_k)`.
   The identity hypothesis must outperform every axis permutation and flip.
5. Inspect raw/registered overlays in axial, coronal, and sagittal views. Include
   asymmetric landmarks; a symmetric cylinder alone cannot reveal every flip.
6. Measure landmark target registration error in millimetres and forward/inverse
   round-trip error. Do not accept the registration based only on visual fit.
7. Map the eight corners or a dense sample grid of each MRI voxel through the
   transform, not only its centre. Rasterize/sample this footprint in XPCT space.
8. Recompute XPCT coverage and compare it with both the table and supplied
   coverage NIfTI before extracting targets.

Multi-stage registration through ex-vivo/block-face intermediate images and a
combination of affine and nonlinear transforms is established practice for
histology-to-MRI alignment. Landmark error and orthogonal overlay review remain
necessary because an optimizer score alone does not detect every biologically
important mismatch.

Relevant references:

- [ANTs point-transform direction and ordering](https://antsx.github.io/ANTsR/reference/antsApplyTransformsToPoints.html)
- [MRI-histology registration using intermediate ex-vivo/block-face data](https://pmc.ncbi.nlm.nih.gov/articles/PMC2575850/)
- [Quantitative nonlinear histology-MRI registration and landmark validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5497585/)
- [Ex-vivo MRI/light-microscopy registration accuracy](https://pmc.ncbi.nlm.nih.gov/articles/PMC3100355/)
- [Explicit flip testing in a histology-registration protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC6604680/)

## Segmentation targets

For each MRI voxel footprint, initially derive a small, interpretable target set:

- cell volume fraction (`f_soma_seg`);
- cell count density per mm3, using a documented centre-in-voxel assignment;
- median and interquartile range of equivalent cell radius in micrometres;
- vessel volume fraction;
- vessel centreline length density in mm/mm3;
- optional vessel-radius summaries after a validated diameter method exists.

Equivalent cell radius should be computed from each instance volume using
`r = (3 V / (4 pi))^(1/3)`. Do not infer radius from a semantic binary mask or
from instance fragments at inference-block boundaries. Density denominators
must use the valid XPCT-covered physical volume, and sensitivity analyses should
be reported for coverage thresholds such as 25%, 50%, and 90%.

Analyse each MRI grid separately. Rows from 0.05, 0.10, 0.25, and 0.30 mm grids
are not independent replicates and should not simply be concatenated.

## Statistical plan

Start with descriptive Spearman correlations, scatterplots, and uncertainty
intervals. Account for spatial autocorrelation and report sample counts after
coverage filtering. Correct families of univariate tests for multiple testing.

Elastic Net is a reasonable first predictive model because MRI contrasts are
correlated and the model remains inspectable. It is a baseline, not proof that
MRI resolves individual neurons. Compare it with a training-set median baseline
using R2, MAE, RMSE, and Spearman correlation on held-out data.

The split hierarchy is critical:

1. Best final test: hold out an entire biological specimen before feature
   selection, model selection, and preprocessing.
2. If multiple XPCT blocks are available: group by block/specimen.
3. With Hippo002 alone: use spatial blocks in nested grouped CV only as a proof
   of concept, and repeat the analysis with larger blocks. This estimates
   spatial interpolation within Hippo002, not generalization to another
   brain/specimen. A later implementation should add a physical exclusion
   buffer around each test region once the physical MRI grids are available.

All rows from an overlapping or neighbouring spatial region must remain in one
fold. Standardization and Elastic Net tuning belong inside the inner CV loop.
Do not randomly split MRI voxels; that would leak strong local spatial
correlation into the test set. See the [Elastic Net paper](https://doi.org/10.1111/j.1467-9868.2005.00503.x),
[nested-CV bias analysis](https://doi.org/10.1186/1471-2105-7-91), and
[spatial cross-validation guidance](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf).

## Commands

Run the table audit from the repository root:

```bash
python register/audit_hippo002_registration.py \
  --tables-dir register/Hippo002_table_correlation \
  --outdir register/audit_output
```

Audit the direct-to-tomogram tables:

```bash
python register/audit_hippo002_tomogram_tables.py \
  --tables-dir register/Hippo002_tables_in_tomos \
  --original-tables-dir register/Hippo002_table_correlation \
  --outdir register/audit_output
```

After receiving a source NIfTI, verify its indexing and orientation. Replace the
example paths and contrast column with the matching files:

```bash
python register/validate_table_against_nifti.py \
  --table register/Hippo002_table_correlation/Hippo002_table_r2s_portugal_0p05mm.csv \
  --column R2s \
  --nifti /path/to/source_r2s.nii.gz \
  --output register/audit_output/r2s_nifti_validation.json
```

Only after exact MRI-footprint aggregation has produced a joined table, run the
model baseline. The target name is illustrative:

```bash
python register/run_spatial_elastic_net.py \
  --input register/derived/Hippo002_dki_with_segmentation_targets.csv \
  --predictors FA,MD,AD,AK,MK,MSD,MSK,RD,RK,S0,b0 \
  --target cell_count_density_mm3 \
  --coverage-min 25 \
  --spatial-block-voxels 4 \
  --output-dir register/results/Hippo002_dki_cell_density
```

The current scripts intentionally stop short of fabricating the missing spatial
mapping. Once the listed inputs arrive, the next implementation step is exact
voxel-footprint sampling and segmentation-target extraction.
