# Full MRI-XPCT registration workflow

## Preflight conclusion

The supplied archive contains 27 MRI tables, 9 pooled acquisition/protocol grids,
3 specimens (P2, P5, P7), and coordinates for 12 tomograms. Local auditing found
no duplicate MRI coordinates, missing MRI indices, missing predictors, unusable
declared owners, or incomplete eight-corner transforms.

| Specimen | XPCT sample ID | Tomograms | Count |
| --- | --- | --- | ---: |
| P2 | 0014 | 0001, 0008, 0015, 0022, 0029, 0036 | 6 |
| P5 | 0004 | 0022, 0029 | 2 |
| P7 | 0011 | 0001, 0002, 0003, 0004 | 4 |

These are 12 XPCT tomograms from three biological specimens, not 12 independent
specimens. Pooled validation therefore leaves out an entire specimen, whereas
the specimen-specific figures use spatially blocked validation within that
specimen.

Hard requirement: all 12 ordinary cell masks must exist on the cluster. The
submission pipeline creates/reuses reconciled `*_instances_merged.ome.zarr`
masks before radius extraction. Missing vessel masks are allowed and produce NA
vessel values for affected MRI voxels.

Portugal DKI is extracted and receives separate and inclusive exploratory
heatmaps, but stays excluded from the primary pooled-results table because the
collaborator flagged its registration as suspicious. Those rows and plots are
explicitly labeled as exploratory. Hamburg in-vivo DKI is too sparse at 80% coverage for a useful
pooled analysis. Only
`r23=sum(r^2)/sum(r^3)` is analyzed; reciprocal `r3/r2` is not generated.

## Files to upload

Upload these files to `.../nucseg/repo/nucseg_lubeck/register/`:

- `full_registration_config.py`
- `audit_full_registration_inputs.py`
- `extract_full_mri_segmentation_features.py`
- `analyze_full_mri_segmentation_correlations.py`
- `plot_full_association_heatmaps.py`
- `plot_full_r23_theory.py`
- `plot_full_generalization_diagnostics.py`
- `run_full_posthoc_plots.slurm`
- `merge_full_cell_instances_array.slurm`
- `extract_full_features_array.slurm`
- `run_full_mri_segmentation.slurm`
- `submit_full_registration.sh`
- `register_full_input.zip`

Also upload the current `merge_block_instances_ome_zarr.py` to the repository
root `.../nucseg/repo/nucseg_lubeck/`.

### Login endpoint

Use `glogin9.hpc.gwdg.de` explicitly for the current upload. At the time this
workflow was prepared, `glogin-gpu.hpc.gwdg.de` resolved to
`134.76.43.150` (`glogin10`), which timed out during the SSH handshake.
`glogin9.hpc.gwdg.de` (`134.76.43.149`) accepted the same account and SSH
key. A timeout through `glogin-gpu` or `glogin10` therefore does not by itself
indicate a broken key or missing cluster permission.

From local Windows PowerShell:

```powershell
$Key = "$env:USERPROFILE\.ssh\id_ed25519_gwdg"
$HpcUser = "u30847"
$HpcHost = "glogin9.hpc.gwdg.de"
$LocalRepo = "C:\Users\Moritz_Sturm\Desktop\Code\Lubeck\nucseg_lubeck"
$RemoteRepo = "/projects/scc/UGOE/UPFB/UPRP/scc_uprp_salditt/dir.project/nucseg/repo/nucseg_lubeck"
$RemoteRegister = "${RemoteRepo}/register"

# Confirm that this endpoint and key work, and create the destination.
ssh -i "$Key" "${HpcUser}@${HpcHost}" `
  "mkdir -p '${RemoteRegister}' && hostname"

# Upload the small workflow files separately from the input archive.
scp -i "$Key" `
  "$LocalRepo\register\full_registration_config.py" `
  "$LocalRepo\register\audit_full_registration_inputs.py" `
  "$LocalRepo\register\extract_full_mri_segmentation_features.py" `
  "$LocalRepo\register\analyze_full_mri_segmentation_correlations.py" `
  "$LocalRepo\register\plot_full_association_heatmaps.py" `
  "$LocalRepo\register\plot_full_r23_theory.py" `
  "$LocalRepo\register\plot_full_generalization_diagnostics.py" `
  "$LocalRepo\register\run_full_posthoc_plots.slurm" `
  "$LocalRepo\register\merge_full_cell_instances_array.slurm" `
  "$LocalRepo\register\extract_full_features_array.slurm" `
  "$LocalRepo\register\run_full_mri_segmentation.slurm" `
  "$LocalRepo\register\submit_full_registration.sh" `
  "${HpcUser}@${HpcHost}:${RemoteRegister}/"

scp -i "$Key" `
  "$LocalRepo\merge_block_instances_ome_zarr.py" `
  "${HpcUser}@${HpcHost}:${RemoteRepo}/"

# Upload the 205 MB archive on its own and keep the SSH session alive.
scp -i "$Key" `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=20 `
  "$LocalRepo\register\register_full_input.zip" `
  "${HpcUser}@${HpcHost}:${RemoteRegister}/"

if ($LASTEXITCODE -ne 0) {
  throw "The archive upload failed. Retry through glogin9; do not replace the SSH key."
}

# The local and remote archive sizes must both be 214889093 bytes.
Get-Item "$LocalRepo\register\register_full_input.zip" |
  Select-Object FullName, Length

ssh -i "$Key" "${HpcUser}@${HpcHost}" `
  "stat -c '%n %s bytes' '${RemoteRegister}/register_full_input.zip'"
```

If an archive transfer is interrupted, it can be resumed interactively:

```powershell
sftp -i "$Key" "${HpcUser}@${HpcHost}"
```

Then run inside SFTP:

```text
cd /projects/scc/UGOE/UPFB/UPRP/scc_uprp_salditt/dir.project/nucseg/repo/nucseg_lubeck/register
lcd C:/Users/Moritz_Sturm/Desktop/Code/Lubeck/nucseg_lubeck/register
reput register_full_input.zip
quit
```

Do not switch back to `glogin-gpu.hpc.gwdg.de` while it resolves to
`134.76.43.150`. Once GWDG restores that node, the generic alias may be used
again.

## Extract input on the cluster

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"

cd "${REGISTER_ROOT}"
test ! -e register_full_input || {
  echo "register_full_input already exists; inspect it instead of overwriting it"
  exit 1
}
python -m zipfile -e register_full_input.zip register_full_input
```

Do not rerun the extraction block if `register_full_input/fuer_Mo` already exists.

## Validate scripts and submit

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
DATA_ROOT="${PROJECT_ROOT}/data_inference/tomos_to_segment/fullzarr"

source "${PROJECT_ROOT}/envs/cellpose-clean/bin/activate"
cd "${REGISTER_ROOT}"

python -m py_compile \
  full_registration_config.py \
  audit_full_registration_inputs.py \
  extract_full_mri_segmentation_features.py \
  analyze_full_mri_segmentation_correlations.py \
  plot_full_association_heatmaps.py \
  plot_full_r23_theory.py \
  plot_full_generalization_diagnostics.py

bash -n \
  merge_full_cell_instances_array.slurm \
  extract_full_features_array.slurm \
  run_full_mri_segmentation.slurm \
  run_full_posthoc_plots.slurm \
  submit_full_registration.sh

# Table audit plus current cluster mask inventory. Missing merged masks are
# expected here because the submitter creates them; missing ordinary masks are not.
python audit_full_registration_inputs.py \
  --input-root "${REGISTER_ROOT}/register_full_input" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${REGISTER_ROOT}/full_preflight_before_submit" \
  --allow-missing-merged-cells

column -s, -t \
  "${REGISTER_ROOT}/full_preflight_before_submit/segmentation_mask_inventory.csv" \
  | less -S

df -h "${DATA_ROOT}"

export PROJECT_ROOT DATA_ROOT
bash submit_full_registration.sh
```

The submitter launches a 12-task cell-instance merge array, a dependent 12-task
exact feature-extraction array (at most four concurrent tasks), and a final
combine/analysis job.

## Monitoring

```bash
RUN_ROOT=$(ls -dt "${PROJECT_ROOT}/runs/run_full_mri_segmentation_"* | head -1)
cat "${RUN_ROOT}/submitted_pipeline.txt"

MERGE_JOB=$(awk -F= '$1=="merge_job"{print $2}' "${RUN_ROOT}/submitted_pipeline.txt")
EXTRACT_JOB=$(awk -F= '$1=="extract_job"{print $2}' "${RUN_ROOT}/submitted_pipeline.txt")
ANALYSIS_JOB=$(awk -F= '$1=="analysis_job"{print $2}' "${RUN_ROOT}/submitted_pipeline.txt")

squeue -j "${MERGE_JOB},${EXTRACT_JOB},${ANALYSIS_JOB}" \
  -o "%.18i %.24j %.10T %.12M %.40R"

grep -hE 'cell chunks|vessel chunks|Completed extraction partial' \
  "${RUN_ROOT}"/extract_*.log | tail -n 30

sacct -j "${MERGE_JOB},${EXTRACT_JOB},${ANALYSIS_JOB}" \
  --format=JobID,JobName%26,State,ExitCode,Elapsed,MaxRSS
```

Expected compute time after scheduling is approximately 12-24 hours, depending
on how many vessel masks are available and Ceph I/O load. Queue time is additional.

## Main outputs

All outputs are below the printed `RUN_ROOT`:

- `preflight/registration_table_audit.csv`
- `preflight/segmentation_mask_inventory.csv`
- `features/feature_manifest.csv`
- `features/tables/*_segmentation_features.csv.gz`
- `features/per_cell_radius_tables/*_cell_radii_by_mri_voxel.csv.gz`
- `analysis_cov_80/raw_voxel_tables/*_raw_voxel_values.csv.gz`
- `analysis_cov_80/all_univariate_correlations_and_grouped_cv.csv`
- `analysis_cov_80/primary_pooled_protocol_correlations_and_loso_cv.csv`
- `analysis_cov_80/r1_vs_r23_theory_fits.csv`
- `analysis_cov_80/experiment_feasibility.csv`
- `analysis_cov_80/heatmaps/*.png`
- `analysis_cov_80/r1_r23_plots/*.png`
- `analysis_cov_80/generalization_diagnostics/*.png`

The heatmap directory contains per-acquisition/protocol figure pairs plus the
cross-protocol primary overviews
`all_primary_associations__spearman.png` and
`all_primary_associations__heldout_r2.png`. Portugal DKI remains in separately
labelled exploratory figures and is not included in these primary overviews.
The explicitly non-primary overviews
`all_associations_including_exploratory__spearman.png` and
`all_associations_including_exploratory__heldout_r2.png` additionally contain
the registration-flagged Portugal DKI `FA`, `MD`, and `MK` rows.
The same directory also contains `P2`, `P5`, and `P7` all-association heatmaps.
Their held-out panels report five-fold spatial-block validation within one
specimen, not leave-one-specimen-out validation.

`generalization_diagnostics/` contains protocol-level validation ladders,
specimen effect-consistency matrices, association-strength-versus-LOSO plots,
and selected specimen-colored scatter plots. These figures make clear whether
an association is consistent within specimens but fails to transfer between
them.

The per-cell tables contain the MRI voxel row and coordinates, specimen,
tomogram, reconciled cell-instance ID, cell volume, centroid, and raw
sphere-equivalent radius. Analysis plots can therefore be reproduced without
rescanning segmentation masks.

## Correcting one production tomogram

Use `CORRECTED_P5_T0022_WORKFLOW.md` and
`submit_corrected_p5_t0022.sh` for the corrected P5/0022 raw volume. That
workflow stages and validates new cell and vessel masks, preserves the old
masks, recomputes only the P5/0022 extraction partial, and rebuilds all tables
and plots from the complete 12-tomogram dataset.

## Post-hoc plots for an existing completed run

The post-hoc job repairs the obsolete extra HH ex-vivo factor from the retained
`*_uncorrected` columns, reruns statistics, and creates the new plots. It does
not reopen masks or repeat feature extraction.

```bash
PROJECT_ROOT="${PROJECT_DIR}/nucseg"
REGISTER_ROOT="${PROJECT_ROOT}/repo/nucseg_lubeck/register"
RUN_ROOT="${PROJECT_ROOT}/runs/run_full_mri_segmentation_20261015_152251_15836"

export PROJECT_ROOT RUN_ROOT
cd "${REGISTER_ROOT}"
POSTHOC_JOB=$(sbatch --parsable --export=ALL run_full_posthoc_plots.slurm)
echo "Submitted ${POSTHOC_JOB}"
squeue -j "${POSTHOC_JOB}" -o "%.18i %.24j %.10T %.12M %.40R"
```

Completion is marked by `${RUN_ROOT}/POSTHOC_PLOTS_COMPLETED`; progress and
errors are written to `posthoc_plots.log` and `full-mri-posthoc-<jobid>.err`.
