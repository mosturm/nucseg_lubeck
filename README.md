# Cellpose-SAM 3D Volume Training Pipeline

This repository provides a small, self-contained pipeline to **train a Cellpose-SAM model** on 3D TIFF volumes and evaluate it on held-out quadrants.

The workflow is:

1. Take 3D image + mask TIFF stacks.
2. Automatically orient them to `(Z, Y, X)` and split into quadrants.
3. Convert **binary masks** (0/1 or 0/255) into **instance labels** via connected components.
4. Export 2D slices for training / testing.
5. Fine-tune a **Cellpose-SAM** model on those slices.
6. Run **3D inference** on held-out quadrants and save:
   - 3D predicted masks,
   - PNG overlays for quick visual QC.

The main script is: **`segment_nuc.py`**.

---

## Quick Start

### 1. Create and activate the environment

```bash
# 1) create virtual environment
python3 -m venv ~/envs/cellpose-clean
source ~/envs/cellpose-clean/bin/activate
# did not work on geri so alternatively
conda create -n cellpose-clean python=3
conda activate cellpose-clean

# 2) modern build tools
python -m pip install --upgrade pip setuptools wheel

# 3) install scientific stack (NumPy < 2, versions compatible with Cellpose + PyTorch) --> scipy created an error for me, so I didn't install it...
pip install "numpy<2" "scipy>=1.10,<1.13" "matplotlib<3.9" "pandas" "scikit-image" "plotly" "pyarrow" "pynrrd"

# 4) install PyTorch (GPU build; adjust CUDA version if needed) --> Url didn't work for me, so I just left that part out
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5) install Cellpose 4 (includes Cellpose-SAM)
pip install "cellpose==4.0.6"
```

> **Note:** Cellpose 4.x uses **Cellpose-SAM (`cpsam`) as the default pretrained model**, so this script fine-tunes SAM without extra configuration.

---

### 2. Upload your data

Only upload TIFF files that follow the naming convention and live in the **repository root** on the remote machine.

For each sample you must provide a pair:

- Image volume: `SAMPLE_img.tif`
- Mask volume:  `SAMPLE_masks.tif`

Example file pairs:

- `roi01_img.tif`  + `roi01_masks.tif`
- `sampleA_img.tif` + `sampleA_masks.tif`

Example `scp` command from your local machine:

```bash
scp -pC /home/mo/Desktop/CellSegLubeck/*.tif     remote@138.2.233.225:/home/ubuntu/nucseg_lubeck/
```

> **Important:**  
> - Only these `*_img.tif` / `*_masks.tif` files need to be in the main repo folder.  
> - All other data (exported slices, models, PNG overlays) are generated automatically and are already ignored by `.gitignore`.

---

### 3. Run the pipeline

From the repository root on the remote machine:

```bash
cd /home/ubuntu/cellseglubeck
source ~/envs/cellpose-clean/bin/activate

python segment_nuc.py
```

The script will:

1. Scan for `*_img.tif` / `*_masks.tif` pairs.
2. Preprocess & export slices into `dataset/train` and `dataset/test`.
3. Train a Cellpose-SAM model.
4. Run 3D inference on the held-out quadrant.
5. Save TIF predictions and PNG overlays.

---

## Input

### Expected input files

- Location: **repository root**.
- Format:
  - **Image**: single-channel 3D TIFF stack, shape roughly `(Z, Y, X)`.
  - **Mask**: same shape as image.
- Naming convention (per sample):

  ```text
  SAMPLE_img.tif
  SAMPLE_masks.tif
  ```

- Mask content:
  - Either **binary** (`0/1` or `0/255`), OR  
  - Already instance-labeled (`0 = background`, `1, 2, … = objects`).

If binary, the script automatically converts them to instance labels using 2D connected components (configurable 4- or 8-connectivity).

### Geometric assumptions

- XY size is assumed to be **512 × 512** by default and each volume is split into four quadrants:
  - TL (top-left), TR (top-right), BL (bottom-left), BR (bottom-right).
- One quadrant (default: **BR**) is held out for testing; the other three are used for training.
- The script tries to auto-detect the Z axis and channel axis, but these can be explicitly set at the top of `segment_nuc.py` (`Z_AXIS`, `CHANNEL_AXIS`, `SPLIT_SIZE`).

---

## Output

When you run `segment_nuc.py`, the following are created:

### 1. 2D training / test dataset

- Folder: `dataset/`
  - `dataset/train/`:
    - 2D slices from the **training quadrants**, one TIFF per slice:
      - `SAMPLE_tile_TL_z000_img.tif`
      - `SAMPLE_tile_TL_z000_masks.tif`
      - etc.
    - For each slice, a pair `*_img.tif` + `*_masks.tif`.
  - `dataset/test/`:
    - 2D slices from the **held-out quadrant** (e.g. BR), same naming scheme.

Each mask slice is guaranteed to be **instance-labeled**:
- Binary masks are converted to instances via connected components.
- Pre-existing instance labels are kept as-is.

Additionally, per-slice visualization masks are written under:

- `dataset/train/viz/`
- `dataset/test/viz/`

These include:
- `*_maskBinary.png` — binary foreground/background view.
- `*_maskInstances.png` — each instance colored uniquely.

---

### 2. Trained model

- Folder: `models/` (created by Cellpose).
- Model file: something like:

  ```text
  models/my_3d_finetune_*.npy
  ```

The exact path is printed at the end of training, e.g.:

```text
Model saved to: models/my_3d_finetune_2025_...
```

This is a **fine-tuned Cellpose-SAM** model.

---

### 3. 3D prediction TIFs

For each sample and held-out quadrant, a 3D predicted mask volume is saved:

- Folder: `dataset/`
- File pattern:

  ```text
  pred_{SAMPLE_ID}_{TEST_QUADRANT}_3d_mask.tif
  ```

Example:

```text
dataset/pred_roi01_BR_3d_mask.tif
```

These are instance-labeled 3D masks (int32), predicted by the trained model on the test quadrant.

For each prediction, the script also prints a **voxel-wise IoU** between:

- Predicted mask > 0, and
- Ground-truth mask > 0

on the held-out 3D quadrant.

---

### 4. PNG overlays (visual QC)

- Folder: `png_eval/`
- For each sample:

  - `png_eval/{SAMPLE_ID}/raw_z000.png` — normalized raw slice.
  - `png_eval/{SAMPLE_ID}/mask_rgba_z000.png` — colored instances with transparent background.
  - `png_eval/{SAMPLE_ID}/overlay_z000.png` — raw image + mask overlay.

These overlays make it easy to qualitatively inspect model performance slice by slice.

---

## What the Script Does (Short Summary)

The main script `segment_nuc.py`:

1. **Preprocessing**
   - Ensures image and mask volumes are oriented as `(Z, Y, X)`.
   - Splits each volume into four XY quadrants (TL/TR/BL/BR).
   - Uses three quadrants for training and holds one out for testing.
   - Converts **binary masks** into **instance masks** using 2D connected components.
   - Exports per-slice TIFFs plus PNG visualizations for debugging.

2. **Training**
   - Loads the exported dataset via `cellpose.io.load_train_test_data`.
   - Filters out slices with too few objects.
   - Fine-tunes a **Cellpose-SAM** model (`CellposeModel` from Cellpose 4.0.6, which uses `cpsam` by default) for `N_EPOCHS`.

3. **3D Inference & Evaluation**
   - Runs 3D Cellpose-SAM inference (`do_3D=True`) on the held-out quadrant for each sample.
   - Saves 3D predicted masks as TIFF.
   - Computes voxel-level IoU for quick numeric feedback.
   - Exports PNG overlays (raw + prediction) for qualitative inspection.

---

## Git Ignore

The repository is configured so that temporary / generated artifacts do **not** clutter git:

- `png_eval/` — PNG overlays.
- `dataset/` — exported train/test slices + 3D predictions.
- `models/` — trained Cellpose models.
- `/*.tif` — raw TIFF volumes in the repo root.

You only need to version-control:

- `segment_nuc.py` and other source files,
- The `README.md`,
- Maybe a config file if you add one.

All raw TIFF data and training outputs stay local and are ignored by git.

---

## Cell Model: Five-Fold CV to Large-Volume Inference

This section records the current cell-segmentation workflow in the required
execution order. Cluster paths below use the current GWDG project mount:

```text
/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
```

The SLURM scripts are expected in:

```text
/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/repo/nucseg_lubeck/slurmscripts
```

### 1. Starting data and five-fold split

The prepared cell dataset contains 13 image/label pairs distributed across:

```text
data_cell_only/Trainingsdaten/
data_cell_only/Valdaten/
data_cell_only/Testdaten/
```

Each pair uses this naming convention:

```text
<sample_id>_img.tif
<sample_id>_label.seg.nrrd
```

`tomo_reco_id0004_t0007vn` is excluded. The resulting CV dataset therefore
contains 12 labeled volumes. All prepared masks are binary, with background 0
and cell foreground 1. Image and label volumes are checked after loading in
ZYX orientation; the NRRD label requires the project-specific XY transpose.

Create the deterministic five-fold dataset with seed 46 from the project root:

```bash
cd /mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
module load gcc/13.2.0
module load python/3.11.9
source /mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/envs/cellpose-clean/bin/activate
python repo/nucseg_lubeck/prepare_cell_cv5.py
```

The script writes:

```text
data_cell_only/cv5_seed46/
  metadata.csv
  test_coverage.csv
  README.txt
  fold_01/combined/{Trainingsdaten,Valdaten,Testdaten}/
  ...
  fold_05/combined/{Trainingsdaten,Valdaten,Testdaten}/
```

The shuffled outer-test group sizes are 3, 3, 2, 2, and 2. Every accepted
volume occurs in `Testdaten` exactly once. For each fold, the next test group
cyclically becomes `Valdaten`; all remaining volumes are `Trainingsdaten`.
`metadata.csv` records every assignment and `test_coverage.csv` records the
unique outer-test coverage.

`prepare_cell_cv5.py` refuses to overwrite an existing `cv5_seed46` directory.
Regeneration therefore requires deliberately moving or removing the old output
first.

### 2. Initial five-fold training

Run the complete initial CV:

```bash
cd /mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/repo/nucseg_lubeck
CV_JOB=$(sbatch --parsable slurmscripts/run_cell_cv5.slurm)
echo "CV job: ${CV_JOB}"
```

Input:

```text
data_cell_only/cv5_seed46/fold_<XX>/combined/
```

For each fold, `run_cell_cv5.slurm`:

1. Trains Cellpose-SAM for 120 epochs with seed `46 + fold`, learning rate
   `3e-5`, weight decay `0.05`, batch size 1, and `mask_id=1`.
2. Selects `models/best_model` by validation loss.
3. Tunes the cell-probability threshold on that fold's validation data using a
   coarse sweep followed by a local fine sweep. `min_size=15` and
   `flow3D_smooth=0` remain fixed.
4. Evaluates the selected fold recipe on the untouched outer-test split.
5. Aggregates all 12 out-of-fold test predictions.

The run directory is:

```text
runs/run_cell_cv5_seed46_<SLURM_JOB_ID>/
```

Important contents are:

```text
fold_<XX>/models/cell_fold_<XX>_final
fold_<XX>/models/cell_fold_<XX>_final_epoch_<zero-based epoch index>
fold_<XX>/models/best_model
fold_<XX>/threshold_sweep_val_coarse/
fold_<XX>/threshold_sweep_val_fine/
fold_<XX>/inference_eval/metrics_per_sample.csv
fold_<XX>/inference_eval/metrics_summary.csv
selected_thresholds.tsv
test_metrics_all.csv
test_metrics_summary.csv
cv_report.txt
test_dice_violin.png
test_dice_violin.pdf
workflow.log
COMPLETED
```

Cellpose checkpoint indices are zero-based. For example,
`cell_fold_01_final_epoch_0050` is the checkpoint after completed epoch 51.

The initial fold-specific test results are useful for inspecting training and
choosing a common recipe, but they are not the final fixed-recipe performance
estimate because each fold originally used a different validation-selected
checkpoint and threshold.

### 3. Fixed-recipe performance estimate and deployment calibration

Across the five initial folds, the median selected checkpoint was completed
epoch 51 and the median validation-selected cell-probability threshold was
`-3.5`. The evaluation job first freezes these settings and applies exactly the
same recipe to every original outer-test split:

```text
checkpoint: epoch 51 (`epoch_0050`)
cellprob_threshold: -3.5
min_size: 15
flow3D_smooth: 0
```

This frozen evaluation is the primary cross-validated performance estimate used
for reporting. Threshold calibration performed later must not replace these
reported test metrics.

Start the combined fixed evaluation and deployment-calibration job after the CV
job has completed:

```bash
PROJECT_ROOT=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
CV_RUN="${PROJECT_ROOT}/runs/run_cell_cv5_seed46_${CV_JOB}"

EVAL_JOB=$(sbatch --parsable \
  --export=ALL,CV_RUN_ROOT="${CV_RUN}" \
  slurmscripts/evaluate_cell_fixed_recipe_cv.slurm)
echo "Evaluation/calibration job: ${EVAL_JOB}"
```

Alternatively, when submitting immediately after the CV job, add:

```bash
--dependency=afterok:${CV_JOB}
```

The complete reproducibility output is written to:

```text
runs/run_cell_fixed_eval_calibration_<SLURM_JOB_ID>/
  fixed_reporting/
  deployment_calibration/
  workflow.log
  COMPLETED
```

Phase 1, `fixed_reporting`, evaluates each original fold's epoch-51 model on its
untouched outer-test split with threshold `-3.5`. Its aggregate files are also
copied into the source CV run with these names:

```text
fixed_recipe_epoch51_cpneg3p5_test_metrics_all.csv
fixed_recipe_epoch51_cpneg3p5_test_metrics_summary.csv
fixed_recipe_epoch51_cpneg3p5_cv_report.txt
fixed_recipe_epoch51_cpneg3p5_test_dice_violin.png
fixed_recipe_epoch51_cpneg3p5_test_dice_violin.pdf
```

Phase 2, `deployment_calibration`, has a different purpose. For each fold it:

1. Combines that fold's `Trainingsdaten` and `Valdaten`.
2. Retrains for the same 120-epoch schedule and uses the fixed epoch-51
   checkpoint, preserving the optimizer settings from CV.
3. Uses the former outer-test split only as held-out calibration data.
4. Evaluates every threshold from `-6` through `2` in steps of `0.25` with
   `min_size=15` and `flow3D_smooth=0`.
5. Pools exactly one held-out prediction for each of the 12 volumes.
6. Finds the best mean-Dice threshold, retains thresholds no more than 0.01
   absolute Dice below that optimum, and selects the retained threshold whose
   pooled predicted/ground-truth foreground-volume ratio is closest to 1.

The calibration fold models and sweeps are stored under:

```text
runs/run_cell_fixed_eval_calibration_<JOB_ID>/deployment_calibration/
  fold_<XX>/models/cell_calibration_fold_<XX>_120ep_epoch_0050
  fold_<XX>/threshold_sweep_volume_balance/
```

The pooled deployment-calibration artifacts are copied to the source CV run:

```text
deployment_calibration_epoch51_volume_balance_per_sample_all_thresholds.csv
deployment_calibration_epoch51_volume_balance_threshold_summary.csv
deployment_calibration_epoch51_volume_balance_best.json
deployment_calibration_epoch51_volume_balance_report.txt
```

For the current dataset, the selected deployment setting is:

```text
cellprob_threshold: -2.25
min_size: 15
flow3D_smooth: 0
checkpoint epoch: 51
```

The threshold `-2.25` is a deployment calibration setting, not an additional
unbiased test result. The fixed epoch-51/`-3.5` outer-test metrics remain the
performance estimate. Because the recipe was summarized from this same small CV
campaign, an entirely independent labeled dataset would still be the strongest
external confirmation of generalization.

### 4. Final model trained on all 12 volumes

After successful fixed evaluation and calibration, train the deployable model
on all 12 accepted labeled volumes:

```bash
FINAL_JOB=$(sbatch --parsable \
  --export=ALL,CV_SOURCE_RUN="${CV_RUN}" \
  slurmscripts/run_cell_final_all.slurm)
echo "Final-model job: ${FINAL_JOB}"
```

When chaining jobs, use:

```bash
--dependency=afterok:${EVAL_JOB}
```

The final run is written to:

```text
runs/run_cell_final_all_seed46_<SLURM_JOB_ID>/
```

The job reassembles all 12 unique volumes, trains for 120 epochs with seed 46,
learning rate `3e-5`, weight decay `0.05`, and batch size 1, and deploys the
preselected epoch-51 checkpoint:

```text
runs/run_cell_final_all_seed46_<JOB_ID>/models/
  cell_final_all_120ep
  cell_final_all_120ep_epoch_0050
```

The deployment model is `cell_final_all_120ep_epoch_0050`. The 120-epoch model
and `best_model` are not used for deployment. The run also contains:

```text
deployment_config.txt
deployment_calibration_best.json
deployment_calibration_report.txt
fixed_recipe_cv_test_metrics_all.csv
fixed_recipe_cv_test_metrics_summary.csv
fixed_recipe_cv_report.txt
training_data_qc_not_validation/
train.log
workflow.log
COMPLETED
```

`training_data_qc_not_validation` is in-sample quality control and must not be
reported as an independent performance estimate.

### 5. Convert a large TIFF volume to OME-Zarr

The conversion script is:

```text
/user/unigoe.uprp-moham/u26421/.project/dir.project/nucseg/repo/nucseg_lubeck/convert_tif_mask_to_ome_zarr.py
```

The historical `/user/.../.project/dir.project` path may no longer resolve on
the cluster. Under the current project mount, use:

```text
/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/repo/nucseg_lubeck/convert_tif_mask_to_ome_zarr.py
```

The raw inference volume was converted on Windows PowerShell with:

```powershell
python .\convert_tif_mask_to_ome_zarr.py `
  --tif "D:\DESY_Downloads\id0004_t0015_roehrle27_5_multiDist\tomo_reco_id0004_t0015.tif" `
  --mask="" `
  --scaling-metadata="" `
  --out "D:\DESY_Downloads\id0004_t0015_roehrle27_5_multiDist\tomo_reco_id0004_t0015_raw.ome.zarr" `
  --z-chunk 8 `
  --zarr-chunks 8,256,256 `
  --compressor-level 5
```

After transfer to GWDG, the current inference input is expected at:

```text
data_inference/id0004_t0015/fullzarr/
  tomo_reco_id0004_t0015_raw.ome.zarr
```

The image array is read from OME-Zarr key `0` in ZYX order.

### 6. Large-volume OME-Zarr inference

The large-volume job uses `infer_cells_large_ome_zarr.slurm`, which invokes
`infer_ome_zarr.py`. By default it automatically chooses the newest completed
final all-data run. For complete reproducibility, pass the intended model
explicitly:

```bash
PROJECT_ROOT=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
FINAL_RUN="${PROJECT_ROOT}/runs/run_cell_final_all_seed46_${FINAL_JOB}"
FINAL_MODEL="${FINAL_RUN}/models/cell_final_all_120ep_epoch_0050"

INFER_JOB=$(sbatch --parsable \
  --export=ALL,MODEL_PATH="${FINAL_MODEL}",CELLPROB_THRESHOLD=-2.25,MIN_SIZE=15,FLOW3D_SMOOTH=0 \
  slurmscripts/infer_cells_large_ome_zarr.slurm)
echo "Inference job: ${INFER_JOB}"
```

When chaining directly after final training, add:

```bash
--dependency=afterok:${FINAL_JOB}
```

Default inference settings are:

```text
input: data_inference/id0004_t0015/fullzarr/tomo_reco_id0004_t0015_raw.ome.zarr
array_key: 0
block_shape: 40,256,256
halo: 5,32,32
output_chunks: 8,256,256
anisotropy: 1.0
cellprob_threshold: -2.25
min_size: 15
flow3D_smooth: 0
skip policy: skip exactly zero input cores only
```

The full segmentation is written beside the input as:

```text
data_inference/id0004_t0015/fullzarr/
  tomo_reco_id0004_t0015_cells_cpneg2p25.ome.zarr
```

Run metadata and logs are written to:

```text
runs/run_cell_inference_tomo_reco_id0004_t0015_<SLURM_JOB_ID>/
  inference_config.txt
  infer_ome_zarr.log
  foreground_volume_summary.json
  python_packages.txt
  submitted_job.slurm
  workflow.log
  COMPLETED
```

`foreground_volume_summary.json` reports the number and fraction of output
voxels with label greater than zero. Conversion to physical volume additionally
requires the correct voxel spacing. Block labels are made unique, but instances
crossing block boundaries are not merged; binary segmented volume is therefore
usable, while cell counts and per-cell morphology require a separate
cross-block instance-merging procedure.
