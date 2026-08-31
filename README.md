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

---

## Vessel Model: Five-Fold CV to Scale-Specific Inference

The vessel deployment pipeline mirrors the cell workflow but retains the
multi-scale training design. One mixed-scale model first learns from unscaled,
mid, and coarse slabs. Three separate deployment models are then fine-tuned,
calibrated, and used for inference at their matching scale.

### 1. Existing five-fold inputs and outputs

The deterministic vessel dataset is created by `prepare_vessel_cv5.py` under:

```text
data_vessel_only/cv5_seed46/
  fold_<XX>/
    combined/{Trainingsdaten,Valdaten,Testdaten}/
    unscaled/{Trainingsdaten,Valdaten,Testdaten}/
    mid/{Trainingsdaten,Valdaten,Testdaten}/
    coarse/{Trainingsdaten,Valdaten,Testdaten}/
```

It contains 20 accepted binary vessel volumes: 10 unscaled, 5 mid, and 5
coarse. `tomo_reco_id0004_t0007vn` is excluded. Each fold has one outer-test
volume per scale. Consequently, every mid and coarse volume is tested once;
five of the ten unscaled volumes occur in the outer test sets.

#### Scale definitions and TIFF geometry

The scale categories describe the amount of source anatomy represented by one
model voxel. They are not merely different array shapes. The recorded
preprocessing procedure is in `preprocessing_roi_scaling_protocol.txt`. Its
base source stack has Fiji XYZ dimensions `2560 x 2560 x 2156`, corresponding
to script/OME-Zarr ZYX dimensions `2156 x 2560 x 2560`.

The planned ROI transformations were:

| Scale | Source ROI in Fiji XYZ | Scaled ROI in Fiji XYZ | Fiji output/source scale XYZ |
| --- | ---: | ---: | ---: |
| unscaled | `180 x 180 x 130` | `180 x 180 x 130` | `1 x 1 x 1` |
| mid | `768 x 768 x 555` | `180 x 180 x 130` | `0.234375 x 0.234375 x 0.234234` |
| coarse | `2560 x 2560 x 1849` | `180 x 180 x 130` | `0.0703125 x 0.0703125 x 0.070308` |

Inference code uses ZYX and expresses the reciprocal quantity, source voxels
per model voxel:

```text
unscaled ZYX: 1 x 1 x 1
mid ZYX:      555/130 x 768/180 x 768/180
              4.269230769 x 4.266666667 x 4.266666667
coarse ZYX:   1849/130 x 2560/180 x 2560/180
              14.223076923 x 14.222222222 x 14.222222222
```

The Fiji protocol crops first and then uses `Image > Scale`. Raw intensities
use bilinear or equivalent intensity interpolation; segmentation masks must use
nearest-neighbor interpolation. Fractional interpolation must never be applied
to label IDs.

The TIFF XY resolution tags independently preserve the planned relative
scales: mid files contain `0.234375`, while coarse files contain approximately
`0.0703125`. These are useful relative-scale checks, not reliable physical
voxel sizes: the TIFF resolution unit is `NONE`, and the prepared NRRD headers
contain identity space directions.

The actual prepared CV TIFFs are more heterogeneous than the original plan.
Most new mid and coarse annotation volumes are ZYX `72 x 100 x 100`, consistent
with later annotation subcrops from the scaled ROIs. The dataset contains five
mid and five coarse samples rather than the three of each in the original plan.
Unscaled data also include older ZYX `21 x 256 x 256` slabs and other crop
sizes. Training and inference therefore must not infer physical scale from
array dimensions alone; the `scale` assignment in dataset metadata is
authoritative.

The completed initial CV is expected at:

```text
runs/run_vessel_cv5_seed46_<CV_JOB_ID>/
```

`run_vessel_cv5.slurm` trained, in each fold:

1. A `combined_final` precursor for 120 epochs using all three scales,
   learning rate `3e-5`, weight decay `0.05`, and batch size 1.
2. One model per scale for 80 epochs from the final combined checkpoint, using
   learning rate `1e-5`, weight decay `0.05`, and batch size 1.
3. Validation threshold sweeps and untouched scale-specific test inference.

The fixed-recipe and final-model scripts below do not assume cell thresholds.
`derive_vessel_fixed_recipe.py` reads each scale's five `train.log` files and
`selected_thresholds.tsv`, then selects the median completed epoch and median
validation threshold separately for unscaled, mid, and coarse data.

### 2. Fixed performance estimate and deployment calibration

Upload the Python files into `nucseg_lubeck/` and the SLURM files into
`nucseg_lubeck/slurmscripts/`:

```text
derive_vessel_fixed_recipe.py
aggregate_vessel_calibration.py
evaluate_vessel_fixed_recipe_cv.slurm
run_vessel_final_all.slurm
infer_vessels_large_ome_zarr.slurm
```

Submit the fixed evaluation and calibration from `nucseg_lubeck`:

```bash
cd /mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg/repo/nucseg_lubeck
PROJECT_ROOT=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
CV_RUN="${PROJECT_ROOT}/runs/run_vessel_cv5_seed46_15083193"

EVAL_JOB=$(sbatch --parsable \
  --export=ALL,CV_RUN_ROOT="${CV_RUN}" \
  slurmscripts/evaluate_vessel_fixed_recipe_cv.slurm)
echo "Vessel evaluation/calibration job: ${EVAL_JOB}"
```

Phase 1 evaluates one frozen recipe per scale on the original untouched outer
test data. It uses each original fold's common fixed-epoch checkpoint and the
common validation-derived threshold. These are the performance estimates to
report. They are copied into the source CV run as:

```text
fixed_recipe_vessel.json
fixed_recipe_vessel_test_metrics_all_scales.csv
fixed_recipe_vessel_test_metrics_summary_by_scale.csv
fixed_recipe_vessel_cv_report.txt
fixed_recipe_vessel_test_dice_violin_by_scale.png
fixed_recipe_vessel_test_dice_violin_by_scale.pdf
```

Phase 2 produces deployment calibration. In each fold it excludes all three
former outer-test volumes, retrains the mixed-scale precursor on combined
train+validation, and then fine-tunes each scale model on that scale's
train+validation data. The fixed epoch from Phase 1 is used; `best_model` is
not used. Each former scale test volume is scanned from `-6` through `2` in
`0.25` steps.

Thresholds are selected separately for each scale. The selected threshold has
the most volume-balanced pooled foreground ratio among settings no more than
`0.01` mean Dice below that scale's best threshold. Outputs copied into the CV
run are:

```text
deployment_calibration_vessel_volume_balance_per_sample_all_thresholds.csv
deployment_calibration_vessel_volume_balance_threshold_summary_by_scale.csv
deployment_calibration_vessel_volume_balance_best_by_scale.json
deployment_calibration_vessel_volume_balance_report.txt
```

The full retraining and sweep artifacts remain in:

```text
runs/run_vessel_fixed_eval_calibration_<EVAL_JOB>/
```

Calibration on the former test samples is for final deployment settings. It
does not replace the frozen Phase 1 outer-test performance estimate.

### 3. Final models on all 20 vessel volumes

After `EVAL_JOB` succeeds, train the final model bundle:

```bash
FINAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${EVAL_JOB} \
  --export=ALL,CV_SOURCE_RUN="${CV_RUN}" \
  slurmscripts/run_vessel_final_all.slurm)
echo "Final vessel-model job: ${FINAL_JOB}"
```

The job first trains one combined all-scale precursor for 120 epochs on all 20
volumes. It then starts three independent 80-epoch fine-tunings from that same
final precursor: 10 unscaled volumes, 5 mid volumes, and 5 coarse volumes. The
preselected scale-specific epoch is copied to a stable deployment filename.

```text
runs/run_vessel_final_all_seed46_<FINAL_JOB>/
  combined/models/vessel_combined_final_all_120ep
  unscaled/models/vessel_unscaled_deployment_model
  mid/models/vessel_mid_deployment_model
  coarse/models/vessel_coarse_deployment_model
  deployment_config.json
  fixed_recipe_vessel.json
  deployment_calibration_best_by_scale.json
  <scale>/training_data_qc_not_validation/
  COMPLETED
```

The combined model is a shared representation-learning precursor, not the
preferred inference model. `deployment_config.json` is the authoritative map
from scale to model, selected epoch, cell-probability threshold, `min_size`,
and `flow3D_smooth`. The per-scale QC uses training data and is not an
independent performance estimate.

### 4. Scale-specific large-volume inference

Select the scale that matches how the inference volume was prepared. Do not
choose a scale based on which model gives the most visually pleasing output.
The script reads the matching model and threshold from `deployment_config.json`.

```bash
FINAL_RUN="${PROJECT_ROOT}/runs/run_vessel_final_all_seed46_${FINAL_JOB}"

INFER_JOB=$(sbatch --parsable \
  --dependency=afterok:${FINAL_JOB} \
  --export=ALL,FINAL_RUN_ROOT="${FINAL_RUN}",SCALE=coarse,INPUT_ZARR=/path/to/raw.ome.zarr,SAMPLE_ID=sample_name \
  slurmscripts/infer_vessels_large_ome_zarr.slurm)
echo "Vessel inference job: ${INFER_JOB}"
```

`SCALE` is required; valid values are `SCALE=unscaled`, `SCALE=mid`, and
`SCALE=coarse`. Explicit
`MODEL_PATH`, `CELLPROB_THRESHOLD`, `MIN_SIZE`, or `FLOW3D_SMOOTH` environment
variables override the deployment config when an intentional experiment is
needed. Default block settings remain `40,256,256` with halo `5,32,32` and
output chunks `8,256,256`.

The default output is beside the input and includes both scale and threshold:

```text
<sample_id>_vessels_<scale>_cp<threshold>.ome.zarr
```

Run logs and the exact resolved settings are stored under:

```text
runs/run_vessel_inference_<scale>_<sample_id>_<INFER_JOB>/
```

The analytical tubularity postprocessor remains a separate optional step. Its
cross-validated results did not improve every scale, so it is not silently
applied by the base inference job. Preserve the unfiltered segmentation when
evaluating any coarse-specific postprocessing rule.

### 5. Experimental multiscale-union inference

When only an unscaled raw OME-Zarr is available, the experimental multiscale
pipeline can create mid and coarse inputs, run all three deployment models, map
their semantic masks back to the original grid, and calculate their voxel-wise
union. The files are:

```text
vessel_multiscale_ome_zarr.py
submit_vessel_multiscale_inference.sh
slurmscripts/prepare_vessel_multiscale_inference.slurm
slurmscripts/infer_vessel_multiscale_array.slurm
slurmscripts/finalize_vessel_multiscale_inference.slurm
```

This stage requires a completed final-model run from Section 3. The authoritative
input is its `deployment_config.json`, not manually entered model paths or
thresholds. For the current completed run
`run_vessel_final_all_seed46_15305823`, the mapping is:

| Scale | Deployment model | Selected epoch | Cell-probability threshold |
| --- | --- | ---: | ---: |
| unscaled | `unscaled/models/vessel_unscaled_deployment_model` | 41 | `-3.75` |
| mid | `mid/models/vessel_mid_deployment_model` | 31 | `-4.25` |
| coarse | `coarse/models/vessel_coarse_deployment_model` | 31 | `-4.0` |

All three currently use `min_size=15` and `flow3D_smooth=0`. The combined
120-epoch model is only the shared fine-tuning precursor and is not loaded for
inference.

One invocation of `submit_vessel_multiscale_inference.sh` submits the complete
dependency chain:

1. **CPU preparation:** Read the unscaled OME-Zarr array `0`, retain it as the
   unscaled input, generate mid and coarse intensity OME-Zarrs with trilinear
   interpolation, and initialize three binary semantic output stores.
2. **Eight-task GPU array:** Submit `--array=0-7%4`, with one A100 requested per
   task and at most four tasks running simultaneously. Every worker processes
   its disjoint one-eighth of the core blocks for unscaled, then mid, then
   coarse. It loads only the matching deployment model and settings from
   `deployment_config.json`. Models are loaded sequentially, not held in GPU
   memory together.
3. **CPU remapping and union:** Start only after every GPU task succeeds. Keep
   the unscaled mask on its original grid, map the mid and coarse masks back by
   nearest-neighbor center-coordinate sampling, and write the binary operation
   `unscaled OR mid OR coarse` at original resolution.

Core blocks default to ZYX `40 x 256 x 256`, halo to `5 x 32 x 32`, and output
chunks to `8 x 256 x 256`. These dimensions make each worker write distinct
Zarr chunks; workers never write the same chunk. The eight logical workers run
with a concurrency cap of four; actual concurrency remains subject to scheduler
capacity and account limits.

The training-time transformation is recorded in
`preprocessing_roi_scaling_protocol.txt`. The scale factors are based on these
source and final ROI dimensions:

```text
mid:    source 768 x 768 x 555  -> final 180 x 180 x 130
coarse: source 2560 x 2560 x 1849 -> final 180 x 180 x 130
```

In ZYX source-voxels per output voxel, the defaults are therefore:

```text
mid:    555/130 x 768/180 x 768/180
        4.269230769 x 4.266666667 x 4.266666667
coarse: 1849/130 x 2560/180 x 2560/180
        14.223076923 x 14.222222222 x 14.222222222
```

Run the complete pipeline with the recorded scale defaults:

```bash
export PROJECT_ROOT=/mnt/ceph-hdd/projects/scc_uprp_salditt/nucseg
export INPUT_ZARR="${PROJECT_ROOT}/data_inference/id0004_t0015/fullzarr/tomo_reco_id0004_t0015_raw.ome.zarr"
export SAMPLE_ID=tomo_reco_id0004_t0015
export FINAL_RUN_ROOT="${PROJECT_ROOT}/runs/run_vessel_final_all_seed46_15305823"

cd "${PROJECT_ROOT}/repo/nucseg_lubeck"
bash submit_vessel_multiscale_inference.sh
```

The wrapper checks that the raw input, final `COMPLETED` marker, and deployment
configuration exist. It prints the preparation job ID, GPU-array job ID, union
job ID, run directory, and final output. By default the semantic union is:

```text
data_inference/id0004_t0015/fullzarr/
  tomo_reco_id0004_t0015_vessels_multiscale_union.ome.zarr
```

Intermediate and reproducibility outputs are retained under:

```text
runs/run_vessel_multiscale_<sample_id>_<timestamp>/
  scaled_inputs/
    mid.ome.zarr
    coarse.ome.zarr
  scale_masks/
    unscaled_semantic.ome.zarr
    mid_semantic.ome.zarr
    coarse_semantic.ome.zarr
  workers/
    worker_<00-07>.log
    worker_<00-07>.DONE
  manifest.json
  submitted_pipeline.txt
  prepare.log
  finalize.log
  union_summary.json
  PREPARED
  COMPLETED
```

The three scale masks are intentionally preserved so they can be compared and
combined differently without rerunning Cellpose. If any preparation or array
task fails, Slurm's `afterok` dependencies prevent the union job from running.

The TIFF XY resolution tags independently agree: `0.234375 = 180/768` for mid
and approximately `0.0703125 = 180/2560` for coarse. The protocol resolves the
previously missing Z factors as `130/555` and `130/1849` respectively.

The protocol describes planned `180 x 180 x 130` scaled ROIs, whereas the
prepared training TIFFs currently used by CV are mostly `100 x 100 x 72` and
there are five rather than three samples at each downscaled level. Their
preserved XY scale tags still match the protocol, which is consistent with
smaller annotation subcrops being taken from the scaled ROIs, but that later
subcropping step is not documented in the protocol.

The implementation uses trilinear center-coordinate intensity interpolation as
the 3D equivalent of the recorded Fiji bilinear intensity scaling. It does not
add Gaussian prefiltering because none is documented. Semantic masks are
returned with nearest-neighbor sampling and combined as `unscaled OR mid OR
coarse`. All individual scale masks are preserved. Because a union can increase
recall while also accumulating false positives, its performance still needs
validation against labeled slabs; it does not inherit the individual models'
CV estimates automatically. The workflow creates semantic `uint8` masks only;
it does not merge instances or apply the optional analytical tubularity filter.

## File And Workflow Inventory

The table below covers every source, configuration, and submission file shown
in the repository listings above. It is ordered first by workflow and then by
the order in which a user or parent script normally activates each file.
`__pycache__/` is not included because it is an automatically generated Python
bytecode-cache directory, not a project file. Entries marked **alternative** or
**optional** are not required by the main cell or multiscale-vessel deployment
path.

| Name | Task | Associated workflow |
| --- | --- | --- |
| `README.md` | Main operational documentation: data conventions, CV methodology, deployment recipes, output naming, and cluster commands. | **Project reference (read first).** Documents every workflow; it is not executed. |
| `setup.txt` | Records the Python/Cellpose package setup used to create the project environment. | **Environment 1.** Use when creating or reproducing `envs/cellpose-clean`. |
| `warmstartgwdg.txt` | Short GWDG login, project-path, module, virtual-environment, and GPU-session notes. | **Environment 2.** Use after login and before running Python or submitting jobs. |
| `preprocessing_roi_scaling_protocol.txt` | Records how the unscaled ROIs were transformed into the mid and coarse image scales, including source/final dimensions and Fiji interpolation settings. | **Shared data provenance.** Consult before vessel dataset preparation or multiscale inference; its factors drive downscaling and inverse mapping. |
| `segment_nuc.py` | Core Cellpose 3D training/fine-tuning program. Reads TIFF/`.seg.nrrd` pairs, normalizes axis order with Z as the smallest axis, selects `mask_id`, supports pretrained checkpoints, and saves epoch/final/best models. | **Shared training engine.** Called by both CV scripts, both fixed-recipe evaluation scripts, both final-training scripts, and the older vessel fine-tuning script. |
| `sweep_thresholds_segment_nuc.py` | Runs repeated validation inference over Cellpose postprocessing settings and writes per-sample and aggregate threshold metrics; supports cell-probability, `min_size`, and `flow3D_smooth` sweeps. | **Shared calibration engine.** Called after training by cell/vessel CV, fixed-recipe calibration, one-time cell calibration, and older scale evaluation. |
| `test_infer.py` | Runs one selected model/parameter recipe on a labeled test directory and writes predictions, previews, per-sample metrics, and metric summaries. | **Shared evaluation engine.** Called only after model and inference parameters have been fixed for the relevant evaluation stage. |
| `convert_tif_mask_to_ome_zarr.py` | Converts a large TIFF and optional mask/scaling metadata into chunked OME-Zarr, preserving the raw volume as array `0`. | **Large-volume input preparation.** Run before either cell or vessel OME-Zarr inference when the source is TIFF. |
| `infer_ome_zarr.py` | Performs halo-aware, blockwise Cellpose inference on one OME-Zarr scale and writes a same-grid label/semantic Zarr plus progress and timing information. | **Shared single-GPU inference engine.** Called by `infer_cells_large_ome_zarr.slurm` and the alternative `infer_vessels_large_ome_zarr.slurm`. It is not the engine used by the eight-worker multiscale vessel array. |
| `prepare_cell_only_dataset.py` | Combines the old and new annotation sources, keeps available cell foreground, remaps it to mask ID `1`, pairs TIFFs with labels, and creates the original seed-46 train/validation/test dataset. | **Cell 1: base dataset preparation.** Run only when rebuilding `data_cell_only/` from the source annotation folders. |
| `prepare_cell_cv5.py` | Builds the deterministic five-fold cell dataset from the accepted cell samples, keeps image/label orientation aligned, and excludes the rejected `tomo_reco_id0004_t0007vn` sample. | **Cell 2: CV split preparation.** Produces the fold directories consumed by `run_cell_cv5.slurm`. |
| `slurmscripts/run_cell_cv5.slurm` | Trains one cell model per outer fold, performs coarse/fine validation threshold selection, evaluates each untouched test fold, and launches aggregation. | **Cell 3: unbiased 5-fold CV.** Primary cell-performance-estimation job; calls `segment_nuc.py`, `sweep_thresholds_segment_nuc.py`, `test_infer.py`, then `aggregate_cell_cv.py`. |
| `aggregate_cell_cv.py` | Combines fold-level cell test CSVs, calculates pooled/summary statistics, writes the CV report, and creates Dice violin plots. | **Cell 3a: CV aggregation.** Called near the end of `run_cell_cv5.slurm` and reused for fixed-recipe reporting. |
| `slurmscripts/evaluate_cell_fixed_recipe_cv.slurm` | Evaluates the frozen epoch/threshold recipe on the untouched outer tests, then separately retrains each fold on train+validation and sweeps the former test fold for deployment-only threshold calibration. | **Cell 4: fixed-recipe performance plus deployment calibration.** Run after the original five-fold CV has completed. |
| `aggregate_cell_calibration.py` | Pools threshold-level cell calibration results across folds, compares best-Dice and volume-balanced choices, applies the Dice-tolerance rule, and writes the selected deployment setting/report. | **Cell 4a: deployment calibration aggregation.** Called by `evaluate_cell_fixed_recipe_cv.slurm` and the one-time calibration job. |
| `slurmscripts/calibrate_cell_volume_once.slurm` | Repeats a one-off threshold scan/volume-balance analysis against existing held-out fold outputs without repeating the full primary CV workflow. | **Cell 4b: optional calibration utility.** Use for diagnostic or revised threshold grids; it does not replace the frozen CV performance estimate. |
| `slurmscripts/run_cell_final_all.slurm` | Collects all accepted labeled cell slabs, reproduces the selected training schedule, selects the fixed deployment epoch, copies the CV/calibration evidence, performs training-data QC, and writes the final model/configuration. | **Cell 5: final deployment training.** Run after `evaluate_cell_fixed_recipe_cv.slurm`; output is `runs/run_cell_final_all_seed46_<jobid>/`. |
| `slurmscripts/infer_cells_large_ome_zarr.slurm` | Resolves the final cell model/settings and runs blockwise Cellpose inference on a full raw OME-Zarr, with reproducibility metadata and foreground-volume summary. | **Cell 6: large-volume inference.** Calls `infer_ome_zarr.py` after TIFF-to-Zarr conversion and final-model training. |
| `merge_block_instances_ome_zarr.py` | Reconciles instance IDs that were split at known inference-core boundaries and writes a merged instance OME-Zarr plus merge audit/summary. | **Cell 7: optional instance correction.** Needed for cell-size/radius statistics; not required for binary `f_soma`. |
| `slurmscripts/merge_cell_block_instances.slurm` | CPU SLURM wrapper that supplies the block geometry and paths to `merge_block_instances_ome_zarr.py`. | **Cell 7a: optional merge submission.** Run after cell inference and before publication-quality instance-radius analysis. |
| `visualize_semantic_ome_zarr.py` | Creates center-section raw/overlay views and soma statistics, including equivalent-radius and `f_soma` distributions and 250-micrometer binned maps. | **Cell 8: analysis and visualization.** Reads raw and inferred/merged OME-Zarr outputs. |
| `slurmscripts/visualize_cells_center_sections.slurm` | CPU SLURM wrapper for the semantic OME-Zarr visualization/statistics script with project-specific paths and physical dimensions. | **Cell 8a: visualization submission.** Run after inference, and after instance merging when radius results are required. |
| `visualize_cell_vessel_overlay_ome_zarr.py` | Writes aligned raw, red-cell, and red-cell/blue-vessel center sections, then creates a self-contained interactive 3D HTML from only the mid/coarse masks. It compares `mid OR coarse` against `postprocessed mid OR postprocessed coarse`; each scale is filtered on its native grid before coarse is mapped to the mid display grid. | **Cell/vessel 9: combined 2D and 3D visualization.** Run after segmentation, retained multiscale intermediates, and analytical calibration exist. |
| `slurmscripts/visualize_cell_vessel_overlay.slurm` | CPU wrapper that resolves the latest multiscale run, the scale-specific saved mid rule, and the coarse stage-2 rule, then runs the 2D renderer and interactive before/after comparison. | **Cell/vessel 9a: visualization submission.** Produces aligned XZ/YZ panels and toggleable `vessels_mid_coarse_3d.html`. |
| `slurmscripts/run_vessel_cv5.slurm` | In every outer fold, trains the shared combined-scale model, fine-tunes separate unscaled/mid/coarse models, calibrates each scale on its validation data, tests on untouched scale-specific data, and aggregates results. | **Vessel 1: unbiased multiscale 5-fold CV.** Calls the shared training/sweep/test engines and `aggregate_vessel_cv.py`. |
| `aggregate_vessel_cv.py` | Combines all fold/scale vessel test metrics, produces scale-wise summaries and reports, and creates the scale-grouped Dice violin plots. | **Vessel 1a: CV aggregation.** Called at the end of `run_vessel_cv5.slurm` and reused for fixed-recipe reporting. |
| `derive_vessel_fixed_recipe.py` | Reads the completed vessel CV artifacts and derives one fixed completed-epoch count and initial cell-probability threshold for each scale. | **Vessel 2: recipe derivation.** Called first by `evaluate_vessel_fixed_recipe_cv.slurm`; its output is `fixed_vessel_recipe.json`. |
| `slurmscripts/evaluate_vessel_fixed_recipe_cv.slurm` | Tests the fixed per-scale epoch/threshold recipe on untouched outer folds, then retrains on train+validation and calibrates deployment thresholds on the former test folds without changing the reported CV estimate. | **Vessel 3: fixed-recipe evaluation and deployment calibration.** Run after vessel CV and before final all-data training. |
| `aggregate_vessel_calibration.py` | Pools all per-file/per-threshold vessel calibration measurements by scale and selects best-Dice and volume-balanced thresholds under the configured Dice tolerance. | **Vessel 3a: calibration aggregation.** Called by the fixed-recipe vessel evaluation job. |
| `slurmscripts/run_vessel_final_all.slurm` | Trains a 120-epoch combined all-scale precursor on all 20 slabs, fine-tunes all-data unscaled/mid/coarse deployment models for their selected epochs, performs QC, and writes `deployment_config.json`. | **Vessel 4: final deployment training.** Run with an `afterok` dependency on Vessel 3; output is `runs/run_vessel_final_all_seed46_<jobid>/`. |
| `submit_vessel_multiscale_inference.sh` | User-facing orchestrator that validates input/final-model paths and submits the CPU preparation, eight-worker GPU array (maximum four concurrent tasks), and dependent CPU union jobs with the required shared environment variables. | **Vessel 5: production multiscale inference entry point.** This is the one command normally submitted for an unscaled full OME-Zarr. |
| `slurmscripts/prepare_vessel_multiscale_inference.slurm` | CPU stage that initializes the run, downsamples the raw unscaled Zarr to the documented mid/coarse grids, creates output stores, and writes the manifest/`PREPARED` marker. | **Vessel 5a: automatic preparation.** First job submitted by `submit_vessel_multiscale_inference.sh`. |
| `slurmscripts/infer_vessel_multiscale_array.slurm` | Eight-task GPU-array wrapper; each task processes a disjoint subset of blocks at all three scales with the matching model and parameters from `deployment_config.json`. | **Vessel 5b: parallel scale inference.** Starts after successful preparation and invokes the `infer-worker` command in `vessel_multiscale_ome_zarr.py`. |
| `slurmscripts/finalize_vessel_multiscale_inference.slurm` | CPU finalization wrapper that checks all worker markers, maps mid/coarse semantic masks back to the original grid, unions them with the unscaled mask, and writes completion metadata. | **Vessel 5c: remapping and union.** Starts only after every array task succeeds. |
| `vessel_multiscale_ome_zarr.py` | Implementation shared by all three multiscale stages: trilinear intensity downscaling, chunk-safe distributed Cellpose inference, nearest-neighbor mask upscaling, and semantic union. | **Vessel 5 engine.** Called with `prepare`, `infer-worker`, and `finalize` by the three SLURM stages above. |
| `slurmscripts/infer_vessels_large_ome_zarr.slurm` | Runs one selected vessel scale/model on an already matching-resolution OME-Zarr and writes a foreground summary. It does not create other scales or union their masks. | **Vessel alternative 5.** Single-scale diagnostic/production path via `infer_ome_zarr.py`; not part of the automatic multiscale chain. |
| `vessel_analytical_postprocess.py` | Calibrates/evaluates fixed-window local-PCA tubularity filtering from out-of-fold vessel predictions, including the coarse-specific second-stage window/threshold search and recall-loss constraints. | **Vessel optional 6: analytical postprocessing research.** Uses completed CV predictions; currently not integrated into the multiscale production union. |
| `slurmscripts/run_vessel_analytical_postprocess.slurm` | CPU wrapper that finds or accepts a completed vessel CV run and executes the analytical calibration/evaluation into a separate reproducibility run. | **Vessel optional 6a: postprocessor submission.** Run after Vessel 1 when investigating false-positive suppression. |
| `slurmscripts/finetune_vessel_scales.slurm` | Fine-tunes unscaled, mid, and coarse models from an already trained combined vessel checkpoint in separate run directories. | **Vessel legacy/manual path.** Superseded for full CV/final deployment by `run_vessel_cv5.slurm` and `run_vessel_final_all.slurm`. |
| `slurmscripts/evaluate_vessel_scales.slurm` | Performs threshold sweeps and labeled test inference for models produced by the manual scale fine-tuning job. | **Vessel legacy/manual evaluation.** Companion to `finetune_vessel_scales.slurm`; not used by the current final-model pipeline. |
| `infer3D_and_plot.py` | Standalone 3D inference and plotting utility for quickly inspecting a model, image, and segmentation outside the structured CV/Zarr pipelines. | **Ad hoc diagnostics/legacy visualization.** Run manually when a small-volume visual check is useful. |
| `infer_combined.py` | Runs the older dual-model cell-plus-vessel inference path with separate models, mask IDs, and Cellpose thresholds. | **Legacy combined inference.** Predates the final all-data cell model and automatic three-scale vessel-union workflow. |
