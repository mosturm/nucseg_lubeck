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

# 2) modern build tools
python -m pip install --upgrade pip setuptools wheel

# 3) install scientific stack (NumPy < 2, versions compatible with Cellpose + PyTorch)
pip install "numpy<2" "scipy>=1.10,<1.13" "matplotlib<3.9" "pandas" "scikit-image" "plotly" "pyarrow"

# 4) install PyTorch (GPU build; adjust CUDA version if needed)
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