#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cellpose training/eval pipeline for 3D volumes sliced to 2D.
Fixes:
- Robust binary/instance detection for 8-bit masks (0/255, 0/1)
- Visual verification PNGs: maskBinary.png + maskInstances.png per slice
- Sensible default training/eval knobs (can be tweaked below)

Assumptions:
- Input image & mask are single-channel TIF stacks with shape (Z, Y, X) or similar.
- Volume XY size is 512x512 (for quadrant split). Adjust SPLIT_SIZE if needed.

"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import tifffile as tiff
from PIL import Image
import colorsys
import argparse
import nrrd
import csv
from cellpose import io, models, core, train

# ---------------------- Utilities: connected components -------------------------------

def _get_cc_label():
    """Return a 2D connected-components function with scipy or scikit-image."""
    try:
        from skimage.measure import label as sk_label

        def cc_label2d(arr, connectivity=2):
            return sk_label(arr, connectivity=connectivity)

        return cc_label2d
    except Exception:
        try:
            from scipy.ndimage import label as sp_label

            def cc_label2d(arr, connectivity=2):
                if connectivity == 1:
                    structure = np.array([[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]], dtype=int)
                else:
                    structure = np.ones((3, 3), dtype=int)
                out, _ = sp_label(arr, structure=structure)
                return out

            return cc_label2d
        except Exception:
            def cc_label2d(_arr, _conn=2):
                raise ImportError(
                    "Need scikit-image or scipy for connected components.\n"
                    "Install one of:\n"
                    "  pip install scikit-image\n"
                    "  pip install scipy"
                )
            return cc_label2d


cc_label2d = _get_cc_label()


# ---------------------- Utilities: orientation / slicing -------------------------------

def ensure_zyx(arr: np.ndarray, CHANNEL_AXIS: int, Z_AXIS: int, name: str = "array") -> np.ndarray:
    """
    Ensure array is (Z, Y, X). If Z_AXIS is provided, move that axis to 0 and
    squeeze any singleton channel axis. Otherwise, auto-guess Z as the smallest
    axis if one dimension is much smaller than the other two.
    """
    a = np.asarray(arr)

    # Drop channel axis if provided
    if CHANNEL_AXIS is not None and CHANNEL_AXIS < a.ndim:
        a = np.take(a, indices=0, axis=CHANNEL_AXIS)  # use channel 0
        a = np.array(a)

    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D after channel handling; got shape {a.shape}")

    if Z_AXIS is not None:
        if not (0 <= Z_AXIS <= 2):
            raise ValueError("Z_AXIS must be 0/1/2 or None")
        if Z_AXIS != 0:
            a = np.moveaxis(a, Z_AXIS, 0)
        return a

    # Auto guess: always use the smallest axis as Z.
    sizes = a.shape
    z_guess = int(np.argmin(sizes))
    if z_guess != 0:
        a = np.moveaxis(a, z_guess, 0)

    return a


def split_quadrants(vol: np.ndarray, SPLIT_SIZE: int) -> Dict[str, np.ndarray]:
    """Return dict of four XY quadrants along the last two dims (Y, X)."""
    Z, Y, X = vol.shape
    if (Y, X) != (SPLIT_SIZE, SPLIT_SIZE):
        raise ValueError(f"Expected Y=X={SPLIT_SIZE}, got {(Y, X)}")
    return {
        "TL": vol[:, 0:SPLIT_SIZE // 2, 0:SPLIT_SIZE // 2],
        "TR": vol[:, 0:SPLIT_SIZE // 2, SPLIT_SIZE // 2:SPLIT_SIZE],
        "BL": vol[:, SPLIT_SIZE // 2:SPLIT_SIZE, 0:SPLIT_SIZE // 2],
        "BR": vol[:, SPLIT_SIZE // 2:SPLIT_SIZE, SPLIT_SIZE // 2:SPLIT_SIZE],
    }


def iter_img_mask_pairs(root=".", label_extension=".seg.nrrd"):
    """
    Yield (img_path, mask_path, sample_id) for every *_img.tif that has a
    matching *_label of type 'label_extension' in 'root'.
    """
    root = Path(root)
    for img_path in sorted(root.glob("*_img.tif")):
        mask_path = img_path.with_name(img_path.name.replace("_img.tif", "_label" + label_extension))
        if mask_path.exists():
            sample_id = img_path.stem.replace("_img", "")  # e.g. 'roi01_id0062_t0018_Rohrle17_2'
            yield img_path, mask_path, sample_id
        else:
            print(f"[WARN] missing mask for {img_path.name} -> skipped")


def normalize_to_uint8(arr2d: np.ndarray) -> np.ndarray:
    """Min-max normalize to 0..255 uint8 for PNG export."""
    a = arr2d.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def label_to_rgba(inst_2d: np.ndarray, alpha: int = int(255 * 0.25)) -> np.ndarray:
    """
    Convert instance labels to RGBA:
      - colored where label>0 with the given alpha (25% opaque = 75% transparent),
      - background fully transparent (alpha=0).
    """
    rgb = _label_to_rgb(inst_2d)  # uses your existing function
    h, w, _ = rgb.shape
    A = np.zeros((h, w), dtype=np.uint8)
    A[inst_2d > 0] = np.uint8(alpha)
    return np.dstack([rgb, A]).astype(np.uint8)


def save_png_overlays(sample_id: str, raw3d: np.ndarray, masks3d: np.ndarray, out_root="png_eval"):
    """
    For each Z: save raw PNG, mask RGBA PNG (transparent background),
    and overlay PNG (raw with 75%-transparent colored mask).
    """
    outdir = Path(out_root) / sample_id
    outdir.mkdir(parents=True, exist_ok=True)

    for z in range(raw3d.shape[0]):
        raw2d = normalize_to_uint8(raw3d[z])
        inst2d = masks3d[z].astype(np.int32)

        # raw as grayscale PNG
        raw_img = Image.fromarray(raw2d)  # "L"
        raw_img.save(outdir / f"raw_z{z:03d}.png")

        # mask-only RGBA (transparent background, semi-transparent instances)
        rgba = Image.fromarray(label_to_rgba(inst2d))
        rgba.save(outdir / f"mask_rgba_z{z:03d}.png")

        # overlay (composited)
        base_rgba = raw_img.convert("RGBA")
        comp = Image.alpha_composite(base_rgba, rgba)
        comp.save(outdir / f"overlay_z{z:03d}.png")


# ---------------------- Binary/instance handling + visualization ----------------------

def is_binary_mask(msk2d: np.ndarray) -> bool:
    """
    Detect typical binary encodings: {0}, {1}, {255}, {0,1}, {0,255}.
    Works for your 8-bit masks that are 0/255.
    """
    u = np.unique(msk2d)
    if u.size == 1:
        return int(u[0]) in (0, 1, 255)
    if u.size == 2:
        return set(map(int, u)) in ({0, 1}, {0, 255})
    return False


def to_instances_2d(msk2d: np.ndarray, CONNECTIVITY: int) -> np.ndarray:
    """Convert 0/1 or 0/255 to connected-component instance labels; else trust labels."""
    if is_binary_mask(msk2d):
        return cc_label2d((msk2d > 0), connectivity=CONNECTIVITY).astype(np.int32)
    return msk2d.astype(np.int32)


def count_instances_2d(msk2d: np.ndarray, CONNECTIVITY: int) -> int:
    """Count instances consistently with to_instances_2d."""
    if is_binary_mask(msk2d):
        lab = cc_label2d((msk2d > 0), connectivity=CONNECTIVITY)
        return int(lab.max())
    else:
        u = np.unique(msk2d)
        return int((u > 0).sum())

def remove_small_instances_2d(inst_2d: np.ndarray, min_area: int = 2) -> np.ndarray:
    """
    Remove labeled instances smaller than min_area pixels and relabel compactly.
    Works on instance masks where background=0 and objects are positive integers.
    """
    inst = np.asarray(inst_2d).astype(np.int32)
    out = np.zeros_like(inst, dtype=np.int32)

    kept_id = 1
    for k in np.unique(inst):
        if k == 0:
            continue
        area = int((inst == k).sum())
        if area >= min_area:
            out[inst == k] = kept_id
            kept_id += 1
    return out


def sanitize_mask_2d(msk2d: np.ndarray, CONNECTIVITY: int, MIN_INSTANCE_PIXELS: int = 2) -> np.ndarray:
    """
    Full 2D mask sanitation pipeline:
    1) convert binary masks to connected-component instances if needed
    2) remove tiny instances
    3) return compact int32 labels
    """
    inst = to_instances_2d(msk2d, CONNECTIVITY)
    inst = remove_small_instances_2d(inst, min_area=MIN_INSTANCE_PIXELS)
    return inst.astype(np.int32)

def _label_to_rgb(inst_2d: np.ndarray) -> np.ndarray:
    """Color each instance >0 with a distinct (deterministic) color."""
    h, w = inst_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(inst_2d)
    labels = labels[labels > 0]
    for k in labels:
        hue = (k * 0.61803398875) % 1.0  # golden-ratio hue stepping
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        rgb[inst_2d == k] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


def save_debug_masks(img_2d: np.ndarray, msk_raw_2d: np.ndarray, inst_2d: np.ndarray, outdir: str | Path, stem: str):
    """
    Save:
      1) the raw image tile as grayscale PNG
      2) binary mask as fed to the code
      3) colored instances after connected components
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Raw image tile
    img8 = normalize_to_uint8(img_2d)
    Image.fromarray(img8).save(outdir / f"{stem}_imgTile.png")

    # Binary view of the raw mask
    bin8 = ((msk_raw_2d > 0).astype(np.uint8) * 255)
    Image.fromarray(bin8).save(outdir / f"{stem}_maskBinary.png")

    # Colored instances
    rgb = _label_to_rgb(inst_2d)
    Image.fromarray(rgb).save(outdir / f"{stem}_maskInstances.png")

def binary_volume_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "pred_fg_voxels": int(pred_bin.sum()),
        "gt_fg_voxels": int(gt_bin.sum()),
    }

def save_best_test_loss_model(
    model_path: str | Path,
    test_losses: np.ndarray,
    n_epochs: int,
    save_every: int = 5,
    ) -> tuple[Path | None, int | None, float | None]:
    """
    Copy the checkpoint with the lowest computed test loss to:
        <models>/best_model

    IMPORTANT
    ---------
    - This assumes train.train_seg(..., save_each=True).
    - Cellpose computes test loss only when:
          iepoch == 5 or iepoch % 10 == 0
    - With save_every=5, every evaluated epoch except the very first one
      has a saved checkpoint file we can copy.
    """
    model_path = Path(model_path)
    best_model_path = model_path.parent / "best_model"

    candidates = []

    for iepoch in range(n_epochs):
        test_loss_was_computed = (iepoch == 5 or iepoch % 10 == 0)
        if not test_loss_was_computed:
            continue

        # Map epoch -> checkpoint filename
        if iepoch == n_epochs - 1:
            ckpt = model_path
        elif iepoch != 0 and iepoch % save_every == 0:
            ckpt = model_path.parent / f"{model_path.name}_epoch_{iepoch:04d}"
        else:
            # no distinct checkpoint exists for this evaluated epoch
            continue

        if ckpt.exists():
            candidates.append((float(test_losses[iepoch]), iepoch + 1, ckpt))

    if not candidates:
        print(
            "[WARN] No checkpoint matching a computed test loss was found. "
            "Could not create models/best_model."
        )
        return None, None, None

    best_loss, best_epoch, best_ckpt = min(candidates, key=lambda x: x[0])

    shutil.copy2(best_ckpt, best_model_path)

    print(f"Best test-loss checkpoint: epoch={best_epoch}, test_loss={best_loss:.6f}")
    print(f"Copied best checkpoint from {best_ckpt} -> {best_model_path}")

    return best_model_path, best_epoch, best_loss
# ---------------------- Exporters -----------------------------------------------------

def export_xy_slices(vol: np.ndarray, msk: np.ndarray, outdir: str | Path, tag: str, CONNECTIVITY: int) -> int:
    """Export all Z slices to 2D files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for z in range(vol.shape[0]):
        img2d = vol[z]
        raw2d = msk[z]
        msk2d = sanitize_mask_2d(raw2d, CONNECTIVITY, MIN_INSTANCE_PIXELS=2)
        stem = f"{tag}_z{z:03d}"

        # training files
        tiff.imwrite(outdir / f"{stem}_img.tif", img2d, photometric="minisblack")
        tiff.imwrite(outdir / f"{stem}_masks.tif", msk2d.astype(np.int32))

        # visual verifiers
        save_debug_masks(img2d, raw2d, msk2d, outdir / "viz", stem)
        kept += 1
    print(f"exported {kept} slices to {outdir}")
    return kept


def export_xy_slices_subset(vol: np.ndarray, msk: np.ndarray, outdir: str | Path, tag: str,
                            z0: int, z1: int, CONNECTIVITY: int) -> int:
    """Export a Z-range [z0, z1) for quick debug."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for z in range(z0, min(z1, vol.shape[0])):
        img2d = vol[z]
        raw2d = msk[z]
        msk2d = sanitize_mask_2d(raw2d, CONNECTIVITY, MIN_INSTANCE_PIXELS=2)
        stem = f"{tag}_z{z:03d}"
        tiff.imwrite(outdir / f"{stem}_img.tif", img2d, photometric="minisblack")
        tiff.imwrite(outdir / f"{stem}_masks.tif", msk2d.astype(np.int32))

        # visual verifiers
        save_debug_masks(img2d, raw2d, msk2d, outdir / "viz", stem)
        kept += 1
    print(f"[mini] exported {kept} slices z={z0}:{z1} to {outdir}")
    return kept


# ---------------------- Main ----------------------------------------------------------

def main(RNG_SEED, DATASET_DIR, TRAIN_DIR, TEST_DIR, DATA_DIR, VAL_DATA_DIR, LABEL_EXTENSION, TEST_QUADRANT, SPLIT_SIZE, CHANNEL_AXIS, Z_AXIS, MINI_DEBUG, MINI_TRAIN_ZS, MINI_TEST_ZS, CONNECTIVITY, MIN_MASKS_TRAIN, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MODEL_NAME, PRETRAINED_MODEL, INFER_3D, ANISOTROPY, CELLPROB_THRESHOLD, FLOW_THRESHOLD, MASK_ID):
    np.random.seed(RNG_SEED)
    here = Path(".").resolve()
    logger = io.logger_setup(cp_path=str((here / ".cellpose").resolve()))

    # Fresh dataset folder
    if Path(DATASET_DIR).exists():
        shutil.rmtree(DATASET_DIR)
    Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)

    print("\nScanning training pairs ...")
    train_pairs = list(iter_img_mask_pairs(DATA_DIR, LABEL_EXTENSION))
    if not train_pairs:
        raise FileNotFoundError(f"No *_img.tif / *_label{LABEL_EXTENSION} pairs found in {DATA_DIR}")

    print("\nScanning validation pairs ...")
    val_pairs = list(iter_img_mask_pairs(VAL_DATA_DIR, LABEL_EXTENSION))
    if not val_pairs:
        raise FileNotFoundError(f"No *_img.tif / *_label{LABEL_EXTENSION} pairs found in {VAL_DATA_DIR}")

    # Fresh dataset folder
    if Path(DATASET_DIR).exists():
        shutil.rmtree(DATASET_DIR)
    Path(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)

    # collect validation ROIs for later 3D eval & PNG overlays
    test_sets = []
    total_train_slices = total_test_slices = 0

    # -------------------- export TRAIN ROIs (full volume, no quadrant split) --------------------
    for img_path, mask_path, sample_id in train_pairs:
        print(f"Loading TRAIN volume for {sample_id} ...")
        vol = tiff.imread(str(img_path))
        msk = np.permute_dims(nrrd.read(str(mask_path))[0], axes=(1, 0, 2))

        print(sample_id, "raw mask unique (first 20):", np.unique(msk)[:20])

        msk = np.where(msk == MASK_ID, 1, 0).astype(msk.dtype)

        print(sample_id, f"after selecting mask_id={MASK_ID} unique:", np.unique(msk))
        print(sample_id, "foreground voxels:", int((msk > 0).sum()))

        vol = ensure_zyx(vol, CHANNEL_AXIS, Z_AXIS, "image")
        msk = ensure_zyx(msk, CHANNEL_AXIS, Z_AXIS, "mask")
        assert vol.shape == msk.shape, f"[{sample_id}] shape mismatch: {vol.shape} vs {msk.shape}"

        fg = int((msk > 0).sum())
        if fg == 0:
            print(f"[WARN] {sample_id}: mask has no foreground -> skipping this TRAIN ROI")
            continue

        print(f"Exporting TRAIN XY slices for {sample_id} ...")
        if MINI_DEBUG:
            z0, z1 = MINI_TRAIN_ZS
            total_train_slices += export_xy_slices_subset(
                vol, msk, TRAIN_DIR, sample_id, z0, z1, CONNECTIVITY
            )
        else:
            total_train_slices += export_xy_slices(
                vol, msk, TRAIN_DIR, sample_id, CONNECTIVITY
            )

    # -------------------- export VAL ROIs (full volume, no quadrant split) --------------------
    for img_path, mask_path, sample_id in val_pairs:
        print(f"Loading VAL volume for {sample_id} ...")
        vol = tiff.imread(str(img_path))
        msk = np.permute_dims(nrrd.read(str(mask_path))[0], axes=(1, 0, 2))

        print(sample_id, "raw mask unique (first 20):", np.unique(msk)[:20])

        msk = np.where(msk == MASK_ID, 1, 0).astype(msk.dtype)

        print(sample_id, f"after selecting mask_id={MASK_ID} unique:", np.unique(msk))
        print(sample_id, "foreground voxels:", int((msk > 0).sum()))

        vol = ensure_zyx(vol, CHANNEL_AXIS, Z_AXIS, "image")
        msk = ensure_zyx(msk, CHANNEL_AXIS, Z_AXIS, "mask")
        assert vol.shape == msk.shape, f"[{sample_id}] shape mismatch: {vol.shape} vs {msk.shape}"

        fg = int((msk > 0).sum())
        if fg == 0:
            print(f"[WARN] {sample_id}: mask has no foreground -> skipping this VAL ROI")
            continue

        print(f"Exporting VAL XY slices for {sample_id} ...")
        if MINI_DEBUG:
            z0t, z1t = MINI_TEST_ZS
            total_test_slices += export_xy_slices_subset(
                vol, msk, TEST_DIR, sample_id, z0t, z1t, CONNECTIVITY
            )
        else:
            total_test_slices += export_xy_slices(
                vol, msk, TEST_DIR, sample_id, CONNECTIVITY
            )

        # keep full validation ROI for 3D eval/overlays later
        test_sets.append((sample_id, vol, msk))

    print(f"Total slices: train={total_train_slices}, test={total_test_slices}")
    # Quick sanity on instance labels
    print("Verify Instance Segmentation:")
    some = sorted(glob.glob(f'{TRAIN_DIR}/*_masks.tif'))
    if not some:
        raise RuntimeError("No training masks exported — check your export settings.")
    a = tiff.imread(some[0])
    print("dtype:", a.dtype, "min:", a.min(), "max:", a.max(), "unique labels:", len(np.unique(a)))

    # Load images/labels with Cellpose helper
    print("\nLoading training/test lists via cellpose.io.load_train_test_data ...")
    images, labels, train_files, test_images, test_labels, test_files = io.load_train_test_data(
        str((here / TRAIN_DIR).resolve()),
        str((here / TEST_DIR).resolve()),
        image_filter="_img", mask_filter="_masks", look_one_level_down=False
    )

    # Re-sanitize loaded masks to protect Cellpose flow generation,
    # especially on held-out test slices where tiny border fragments may remain.
    labels = [remove_small_instances_2d(np.asarray(lb).astype(np.int32), min_area=2) for lb in labels]
    test_labels = [remove_small_instances_2d(np.asarray(lb).astype(np.int32), min_area=2) for lb in test_labels]

    # Optional debug counts
    bad_test = []
    for i, lb in enumerate(test_labels):
        u = np.unique(lb)
        obj_ids = u[u > 0]
        if len(obj_ids) > 0:
            min_obj = min(int((lb == k).sum()) for k in obj_ids)
            if min_obj < 2:
                bad_test.append((i, test_files[i], min_obj))

    print(f"After sanitizing loaded masks: train={len(labels)} test={len(test_labels)}")
    print(f"Test masks still containing objects <2 px: {len(bad_test)}")
    for item in bad_test[:20]:
        print('BAD_TEST', item)

    # Filter training slices to those with >= MIN_MASKS_TRAIN instances
    def _count(msk2d):
        return count_instances_2d(msk2d, CONNECTIVITY)

    tr = [(im, lb, tf) for im, lb, tf in zip(images, labels, train_files) if _count(lb) >= MIN_MASKS_TRAIN]
    if len(tr) == 0:
        raise RuntimeError(
            "After converting to instance masks, no training slices have "
            f">= {MIN_MASKS_TRAIN} objects. Check your labels or lower MIN_MASKS_TRAIN."
        )

    images      = [x[0] for x in tr]
    labels      = [x[1] for x in tr]
    train_files = [x[2] for x in tr]
    print(f"Training samples kept: {len(images)}  |  Test samples kept: {len(test_images)}")

    # --- debug / remove degenerate masks (prevents flow-gen crash) ---
    MIN_FG_PIXELS = 2  # <=1 can crash; use 10+ if you want stricter quality filtering

    bad = []
    for i, lb in enumerate(labels):
        a = np.asarray(lb)
        fg = int((a > 0).sum())
        if fg < MIN_FG_PIXELS:
            bad.append((i, fg, a.shape, train_files[i]))

    print(f"Found {len(bad)} degenerate training masks (fg < {MIN_FG_PIXELS})")
    for item in bad[:20]:
        print("BAD", item)

    # optionally drop them immediately
    if bad:
        bad_idx = set(i for i, *_ in bad)
        keep = [i for i in range(len(labels)) if i not in bad_idx]
        images = [images[i] for i in keep]
        labels = [labels[i] for i in keep]
        train_files = [train_files[i] for i in keep]
        print(f"After dropping degenerate masks: train={len(images)}")

    bad_test_fg = []
    for i, lb in enumerate(test_labels):
        a = np.asarray(lb)
        fg = int((a > 0).sum())
        if fg < MIN_FG_PIXELS:
            bad_test_fg.append((i, fg, a.shape, test_files[i]))

    print(f"Found {len(bad_test_fg)} degenerate test masks (fg < {MIN_FG_PIXELS})")
    for item in bad_test_fg[:20]:
        print("BAD_TEST_FG", item)

    if bad_test_fg:
        bad_idx = set(i for i, *_ in bad_test_fg)
        keep = [i for i in range(len(test_labels)) if i not in bad_idx]
        test_images = [test_images[i] for i in keep]
        test_labels = [test_labels[i] for i in keep]
        test_files = [test_files[i] for i in keep]
        print(f"After dropping degenerate test masks: test={len(test_images)}")
    # --- end block ---

    if len(images) == 0:
        raise RuntimeError("No training slices remain after filtering.")

    if len(test_images) == 0:
        raise RuntimeError("No validation slices remain after filtering.")

    # GPU?
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")

    # Train Cellpose model
    print("Training...")
    if PRETRAINED_MODEL is not None:
        pretrained_path = Path(PRETRAINED_MODEL).resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}"
            )
        print(f"Fine-tuning pretrained model: {pretrained_path}")
        model = models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=str(pretrained_path),
        )
    else:
        print("Training from the default Cellpose model")
        model = models.CellposeModel(gpu=use_gpu)

    #print("Training... (This will take a while: 100 Epochs ~ 1 hour with two tif-pairs)")
    #print("I couldn't get the logger to actual print the epochs, so it will appear frozen, but it trains! (Maybe go for lunch)")
    SAVE_EVERY = 5

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images, train_labels=labels,
        test_data=test_images, test_labels=test_labels,
        weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS, model_name=MODEL_NAME,
        min_train_masks=MIN_MASKS_TRAIN,
        batch_size=BATCH_SIZE,
        save_path=str(here),
        save_every=SAVE_EVERY,
        save_each=True)

    print(f"Model saved to: {model_path}")

    best_model_path, best_epoch, best_test_loss = save_best_test_loss_model(
        model_path=model_path,
        test_losses=np.asarray(test_losses, dtype=float),
        n_epochs=N_EPOCHS,
        save_every=SAVE_EVERY)

    if best_model_path is not None:
        print(f"Best model written to: {best_model_path}")
        model_path_for_eval = str(best_model_path)
    else:
        print("[WARN] Falling back to final model for held-out inference.")
        model_path_for_eval = str(model_path)
    

    loss_csv = Path(DATASET_DIR) / "losses_per_epoch.csv"
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "test_loss", "test_loss_was_computed"])
        for ep in range(N_EPOCHS):
            test_computed = (ep == 5 or ep % 10 == 0)
            w.writerow([
                ep + 1,
                float(train_losses[ep]),
                float(test_losses[ep]) if test_computed else "",
                int(test_computed),
            ])

    print(f"Saved losses to {loss_csv}")

    print("\nRunning 3D inference on validation ROIs for all samples...")

    eval_model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=model_path_for_eval
    )

    heldout_rows = []

    for sample_id, test_img, test_msk in test_sets:
        masks_pred, flows, styles = eval_model.eval(
            x=test_img,
            channels=[0, 0],
            do_3D=INFER_3D,
            z_axis=0,
            channel_axis=None,
            anisotropy=ANISOTROPY,
            cellprob_threshold=CELLPROB_THRESHOLD,
            flow_threshold=FLOW_THRESHOLD
        )

        out_tif = Path(DATASET_DIR) / f"pred_{sample_id}_3d_mask.tif"
        tiff.imwrite(str(out_tif), masks_pred.astype(np.int32))
        print(f"[{sample_id}] saved 3D predicted mask: {out_tif}")

        pred_bin = (masks_pred > 0)
        gt_bin = (test_msk > 0)

        metrics = binary_volume_metrics(pred_bin, gt_bin)

        print(f"[{sample_id}] voxel IoU on validation ROI: {metrics['iou']:.4f}")
        print(f"[{sample_id}] voxel Dice on validation ROI: {metrics['dice']:.4f}")

        heldout_rows.append({
            "sample_id": sample_id,
            "eval_split": "val_data_dir",
            "infer_3d": int(INFER_3D),
            "anisotropy": float(ANISOTROPY),
            "cellprob_threshold": float(CELLPROB_THRESHOLD),
            "flow_threshold": float(FLOW_THRESHOLD),
            **metrics,
        })

        save_png_overlays(sample_id, test_img, masks_pred, out_root="png_eval")
        print(f"[{sample_id}] PNG overlays written to png_eval/{sample_id}/")

    heldout_csv = Path(DATASET_DIR) / "heldout_metrics_per_sample.csv"

    if heldout_rows:
        with open(heldout_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(heldout_rows[0].keys()))
            writer.writeheader()
            writer.writerows(heldout_rows)

        print(f"Saved held-out per-sample metrics to {heldout_csv}")

    summary_csv = Path(DATASET_DIR) / "heldout_metrics_summary.csv"

    metric_names = [
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "pred_fg_voxels",
        "gt_fg_voxels",
    ]

    if heldout_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "median", "std", "min", "max", "n"])

            for name in metric_names:
                vals = np.array([row[name] for row in heldout_rows], dtype=float)

                writer.writerow([
                    name,
                    float(np.mean(vals)),
                    float(np.median(vals)),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    float(np.min(vals)),
                    float(np.max(vals)),
                    int(len(vals)),
                ])

        print(f"Saved held-out summary metrics to {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D cell segmentation model")
    # seed and dataset paths
    parser.add_argument("--rng_seed",               type=int, default=0,           help="Random seed for reproducibility")
    parser.add_argument("--dataset_dir",            type=str, default="dataset",   help="Dataset destination")
    parser.add_argument("--data_dir",               type=str, default="/projects/crunchie/Jan/Daten/Labeling_Hippo_dataset", help="Input 3D stacks")
    parser.add_argument("--val_data_dir",           type=str, required=True, help="Input 3D stacks used only for validation/test_loss and held-out evaluation")
    parser.add_argument("--label_extension",        type=str, default=".seg.nrrd", help="Extension for label files (default .seg.nrrd for Slicer segmentations)")
    parser.add_argument("--mask_id",                type=int, default=1,           help="Segmentation label id to keep as foreground; everything else becomes 0")
    # quadrants
    parser.add_argument("--test_quadrant",          type=str, default="BR",        choices=["TL", "TR", "BL", "BR"], help="Quadrant to hold out for testing")
    parser.add_argument("--split_size",             type=int, default=512,         help="Expected XY size for quadrant split (default 512 for 512x512 images)")
    # geometry
    parser.add_argument("--channel_axis",           type=int, default=None,        help="Channel axis if present (e.g., 0/1/2/3), or None if single-channel")
    parser.add_argument("--z_axis",                 type=int, default=None,        help="Z axis if known (0/1/2), or None to auto-guess")
    # debugging
    parser.add_argument("--mini_debug",             action="store_true",           help="Export only a small subset of slices for quick debugging")
    parser.add_argument("--mini_train_zs",          type=int, nargs=2,             default=(0, 3), help="Z range [z0, z1) for mini debug train export")
    parser.add_argument("--mini_test_zs",           type=int, nargs=2,             default=(5, 8), help="Z range [z0, z1) for mini debug test export")
    # training settings
    parser.add_argument("--connectivity",           type=int,   default=2,         choices=[1, 2], help="Connectivity for instance labeling (1=4-connectivity, 2=8-connectivity)")
    parser.add_argument("--min_masks_train",        type=int,   default=1,         help="Minimum number of instances in a training slice to keep it (default 1)")
    parser.add_argument("--n_epochs",               type=int,   default=120,       help="Number of training epochs")
    parser.add_argument("--learning_rate",          type=float, default=3e-5,      help="Learning rate for training")
    parser.add_argument("--weight_decay",           type=float, default=0.05,       help="Weight decay for training")
    parser.add_argument("--batch_size",             type=int,   default=1,         help="Batch size for training (effective; cellpose uses internal cropping)")
    parser.add_argument("--diameter",               type=float, default=None,      help="Diameter for cellpose (None to let it estimate)")
    parser.add_argument("--model_name",             type=str,   default="my_3d_finetune", help="Name for the trained model")
    parser.add_argument("--pretrained_model",        type=str,   default=None,      help="Checkpoint to fine-tune; omit to start from the default Cellpose model")
    # inference settings
    parser.add_argument("--infer_3d",               action="store_true",           help="Run 3D inference (default is 2D slice-by-slice)") 
    parser.add_argument("--anisotropy",             type=float, default=1.0,       help="Anisotropy factor for 3D inference (Z spacing / XY spacing)")
    parser.add_argument("--cellprob_threshold",     type=float, default=-1.75,        help="Cell probability threshold for 3D inference (lower to get more predictions early on)")
    parser.add_argument("--flow_threshold",         type=float, default=0.2,       help="Flow threshold for 3D inference (lower to get more predictions early on)")
    args = parser.parse_args()

    main(args.rng_seed, args.dataset_dir, f"{args.dataset_dir}/train", f"{args.dataset_dir}/test", args.data_dir, args.val_data_dir, args.label_extension, args.test_quadrant, args.split_size, args.channel_axis, args.z_axis, args.mini_debug, args.mini_train_zs, args.mini_test_zs, args.connectivity, args.min_masks_train, args.n_epochs, args.learning_rate, args.weight_decay, args.batch_size, args.model_name, args.pretrained_model, args.infer_3d, args.anisotropy, args.cellprob_threshold, args.flow_threshold, args.mask_id)
