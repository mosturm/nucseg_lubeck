#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure inference + evaluation script for a dual-model nucleus Cellpose pipeline.

Compatibility goals:
- mirror segment_nuc.py preprocessing and inference behavior
- same NRRD label loading logic
- same mask_id selection logic, now separately for model 1 and model 2
- same ensure_zyx behavior
- same binary voxel metrics (pred > 0 vs gt > 0)
- no held-out quadrant logic
- no checkpoint selection by training loss logic beyond explicit checkpoint or models/best_model

Outputs per sample:
    image_zyx.tif

    gt_mask3d_model1.tif
    pred_mask3d_model1.tif
    tp_mask3d_model1.tif
    fn_mask3d_model1.tif
    fp_mask3d_model1.tif
    quality_model1_3d.html

    gt_mask3d_model2.tif
    pred_mask3d_model2.tif
    tp_mask3d_model2.tif
    fn_mask3d_model2.tif
    fp_mask3d_model2.tif
    quality_model2_3d.html

    pred_vs_gt.html

    slice_pngs/
        raw_zXXX.png
        model1_gt_instances_zXXX.png
        model1_pred_instances_zXXX.png
        model1_quality_zXXX.png
        model1_tp_zXXX.png
        model1_fn_zXXX.png
        model1_fp_zXXX.png
        model2_gt_instances_zXXX.png
        model2_pred_instances_zXXX.png
        model2_quality_zXXX.png
        model2_tp_zXXX.png
        model2_fn_zXXX.png
        model2_fp_zXXX.png

    metrics.json

Global outputs:
    metrics_per_sample.csv
    metrics_summary.csv
    run_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import colorsys
import numpy as np
import tifffile as tiff
import nrrd
from PIL import Image

from cellpose import models, core

try:
    from skimage.measure import marching_cubes, label as sk_label
except Exception as e:
    raise ImportError(
        "This script requires scikit-image. Install with: pip install scikit-image"
    ) from e


# ======================================================================================
# Utilities copied/adapted for compatibility with segment_nuc.py
# ======================================================================================

def ensure_zyx(arr: np.ndarray, channel_axis: int | None, z_axis: int | None, name: str = "array") -> np.ndarray:
    """
    Ensure array is (Z, Y, X). Mirrors segment_nuc.py logic.
    """
    a = np.asarray(arr)

    if channel_axis is not None and channel_axis < a.ndim:
        a = np.take(a, indices=0, axis=channel_axis)
        a = np.array(a)

    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D after channel handling; got shape {a.shape}")

    if z_axis is not None:
        if not (0 <= z_axis <= 2):
            raise ValueError("z_axis must be 0/1/2 or None")
        if z_axis != 0:
            a = np.moveaxis(a, z_axis, 0)
        return a

    sizes = a.shape
    z_guess = int(np.argmin(sizes))
    if sizes[z_guess] * 2 <= min(sizes[(z_guess + 1) % 3], sizes[(z_guess + 2) % 3]):
        if z_guess != 0:
            a = np.moveaxis(a, z_guess, 0)

    return a


def iter_img_mask_pairs(root: str | Path, label_extension: str) -> Iterable[Tuple[Path, Path, str]]:
    """
    Yield (img_path, mask_path, sample_id) for every *_img.tif with matching *_label<label_extension>.
    Same naming convention as segment_nuc.py.
    """
    root = Path(root)
    for img_path in sorted(root.glob("*_img.tif")):
        mask_path = img_path.with_name(img_path.name.replace("_img.tif", "_label" + label_extension))
        if mask_path.exists():
            sample_id = img_path.stem.replace("_img", "")
            yield img_path, mask_path, sample_id
        else:
            print(f"[WARN] Missing mask for {img_path.name}: expected {mask_path.name}")


def normalize_to_uint8(arr2d: np.ndarray) -> np.ndarray:
    a = arr2d.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


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

    pred_fg_voxels = int(pred_bin.sum())
    gt_fg_voxels = int(gt_bin.sum())
    pred_gt_ratio = float(pred_fg_voxels / gt_fg_voxels) if gt_fg_voxels > 0 else float("inf")

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
        "pred_fg_voxels": pred_fg_voxels,
        "gt_fg_voxels": gt_fg_voxels,
        "pred_gt_ratio": pred_gt_ratio,
    }


def resolve_model_path(model_arg: str | Path) -> Path:
    """
    Accept either:
    - a checkpoint file directly
    - a run folder that contains models/best_model
    - a models folder containing best_model
    """
    p = Path(model_arg)

    if p.is_file():
        return p

    candidate1 = p / "models" / "best_model"
    if candidate1.exists():
        return candidate1

    candidate2 = p / "best_model"
    if candidate2.exists():
        return candidate2

    raise FileNotFoundError(
        f"Could not resolve model from '{p}'. "
        f"Expected either a checkpoint file, '{p}/models/best_model', or '{p}/best_model'."
    )


# ======================================================================================
# Visualization helpers
# ======================================================================================

def _label_to_rgb(inst_2d: np.ndarray) -> np.ndarray:
    h, w = inst_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.unique(inst_2d)
    labels = labels[labels > 0]
    for k in labels:
        hue = (int(k) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        rgb[inst_2d == k] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


def binary_to_instances_2d(bin2d: np.ndarray) -> np.ndarray:
    return sk_label(bin2d.astype(bool), connectivity=2).astype(np.int32)


def save_single_model_slice_pngs(
    raw3d: np.ndarray,
    pred_labels_3d: np.ndarray,
    gt_bin_3d: np.ndarray,
    tp_3d: np.ndarray,
    fn_3d: np.ndarray,
    fp_3d: np.ndarray,
    outdir: Path,
    prefix: str,
) -> None:
    png_dir = outdir / "slice_pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    for z in range(raw3d.shape[0]):
        raw2d = normalize_to_uint8(raw3d[z])
        gt_bin_2d = gt_bin_3d[z] > 0
        tp2d = tp_3d[z] > 0
        fn2d = fn_3d[z] > 0
        fp2d = fp_3d[z] > 0

        pred_inst_2d = pred_labels_3d[z].astype(np.int32)
        gt_inst_2d = binary_to_instances_2d(gt_bin_2d)

        raw_path = png_dir / f"raw_z{z:03d}.png"
        if not raw_path.exists():
            Image.fromarray(raw2d).save(raw_path)

        Image.fromarray(_label_to_rgb(gt_inst_2d)).save(png_dir / f"{prefix}_gt_instances_z{z:03d}.png")
        Image.fromarray(_label_to_rgb(pred_inst_2d)).save(png_dir / f"{prefix}_pred_instances_z{z:03d}.png")

        Image.fromarray((tp2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"{prefix}_tp_z{z:03d}.png")
        Image.fromarray((fn2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"{prefix}_fn_z{z:03d}.png")
        Image.fromarray((fp2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"{prefix}_fp_z{z:03d}.png")

        rgb = np.stack([raw2d, raw2d, raw2d], axis=-1)

        # green = TP, red = FN, blue = FP
        rgb[tp2d] = [0, 255, 0]
        rgb[fn2d] = [255, 0, 0]
        rgb[fp2d] = [0, 0, 255]

        Image.fromarray(rgb).save(png_dir / f"{prefix}_quality_z{z:03d}.png")


def _make_mesh_from_binary(bin_vol: np.ndarray, anisotropy: float) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not np.any(bin_vol):
        return None, None

    verts, faces, _, _ = marching_cubes(
        bin_vol.astype(np.uint8),
        level=0.5,
        spacing=(anisotropy, 1.0, 1.0),
        allow_degenerate=False,
    )
    return verts, faces


def save_quality_3d_html(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, out_html: Path, anisotropy: float) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[WARN] plotly not installed -> skipping 3D HTML export")
        return

    fig = go.Figure()

    for arr, name, color in [
        (tp, "TP", "green"),
        (fn, "FN", "red"),
        (fp, "FP", "blue"),
    ]:
        verts, faces = _make_mesh_from_binary(arr, anisotropy)
        if verts is None:
            continue

        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
        i, j, k = faces[:, 2], faces[:, 1], faces[:, 0]

        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                name=name,
                color=color,
                opacity=0.55,
            )
        )

    fig.update_layout(
        title="3D Quality View (TP green, FN red, FP blue)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
    )
    fig.write_html(str(out_html))


def save_combined_pred_vs_gt_3d_html(
    gt1: np.ndarray,
    pred1: np.ndarray,
    gt2: np.ndarray,
    pred2: np.ndarray,
    out_html: Path,
    anisotropy: float,
) -> None:
    """
    Combined 3D HTML view with these display colors:
      - model 1 GT   : green
      - model 1 pred : blue
      - model 2 GT   : red
      - model 2 pred : violet

    Overwrite rule:
      any voxel occupied by model 2 (GT or prediction) is removed from model 1 display.
      This implements "model 2 overwrites model 1 on intersection".
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[WARN] plotly not installed -> skipping 3D HTML export")
        return

    fig = go.Figure()

    model2_occ = np.logical_or(gt2, pred2)
    gt1_disp = np.logical_and(gt1, ~model2_occ)
    pred1_disp = np.logical_and(pred1, ~model2_occ)
    gt2_disp = gt2
    pred2_disp = pred2

    for arr, name, color, opacity in [
        (gt1_disp, "Model 1 ground truth", "green", 0.35),
        (pred1_disp, "Model 1 prediction", "blue", 0.35),
        (gt2_disp, "Model 2 ground truth", "red", 0.35),
        (pred2_disp, "Model 2 prediction", "yellow", 0.35),
    ]:
        verts, faces = _make_mesh_from_binary(arr, anisotropy)
        if verts is None:
            continue

        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
        i, j, k = faces[:, 2], faces[:, 1], faces[:, 0]

        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                name=name,
                color=color,
                opacity=opacity,
            )
        )

    fig.update_layout(
        title="Prediction vs Ground Truth (model 2 overwrites model 1 in overlaps)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
    )
    fig.write_html(str(out_html))


# ======================================================================================
# Main inference
# ======================================================================================

def infer_one_model(
    model: models.CellposeModel,
    img: np.ndarray,
    gt_bin: np.ndarray,
    sample_id: str,
    model_tag: str,
    mask_id: int,
    cellprob_threshold: float,
    flow_threshold: float,
    infer_3d: bool,
    anisotropy: float,
) -> dict:
    pred_labels, flows, styles = model.eval(
        x=img,
        channels=[0, 0],
        do_3D=infer_3d,
        z_axis=0,
        channel_axis=None,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )

    pred_labels = np.asarray(pred_labels).astype(np.int32)
    pred_bin = pred_labels > 0

    tp = np.logical_and(pred_bin, gt_bin)
    fn = np.logical_and(~pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, ~gt_bin)

    metrics = binary_volume_metrics(pred_bin, gt_bin)
    metrics.update(
        {
            "mask_id": int(mask_id),
            "cellprob_threshold": float(cellprob_threshold),
            "flow_threshold": float(flow_threshold),
            "shape_z": int(img.shape[0]),
            "shape_y": int(img.shape[1]),
            "shape_x": int(img.shape[2]),
        }
    )

    print(
        f"[INFO] {sample_id} | {model_tag} | mask_id={mask_id} "
        f"Dice={metrics['dice']:.4f} "
        f"IoU={metrics['iou']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"Pred/GT={metrics['pred_gt_ratio']:.4f}"
    )

    return {
        "pred_labels": pred_labels,
        "pred_bin": pred_bin,
        "gt_bin": gt_bin,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "metrics": metrics,
    }


def save_model_outputs(sample_out: Path, raw_img: np.ndarray, result: dict, model_tag: str, anisotropy: float) -> None:
    tiff.imwrite(sample_out / f"gt_mask3d_{model_tag}.tif", result["gt_bin"].astype(np.uint8))
    tiff.imwrite(sample_out / f"pred_mask3d_{model_tag}.tif", result["pred_labels"].astype(np.int32))
    tiff.imwrite(sample_out / f"tp_mask3d_{model_tag}.tif", result["tp"].astype(np.uint8))
    tiff.imwrite(sample_out / f"fn_mask3d_{model_tag}.tif", result["fn"].astype(np.uint8))
    tiff.imwrite(sample_out / f"fp_mask3d_{model_tag}.tif", result["fp"].astype(np.uint8))

    save_single_model_slice_pngs(
        raw3d=raw_img,
        pred_labels_3d=result["pred_labels"],
        gt_bin_3d=result["gt_bin"],
        tp_3d=result["tp"],
        fn_3d=result["fn"],
        fp_3d=result["fp"],
        outdir=sample_out,
        prefix=model_tag,
    )

    save_quality_3d_html(
        result["tp"],
        result["fn"],
        result["fp"],
        sample_out / f"quality_{model_tag}_3d.html",
        anisotropy=anisotropy,
    )


def flatten_sample_metrics(sample_id: str, metrics1: dict, metrics2: dict, shape: tuple[int, int, int]) -> dict:
    row = {
        "sample_id": sample_id,
        "shape_z": int(shape[0]),
        "shape_y": int(shape[1]),
        "shape_x": int(shape[2]),
    }

    for prefix, metrics in [("model1", metrics1), ("model2", metrics2)]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v

    return row


def run_one_sample(
    model1: models.CellposeModel,
    model2: models.CellposeModel,
    img_path: Path,
    mask_path: Path,
    sample_id: str,
    out_root: Path,
    args: argparse.Namespace,
) -> dict:
    print(f"\n[INFO] Processing {sample_id}")
    print(f"       image: {img_path.name}")
    print(f"       mask : {mask_path.name}")

    img = tiff.imread(str(img_path))

    # Mirrors segment_nuc.py:
    # msk = np.permute_dims(nrrd.read(str(mask_path))[0], axes=(1,0,2))
    msk_raw = np.transpose(nrrd.read(str(mask_path))[0], axes=(1, 0, 2))

    img = ensure_zyx(img, args.channel_axis, args.z_axis, "image")
    msk_raw = ensure_zyx(msk_raw, args.channel_axis, args.z_axis, "mask")

    if img.shape != msk_raw.shape:
        raise ValueError(f"[{sample_id}] shape mismatch: image {img.shape} vs mask {msk_raw.shape}")

    gt_bin_1 = np.where(msk_raw == args.mask_id1, 1, 0).astype(np.uint8)
    gt_bin_2 = np.where(msk_raw == args.mask_id2, 1, 0).astype(np.uint8)

    sample_out = out_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Volume shape (Z,Y,X): {img.shape}")

    result1 = infer_one_model(
        model=model1,
        img=img,
        gt_bin=gt_bin_1 > 0,
        sample_id=sample_id,
        model_tag="model1",
        mask_id=args.mask_id1,
        cellprob_threshold=args.cellprob_threshold1,
        flow_threshold=args.flow_threshold,
        infer_3d=args.infer_3d,
        anisotropy=args.anisotropy,
    )

    result2 = infer_one_model(
        model=model2,
        img=img,
        gt_bin=gt_bin_2 > 0,
        sample_id=sample_id,
        model_tag="model2",
        mask_id=args.mask_id2,
        cellprob_threshold=args.cellprob_threshold2,
        flow_threshold=args.flow_threshold,
        infer_3d=args.infer_3d,
        anisotropy=args.anisotropy,
    )

    tiff.imwrite(sample_out / "image_zyx.tif", img)

    save_model_outputs(sample_out, img, result1, "model1", args.anisotropy)
    save_model_outputs(sample_out, img, result2, "model2", args.anisotropy)

    save_combined_pred_vs_gt_3d_html(
        gt1=result1["gt_bin"],
        pred1=result1["pred_bin"],
        gt2=result2["gt_bin"],
        pred2=result2["pred_bin"],
        out_html=sample_out / "pred_vs_gt.html",
        anisotropy=args.anisotropy,
    )

    metrics_json = {
        "sample_id": sample_id,
        "shape_z": int(img.shape[0]),
        "shape_y": int(img.shape[1]),
        "shape_x": int(img.shape[2]),
        "model1": result1["metrics"],
        "model2": result2["metrics"],
        "visualization": {
            "pred_vs_gt_html": "pred_vs_gt.html",
            "overwrite_rule": "Any voxel occupied by model 2 GT or model 2 prediction hides model 1 voxels in the combined 3D visualization.",
            "colors": {
                "model1_ground_truth": "green",
                "model1_prediction": "blue",
                "model2_ground_truth": "red",
                "model2_prediction": "yellow",
            },
        },
    }

    with open(sample_out / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    return flatten_sample_metrics(sample_id, result1["metrics"], result2["metrics"], img.shape)


def write_global_outputs(rows: list[dict], out_root: Path, run_meta: dict) -> None:
    run_meta_path = out_root / "run_meta.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[OK] Wrote {run_meta_path}")

    per_sample_csv = out_root / "metrics_per_sample.csv"

    if not rows:
        print("[WARN] No valid samples processed. Writing empty per-sample CSV and skipping summary.")
        with open(per_sample_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "status"])
            writer.writerow(["", "no_valid_samples"])
        return

    preferred_fields = [
        "sample_id",
        "shape_z",
        "shape_y",
        "shape_x",

        "model1_mask_id",
        "model1_cellprob_threshold",
        "model1_flow_threshold",
        "model1_dice",
        "model1_iou",
        "model1_precision",
        "model1_recall",
        "model1_specificity",
        "model1_pred_gt_ratio",
        "model1_tp",
        "model1_fp",
        "model1_fn",
        "model1_tn",
        "model1_pred_fg_voxels",
        "model1_gt_fg_voxels",

        "model2_mask_id",
        "model2_cellprob_threshold",
        "model2_flow_threshold",
        "model2_dice",
        "model2_iou",
        "model2_precision",
        "model2_recall",
        "model2_specificity",
        "model2_pred_gt_ratio",
        "model2_tp",
        "model2_fp",
        "model2_fn",
        "model2_tn",
        "model2_pred_fg_voxels",
        "model2_gt_fg_voxels",
    ]
    all_fields = list(rows[0].keys())
    fieldnames = [f for f in preferred_fields if f in all_fields] + [f for f in all_fields if f not in preferred_fields]

    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {per_sample_csv}")

    metric_suffixes = [
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "pred_gt_ratio",
        "pred_fg_voxels",
        "gt_fg_voxels",
    ]
    metric_names = [f"model1_{m}" for m in metric_suffixes] + [f"model2_{m}" for m in metric_suffixes]

    summary_csv = out_root / "metrics_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "median", "std", "min", "max", "n"])
        for name in metric_names:
            vals = np.array([row[name] for row in rows], dtype=float)
            writer.writerow([
                name,
                float(np.mean(vals)),
                float(np.median(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                float(np.min(vals)),
                float(np.max(vals)),
                int(len(vals)),
            ])

    print(f"[OK] Wrote {summary_csv}")

    m1_dice = np.array([row["model1_dice"] for row in rows], dtype=float)
    m1_iou = np.array([row["model1_iou"] for row in rows], dtype=float)
    m1_precision = np.array([row["model1_precision"] for row in rows], dtype=float)
    m1_recall = np.array([row["model1_recall"] for row in rows], dtype=float)

    m2_dice = np.array([row["model2_dice"] for row in rows], dtype=float)
    m2_iou = np.array([row["model2_iou"] for row in rows], dtype=float)
    m2_precision = np.array([row["model2_precision"] for row in rows], dtype=float)
    m2_recall = np.array([row["model2_recall"] for row in rows], dtype=float)

    print("\n=== Global summary ===")
    print(f"n samples          : {len(rows)}")
    print(f"Model 1 mean Dice  : {m1_dice.mean():.4f}")
    print(f"Model 1 mean IoU   : {m1_iou.mean():.4f}")
    print(f"Model 1 mean Prec. : {m1_precision.mean():.4f}")
    print(f"Model 1 mean Recall: {m1_recall.mean():.4f}")
    print(f"Model 2 mean Dice  : {m2_dice.mean():.4f}")
    print(f"Model 2 mean IoU   : {m2_iou.mean():.4f}")
    print(f"Model 2 mean Prec. : {m2_precision.mean():.4f}")
    print(f"Model 2 mean Recall: {m2_recall.mean():.4f}")


def main(args: argparse.Namespace) -> None:
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path1 = resolve_model_path(args.model1)
    model_path2 = resolve_model_path(args.model2)
    print(f"[INFO] Using model 1: {model_path1}")
    print(f"[INFO] Using model 2: {model_path2}")

    pairs = list(iter_img_mask_pairs(args.test_data_dir, args.label_extension))
    print(f"[INFO] Found {len(pairs)} image/mask pair(s) in {args.test_data_dir}")

    use_gpu = core.use_gpu()
    print(f"[INFO] GPU available: {use_gpu}")

    model1 = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(model_path1),
    )
    model2 = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(model_path2),
    )

    rows: list[dict] = []
    failed_samples: list[dict] = []

    for img_path, mask_path, sample_id in pairs:
        try:
            metrics = run_one_sample(model1, model2, img_path, mask_path, sample_id, out_root, args)
            rows.append(metrics)
        except Exception as e:
            print(f"[ERROR] Failed on sample '{sample_id}': {e}")
            failed_samples.append({
                "sample_id": sample_id,
                "image": str(img_path),
                "mask": str(mask_path),
                "error": str(e),
            })

    run_meta = {
        "model1": str(model_path1),
        "model2": str(model_path2),
        "test_data_dir": str(Path(args.test_data_dir).resolve()),
        "outdir": str(out_root.resolve()),
        "label_extension": args.label_extension,
        "mask_id1": int(args.mask_id1),
        "mask_id2": int(args.mask_id2),
        "channel_axis": args.channel_axis,
        "z_axis": args.z_axis,
        "infer_3d": bool(args.infer_3d),
        "anisotropy": float(args.anisotropy),
        "cellprob_threshold1": float(args.cellprob_threshold1),
        "cellprob_threshold2": float(args.cellprob_threshold2),
        "flow_threshold": float(args.flow_threshold),
        "n_pairs_found": int(len(pairs)),
        "n_success": int(len(rows)),
        "n_failed": int(len(failed_samples)),
        "failed_samples": failed_samples,
        "notes": [
            "Metrics are computed separately for each model on binary voxel masks: pred_labels > 0 versus gt(mask_id_i) > 0.",
            "GT is binarized after selecting mask_id1 or mask_id2 from the same NRRD label volume.",
            "Ground-truth instance coloring in slice PNGs is only for visualization.",
            "pred_vs_gt.html is a combined 3D view: model 1 GT=green, model 1 pred=blue, model 2 GT=red, model 2 pred=violet.",
            "In pred_vs_gt.html, any voxel occupied by model 2 GT or prediction hides model 1 voxels.",
            "This script mirrors segment_nuc.py preprocessing and Cellpose inference behavior."
        ],
    }

    write_global_outputs(rows, out_root, run_meta)

    if len(pairs) == 0:
        print(
            "\n[WARN] No pairs were found.\n"
            "Expected files like:\n"
            "  something_img.tif\n"
            "  something_label.seg.nrrd"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure dual-model inference for nucleus segmentation")

    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Checkpoint file or run folder containing models/best_model for model 1",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Checkpoint file or run folder containing models/best_model for model 2",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Folder containing *_img.tif and matching *_label<ext> files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="inference_eval",
        help="Output directory",
    )

    parser.add_argument(
        "--label_extension",
        type=str,
        default=".seg.nrrd",
        help="Label file extension, e.g. .seg.nrrd",
    )
    parser.add_argument(
        "--mask_id1",
        type=int,
        required=True,
        help="Segmentation label ID to keep as foreground for model 1",
    )
    parser.add_argument(
        "--mask_id2",
        type=int,
        required=True,
        help="Segmentation label ID to keep as foreground for model 2",
    )

    parser.add_argument(
        "--channel_axis",
        type=int,
        default=None,
        help="Channel axis if present, else None",
    )
    parser.add_argument(
        "--z_axis",
        type=int,
        default=None,
        help="Known z-axis (0/1/2), else None to auto-guess",
    )

    parser.add_argument(
        "--infer_3d",
        action="store_true",
        help="Run 3D inference",
    )
    parser.add_argument(
        "--anisotropy",
        type=float,
        default=1.0,
        help="Z spacing / XY spacing",
    )
    parser.add_argument(
        "--cellprob_threshold1",
        type=float,
        required=True,
        help="Cell probability threshold for model 1",
    )
    parser.add_argument(
        "--cellprob_threshold2",
        type=float,
        required=True,
        help="Cell probability threshold for model 2",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.2,
        help="Flow threshold used for both models",
    )

    args = parser.parse_args()
    main(args)