#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure inference + evaluation script for the nucleus Cellpose pipeline.

Compatibility goals:
- mirror segment_nuc.py preprocessing and inference behavior
- same NRRD label loading logic
- same mask_id selection logic
- same ensure_zyx behavior
- same binary voxel metrics (pred > 0 vs gt > 0)
- no held-out quadrant logic
- no checkpoint selection by training loss logic beyond explicit checkpoint or models/best_model

Outputs per sample:
    image_zyx.tif
    gt_mask3d.tif
    pred_mask3d.tif
    tp_mask3d.tif
    fn_mask3d.tif
    fp_mask3d.tif
    quality_3d.html
    pred_vs_gt_3d.html
    slice_pngs/
        raw_zXXX.png
        quality_zXXX.png
        gt_instances_zXXX.png
        pred_instances_zXXX.png
        tp_zXXX.png
        fn_zXXX.png
        fp_zXXX.png
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
from typing import Dict, Iterable, Tuple

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


def save_slice_pngs(
    raw3d: np.ndarray,
    pred_labels_3d: np.ndarray,
    gt_bin_3d: np.ndarray,
    tp_3d: np.ndarray,
    fn_3d: np.ndarray,
    fp_3d: np.ndarray,
    outdir: Path,
) -> None:
    png_dir = outdir / "slice_pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    for z in range(raw3d.shape[0]):
        raw2d = normalize_to_uint8(raw3d[z])
        pred_bin_2d = pred_labels_3d[z] > 0
        gt_bin_2d = gt_bin_3d[z] > 0
        tp2d = tp_3d[z] > 0
        fn2d = fn_3d[z] > 0
        fp2d = fp_3d[z] > 0

        pred_inst_2d = pred_labels_3d[z].astype(np.int32)
        gt_inst_2d = binary_to_instances_2d(gt_bin_2d)

        Image.fromarray(raw2d).save(png_dir / f"raw_z{z:03d}.png")
        Image.fromarray(_label_to_rgb(gt_inst_2d)).save(png_dir / f"gt_instances_z{z:03d}.png")
        Image.fromarray(_label_to_rgb(pred_inst_2d)).save(png_dir / f"pred_instances_z{z:03d}.png")

        Image.fromarray((tp2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"tp_z{z:03d}.png")
        Image.fromarray((fn2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"fn_z{z:03d}.png")
        Image.fromarray((fp2d.astype(np.uint8) * 255), mode="L").save(png_dir / f"fp_z{z:03d}.png")

        rgb = np.stack([raw2d, raw2d, raw2d], axis=-1)

        # green = TP, red = FN, blue = FP
        rgb[tp2d] = [0, 255, 0]
        rgb[fn2d] = [255, 0, 0]
        rgb[fp2d] = [0, 0, 255]

        Image.fromarray(rgb).save(png_dir / f"quality_z{z:03d}.png")


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


def save_pred_vs_gt_3d_html(pred_bin: np.ndarray, gt_bin: np.ndarray, out_html: Path, anisotropy: float) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[WARN] plotly not installed -> skipping 3D HTML export")
        return

    fig = go.Figure()

    for arr, name, color, opacity in [
        (gt_bin, "Ground truth", "green", 0.35),
        (pred_bin, "Prediction", "blue", 0.35),
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
        title="Prediction vs Ground Truth",
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

def run_one_sample(
    model: models.CellposeModel,
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
    msk = np.transpose(nrrd.read(str(mask_path))[0], axes=(1, 0, 2))

    # same mask-id selection logic
    msk = np.where(msk == args.mask_id, 1, 0).astype(np.uint8)

    img = ensure_zyx(img, args.channel_axis, args.z_axis, "image")
    msk = ensure_zyx(msk, args.channel_axis, args.z_axis, "mask")

    if img.shape != msk.shape:
        raise ValueError(f"[{sample_id}] shape mismatch: image {img.shape} vs mask {msk.shape}")

    sample_out = out_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Volume shape (Z,Y,X): {img.shape}")

    pred_labels, flows, styles = model.eval(
        x=img,
        channels=[0, 0],
        do_3D=args.infer_3d,
        z_axis=0,
        channel_axis=None,
        anisotropy=args.anisotropy,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        min_size=args.min_size,
    )

    pred_labels = np.asarray(pred_labels).astype(np.int32)

    pred_bin = pred_labels > 0
    gt_bin = msk > 0

    tp = np.logical_and(pred_bin, gt_bin)
    fn = np.logical_and(~pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, ~gt_bin)

    metrics = binary_volume_metrics(pred_bin, gt_bin)
    metrics["sample_id"] = sample_id
    metrics["shape_z"] = int(img.shape[0])
    metrics["shape_y"] = int(img.shape[1])
    metrics["shape_x"] = int(img.shape[2])

    print(
        f"[INFO] {sample_id} | Dice={metrics['dice']:.4f} "
        f"IoU={metrics['iou']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"Pred/GT={metrics['pred_gt_ratio']:.4f}"
    )

    # Save per-sample TIFs
    tiff.imwrite(sample_out / "image_zyx.tif", img)
    tiff.imwrite(sample_out / "gt_mask3d.tif", gt_bin.astype(np.uint8))
    tiff.imwrite(sample_out / "pred_mask3d.tif", pred_labels.astype(np.int32))
    tiff.imwrite(sample_out / "tp_mask3d.tif", tp.astype(np.uint8))
    tiff.imwrite(sample_out / "fn_mask3d.tif", fn.astype(np.uint8))
    tiff.imwrite(sample_out / "fp_mask3d.tif", fp.astype(np.uint8))

    # Save PNGs
    save_slice_pngs(
        raw3d=img,
        pred_labels_3d=pred_labels,
        gt_bin_3d=gt_bin,
        tp_3d=tp,
        fn_3d=fn,
        fp_3d=fp,
        outdir=sample_out,
    )

    # Save HTML views
    save_quality_3d_html(tp, fn, fp, sample_out / "quality_3d.html", anisotropy=args.anisotropy)
    save_pred_vs_gt_3d_html(pred_bin, gt_bin, sample_out / "pred_vs_gt_3d.html", anisotropy=args.anisotropy)

    # Save per-sample metrics
    with open(sample_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


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

    # stable field order
    preferred_fields = [
        "sample_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "pred_gt_ratio",
        "tp",
        "fp",
        "fn",
        "tn",
        "pred_fg_voxels",
        "gt_fg_voxels",
        "shape_z",
        "shape_y",
        "shape_x",
    ]
    all_fields = list(rows[0].keys())
    fieldnames = [f for f in preferred_fields if f in all_fields] + [f for f in all_fields if f not in preferred_fields]

    with open(per_sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {per_sample_csv}")

    metric_names = [
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "pred_gt_ratio",
        "pred_fg_voxels",
        "gt_fg_voxels",
    ]

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

    dice_vals = np.array([row["dice"] for row in rows], dtype=float)
    iou_vals = np.array([row["iou"] for row in rows], dtype=float)
    precision_vals = np.array([row["precision"] for row in rows], dtype=float)
    recall_vals = np.array([row["recall"] for row in rows], dtype=float)

    print("\n=== Global summary ===")
    print(f"n samples     : {len(rows)}")
    print(f"Mean Dice     : {dice_vals.mean():.4f}")
    print(f"Mean IoU      : {iou_vals.mean():.4f}")
    print(f"Mean Precision: {precision_vals.mean():.4f}")
    print(f"Mean Recall   : {recall_vals.mean():.4f}")
    print(f"Min Dice      : {dice_vals.min():.4f}")
    print(f"Min Recall    : {recall_vals.min():.4f}")


def main(args: argparse.Namespace) -> None:
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model)
    print(f"[INFO] Using model: {model_path}")

    pairs = list(iter_img_mask_pairs(args.test_data_dir, args.label_extension))
    print(f"[INFO] Found {len(pairs)} image/mask pair(s) in {args.test_data_dir}")

    use_gpu = core.use_gpu()
    print(f"[INFO] GPU available: {use_gpu}")

    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(model_path),
    )

    rows: list[dict] = []
    failed_samples: list[dict] = []

    for img_path, mask_path, sample_id in pairs:
        try:
            metrics = run_one_sample(model, img_path, mask_path, sample_id, out_root, args)
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
        "model": str(model_path),
        "test_data_dir": str(Path(args.test_data_dir).resolve()),
        "outdir": str(out_root.resolve()),
        "label_extension": args.label_extension,
        "mask_id": int(args.mask_id),
        "channel_axis": args.channel_axis,
        "z_axis": args.z_axis,
        "infer_3d": bool(args.infer_3d),
        "anisotropy": float(args.anisotropy),
        "cellprob_threshold": float(args.cellprob_threshold),
        "flow_threshold": float(args.flow_threshold),
        "min_size": int(args.min_size),
        "n_pairs_found": int(len(pairs)),
        "n_success": int(len(rows)),
        "n_failed": int(len(failed_samples)),
        "failed_samples": failed_samples,
        "notes": [
            "Metrics are computed on binary voxel masks: pred_labels > 0 versus gt > 0.",
            "GT is binarized after selecting mask_id.",
            "Ground-truth instance coloring in slice PNGs is only for visualization.",
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
    parser = argparse.ArgumentParser(description="Pure inference for nucleus segmentation")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint file or run folder containing models/best_model",
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
        "--mask_id",
        type=int,
        default=1,
        help="Segmentation label ID to keep as foreground",
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
        "--cellprob_threshold",
        type=float,
        default=-1.25,
        help="Cell probability threshold",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.2,
        help="Flow threshold",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=15,
        help="Minimum predicted object size in pixels/voxels",
    )

    args = parser.parse_args()
    main(args)
