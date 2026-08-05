#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Threshold sweep for an already-trained Cellpose model on held-out 3D quadrants.

Current active behavior
-----------------------
- does NOT retrain
- reloads an existing trained Cellpose model
- rebuilds the held-out test quadrants exactly like segment_nuc.py
- sweeps the inference parameters
- computes semantic / voxel-level 3D metrics only:
    * IoU
    * Dice
    * Precision
    * Recall
    * Specificity
    * foreground voxel counts
    * foreground relative error

3D note
-------
In 3D Cellpose, `flow_threshold` is ignored, so in `--infer_3d` mode this script
sweeps:
    * cellprob_threshold
    * min_size
    * flow3D_smooth

and in non-3D mode it sweeps:
    * cellprob_threshold
    * flow_threshold
    * min_size

Instance metrics kept for later
-------------------------------
You asked to keep the instance metrics but comment them out for now.
So the active script below only uses semantic metrics, while the old instance-
metric code paths (AP / mAP, instance F1, SEG, AJI, PQ) are preserved as
commented reference blocks near the end of this file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import colorsys
import numpy as np
import tifffile as tiff
from PIL import Image
import nrrd
from cellpose import core, io, models


# ---------------------- Utilities: orientation / slicing -------------------------------

def ensure_zyx(arr: np.ndarray, channel_axis: int | None, z_axis: int | None, name: str = "array") -> np.ndarray:
    """
    Ensure array is (Z, Y, X). If z_axis is provided, move that axis to 0 and
    squeeze any singleton channel axis. Otherwise, auto-guess Z as the smallest
    axis if one dimension is much smaller than the other two.
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


def split_quadrants(vol: np.ndarray, split_size: int) -> Dict[str, np.ndarray]:
    """Return dict of four XY quadrants along the last two dims (Y, X)."""
    z, y, x = vol.shape
    if (y, x) != (split_size, split_size):
        raise ValueError(f"Expected Y=X={split_size}, got {(y, x)}")

    half = split_size // 2
    return {
        "TL": vol[:, 0:half, 0:half],
        "TR": vol[:, 0:half, half:split_size],
        "BL": vol[:, half:split_size, 0:half],
        "BR": vol[:, half:split_size, half:split_size],
    }


def iter_img_mask_pairs(root: str | Path = ".", label_extension: str = ".seg.nrrd"):
    """
    Yield (img_path, mask_path, sample_id) for every *_img.tif that has a
    matching *_label of type 'label_extension' in 'root'.
    """
    root = Path(root)
    for img_path in sorted(root.glob("*_img.tif")):
        mask_path = img_path.with_name(
            img_path.name.replace("_img.tif", "_label" + label_extension)
        )
        if mask_path.exists():
            sample_id = img_path.stem.replace("_img", "")
            yield img_path, mask_path, sample_id
        else:
            print(f"[WARN] missing mask for {img_path.name} -> skipped")


# ---------------------- Visualization helpers -----------------------------------------

def normalize_to_uint8(arr2d: np.ndarray) -> np.ndarray:
    a = arr2d.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


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


def label_to_rgba(inst_2d: np.ndarray, alpha: int = int(255 * 0.25)) -> np.ndarray:
    rgb = _label_to_rgb(inst_2d)
    h, w, _ = rgb.shape
    a = np.zeros((h, w), dtype=np.uint8)
    a[inst_2d > 0] = np.uint8(alpha)
    return np.dstack([rgb, a]).astype(np.uint8)


def save_png_overlays(sample_id: str, raw3d: np.ndarray, masks3d: np.ndarray, out_root: str | Path) -> None:
    outdir = Path(out_root) / sample_id
    outdir.mkdir(parents=True, exist_ok=True)

    for z in range(raw3d.shape[0]):
        raw2d = normalize_to_uint8(raw3d[z])
        inst2d = masks3d[z].astype(np.int32)

        raw_img = Image.fromarray(raw2d)
        raw_img.save(outdir / f"raw_z{z:03d}.png")

        rgba = Image.fromarray(label_to_rgba(inst2d))
        rgba.save(outdir / f"mask_rgba_z{z:03d}.png")

        base_rgba = raw_img.convert("RGBA")
        comp = Image.alpha_composite(base_rgba, rgba)
        comp.save(outdir / f"overlay_z{z:03d}.png")


# ---------------------- Metrics --------------------------------------------------------

def binary_volume_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
    tn = int(np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum())

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
        "pred_gt_ratio": float(pred_bin.sum() / gt_bin.sum()) if gt_bin.sum() > 0 else 0.0,
        "pred_fg_voxels": int(pred_bin.sum()),
        "gt_fg_voxels": int(gt_bin.sum()),
    }


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def make_gt_binary(raw_mask: np.ndarray, foreground_mode: str, gt_foreground_label: int) -> np.ndarray:
    if foreground_mode == "label1":
        return raw_mask == gt_foreground_label
    if foreground_mode == "positive":
        return raw_mask > 0
    raise ValueError("foreground_mode must be 'label1' or 'positive'")


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty rows.")

    numeric_names = []
    seen = set()
    for row in rows:
        for k, v in row.items():
            if k in seen:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                seen.add(k)
                numeric_names.append(k)

    skip = {
        "cellprob_threshold",
        "flow_threshold",
        "min_size",
        "flow3D_smooth_scalar",
        "anisotropy",
        "infer_3d",
        "gt_foreground_label",
    }
    metric_names = [k for k in numeric_names if k not in skip]

    out: Dict[str, Any] = {"n": len(rows)}
    for name in metric_names:
        vals = np.array([float(r[name]) for r in rows], dtype=float)
        out[f"mean_{name}"] = float(np.mean(vals))
        out[f"median_{name}"] = float(np.median(vals))
        out[f"std_{name}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"min_{name}"] = float(np.min(vals))
        out[f"max_{name}"] = float(np.max(vals))
    return out


def summary_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        -float(row.get("mean_dice", 0.0)),
        -float(row.get("mean_iou", 0.0)),
        float(row.get("mean_abs_fg_rel_error", 0.0)),
    )


def best_by(rows: List[Dict[str, Any]], metric_name: str, higher_is_better: bool = True) -> Dict[str, Any] | None:
    if not rows or metric_name not in rows[0]:
        return None
    if higher_is_better:
        return max(rows, key=lambda r: float(r.get(metric_name, float("-inf"))))
    return min(rows, key=lambda r: float(r.get(metric_name, float("inf"))))


# ---------------------- I/O helpers ---------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", newline="") as f:
            pass
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def tagify_value(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "x".join(tagify_value(x) for x in v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:g}".replace(".", "p").replace("-", "m")
    if isinstance(v, int):
        return str(v)
    return str(v).replace(".", "p").replace("-", "m").replace(",", "x")


# ---------------------- Data loading --------------------------------------------------

def load_test_sets(
    data_dir: Path,
    label_extension: str,
    channel_axis: int | None,
    z_axis: int | None,
    gt_foreground_mode: str,
    gt_foreground_label: int,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    print(f"\nScanning for *_img.tif / *_label{label_extension} pairs in {data_dir} ...")
    pairs = list(iter_img_mask_pairs(data_dir, label_extension))
    if not pairs:
        raise FileNotFoundError(
            f"No *_img.tif / *_label{label_extension} pairs found in {data_dir}"
        )

    test_sets: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for img_path, mask_path, sample_id in pairs:
        print(f"Loading volumes for {sample_id} ...")

        vol = tiff.imread(str(img_path))
        raw_nrrd = nrrd.read(str(mask_path))[0]
        msk_raw = np.transpose(raw_nrrd, (1, 0, 2))

        vol = ensure_zyx(vol, channel_axis, z_axis, "image")
        msk_raw = ensure_zyx(msk_raw, channel_axis, z_axis, "mask")
        assert vol.shape == msk_raw.shape, f"[{sample_id}] shape mismatch: {vol.shape} vs {msk_raw.shape}"

        gt_bin = make_gt_binary(msk_raw, gt_foreground_mode, gt_foreground_label)
        fg = int(gt_bin.sum())

        print(sample_id, "raw mask unique (first 20):", np.unique(msk_raw)[:20])
        print(sample_id, "foreground voxels after GT conversion:", fg)

        if fg == 0:
            print(f"[WARN] {sample_id}: mask has no foreground -> skipping this ROI")
            continue

        test_sets.append((sample_id, vol, gt_bin.astype(bool)))

    if not test_sets:
        raise RuntimeError("No usable test sets were constructed.")

    return test_sets


# ---------------------- Sweep setting builders ----------------------------------------

def parse_flow3d_smooth_spec(spec: str) -> float | List[float]:
    spec = str(spec).strip()
    if "," in spec:
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) != 3:
            raise ValueError(
                f"flow3D_smooth spec '{spec}' is invalid. Use a scalar like '1.5' or a 3-vector like '2,1,1'."
            )
        return vals
    return float(spec)


def smooth_spec_scalar_for_logging(spec: float | List[float]) -> float:
    if isinstance(spec, list):
        return float(np.mean(np.asarray(spec, dtype=float)))
    return float(spec)


def build_sweep_settings(args: argparse.Namespace) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []

    if args.infer_3d:
        smooth_specs = [parse_flow3d_smooth_spec(s) for s in args.flow3D_smooth_values]
        for cp in args.cellprob_values:
            for ms in args.min_size_values:
                for sm in smooth_specs:
                    settings.append(
                        {
                            "cellprob_threshold": float(cp),
                            "flow_threshold": None,
                            "min_size": int(ms),
                            "flow3D_smooth": sm,
                            "flow3D_smooth_scalar": smooth_spec_scalar_for_logging(sm),
                        }
                    )
    else:
        for cp in args.cellprob_values:
            for fl in args.flow_values:
                for ms in args.min_size_values:
                    settings.append(
                        {
                            "cellprob_threshold": float(cp),
                            "flow_threshold": float(fl),
                            "min_size": int(ms),
                            "flow3D_smooth": 0.0,
                            "flow3D_smooth_scalar": 0.0,
                        }
                    )
    return settings


# ---------------------- Main ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep inference thresholds for a trained Cellpose model without retraining"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing *_img.tif and *_label<extension> files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained Cellpose model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="threshold_sweep",
        help="Output directory for results",
    )
    parser.add_argument(
        "--label_extension",
        type=str,
        default=".seg.nrrd",
        help="Extension for label files",
    )
    parser.add_argument(
        "--test_quadrant",
        type=str,
        default="BR",
        choices=["TL", "TR", "BL", "BR"],
        help="Quadrant to evaluate",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=512,
        help="Expected XY size for quadrant split",
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
        help="Z axis if known, else None",
    )
    parser.add_argument(
        "--infer_3d",
        action="store_true",
        help="Run Cellpose in 3D mode",
    )
    parser.add_argument(
        "--anisotropy",
        type=float,
        default=1.0,
        help="Anisotropy factor for 3D inference",
    )
    parser.add_argument(
        "--cellprob_values",
        type=float,
        nargs="+",
        default=[-2.0, 0.0, 1.0, 2.0, 3.0],
        help="List of cellprob_threshold values to test",
    )
    parser.add_argument(
        "--flow_values",
        type=float,
        nargs="+",
        default=[0.4, 0.8, 1.0],
        help="List of flow_threshold values to test (used only when NOT in 3D mode)",
    )
    parser.add_argument(
        "--min_size_values",
        type=int,
        nargs="+",
        default=[15],
        help="List of min_size values to test",
    )
    parser.add_argument(
        "--flow3D_smooth_values",
        type=str,
        nargs="+",
        default=["0", "1", "2"],
        help="List of flow3D_smooth values to test in 3D mode. Each entry can be a scalar like '1.5' or a 3-vector like '2,1,1'.",
    )
    parser.add_argument(
        "--gt_foreground_mode",
        type=str,
        choices=["label1", "positive"],
        default="label1",
        help="How to convert raw NRRD labels into foreground for voxel metrics. 'label1' matches your current script; 'positive' treats all >0 as foreground.",
    )
    parser.add_argument(
        "--gt_foreground_label",
        type=int,
        default=1,
        help="Foreground label when --gt_foreground_mode label1 is used",
    )
    parser.add_argument(
        "--save_pred_tifs",
        action="store_true",
        help="Save predicted 3D masks for every setting",
    )
    parser.add_argument(
        "--save_png_overlays",
        action="store_true",
        help="Save PNG overlays for every setting",
    )

    # ---------------- Optional instance-metric CLI args kept for later ----------------
    # parser.add_argument(
    #     "--gt_instance_mode",
    #     type=str,
    #     choices=["none", "cc", "raw"],
    #     default="cc",
    #     help="How to derive GT instances for AP/F1/SEG/AJI/PQ. Default 'cc' creates 3D connected components on the binary GT. Use 'raw' only if your NRRD already stores one ID per nucleus.",
    # )
    # parser.add_argument(
    #     "--cc_connectivity",
    #     type=int,
    #     default=3,
    #     help="Connectivity for GT connected components in 3D: 1=6-neigh, 2~18-neigh, 3=26-neigh",
    # )
    # parser.add_argument(
    #     "--iou_thresholds",
    #     type=float,
    #     nargs="+",
    #     default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    #     help="IoU thresholds used for AP and instance F1",
    # )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    io.logger_setup(cp_path=str((output_dir / ".cellpose").resolve()))
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")

    if args.infer_3d:
        print(
            "[INFO] 3D mode enabled: Cellpose does not use flow_threshold in 3D, "
            "so this sweep will vary cellprob_threshold, min_size, and flow3D_smooth."
        )
    else:
        print("[INFO] 2D mode: sweep will vary cellprob_threshold, flow_threshold, and min_size.")

    print("\nPreparing held-out test sets...")
    test_sets = load_test_sets(
        data_dir=data_dir,
        label_extension=args.label_extension,
        channel_axis=args.channel_axis,
        z_axis=args.z_axis,
        gt_foreground_mode=args.gt_foreground_mode,
        gt_foreground_label=args.gt_foreground_label,
    )

    print("\nLoading trained model...")
    eval_model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(model_path),
    )

    per_sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    settings = build_sweep_settings(args)
    print(f"\nRunning sweep over {len(settings)} combinations...")

    for cfg in settings:
        cellprob_threshold = float(cfg["cellprob_threshold"])
        flow_threshold = cfg["flow_threshold"]
        min_size = int(cfg["min_size"])
        flow3D_smooth = cfg["flow3D_smooth"]
        flow3D_smooth_scalar = float(cfg["flow3D_smooth_scalar"])

        if args.infer_3d:
            tag = (
                f"cp{tagify_value(cellprob_threshold)}_"
                f"minsz{tagify_value(min_size)}_"
                f"sm{tagify_value(flow3D_smooth)}"
            )
            print(
                f"\n=== Setting: cellprob={cellprob_threshold}, min_size={min_size}, "
                f"flow3D_smooth={flow3D_smooth} ==="
            )
        else:
            tag = (
                f"cp{tagify_value(cellprob_threshold)}_"
                f"flow{tagify_value(flow_threshold)}_"
                f"minsz{tagify_value(min_size)}"
            )
            print(
                f"\n=== Setting: cellprob={cellprob_threshold}, flow={flow_threshold}, min_size={min_size} ==="
            )

        rows_this_setting: List[Dict[str, Any]] = []

        for sample_id, test_img, test_gt_bin in test_sets:
            eval_kwargs = dict(
                x=test_img,
                channels=[0, 0],
                do_3D=args.infer_3d,
                z_axis=0,
                channel_axis=None,
                anisotropy=args.anisotropy,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
            )
            if args.infer_3d:
                eval_kwargs["flow3D_smooth"] = flow3D_smooth
            else:
                eval_kwargs["flow_threshold"] = float(flow_threshold)

            masks_pred, flows, styles = eval_model.eval(**eval_kwargs)

            pred_bin = masks_pred > 0
            gt_bin = test_gt_bin > 0

            voxel_metrics = binary_volume_metrics(pred_bin, gt_bin)
            fg_rel_error = safe_div(
                float(voxel_metrics["pred_fg_voxels"]) - float(voxel_metrics["gt_fg_voxels"]),
                float(voxel_metrics["gt_fg_voxels"]),
            )
            abs_fg_rel_error = abs(fg_rel_error)

            row = {
                "sample_id": sample_id,
                "eval_split": "full_val_roi",
                "infer_3d": int(args.infer_3d),
                "anisotropy": float(args.anisotropy),
                "cellprob_threshold": float(cellprob_threshold),
                "flow_threshold": float(flow_threshold) if flow_threshold is not None else -999.0,
                "min_size": int(min_size),
                "flow3D_smooth_scalar": float(flow3D_smooth_scalar),
                "gt_foreground_label": int(args.gt_foreground_label),
                "fg_rel_error": float(fg_rel_error),
                "abs_fg_rel_error": float(abs_fg_rel_error),
                **voxel_metrics,
            }

            # ---------------- Optional instance metrics kept for later -----------------
            # pred_inst = relabel_sequential(masks_pred.astype(np.int32))
            # if args.gt_instance_mode == "none":
            #     inst_metrics = {}
            # else:
            #     inst_metrics = instance_metrics_3d(
            #         pred_labels=pred_inst,
            #         gt_labels=test_gt_inst,
            #         iou_thresholds=args.iou_thresholds,
            #     )
            # row.update(inst_metrics)

            rows_this_setting.append(row)
            per_sample_rows.append(row)

            msg = (
                f"[{sample_id}] IoU={voxel_metrics['iou']:.4f} "
                f"Dice={voxel_metrics['dice']:.4f} "
                f"Precision={voxel_metrics['precision']:.4f} "
                f"Recall={voxel_metrics['recall']:.4f}"
            )
            # if inst_metrics:
            #     msg += (
            #         f" | mAP={inst_metrics['map']:.4f} "
            #         f"mF1={inst_metrics['mf1']:.4f} "
            #         f"SEG={inst_metrics['seg']:.4f} "
            #         f"AJI={inst_metrics['aji']:.4f} "
            #         f"PQ={inst_metrics['pq']:.4f}"
            #     )
            print(msg)

            if args.save_pred_tifs:
                pred_dir = output_dir / "predictions" / tag
                pred_dir.mkdir(parents=True, exist_ok=True)
                out_tif = pred_dir / f"pred_{sample_id}_3d_mask.tif"
                tiff.imwrite(str(out_tif), masks_pred.astype(np.int32))

            if args.save_png_overlays:
                png_dir = output_dir / "png_eval" / tag
                save_png_overlays(sample_id, test_img, masks_pred, out_root=png_dir)

        summary = summarize_rows(rows_this_setting)
        summary_row = {
            "cellprob_threshold": float(cellprob_threshold),
            "flow_threshold": float(flow_threshold) if flow_threshold is not None else -999.0,
            "min_size": int(min_size),
            "flow3D_smooth_scalar": float(flow3D_smooth_scalar),
            "test_quadrant": args.test_quadrant,
            "infer_3d": int(args.infer_3d),
            "anisotropy": float(args.anisotropy),
            **summary,
        }
        summary_rows.append(summary_row)

        msg = (
            f"[SUMMARY] mean Dice={summary_row.get('mean_dice', 0.0):.4f}, "
            f"mean IoU={summary_row.get('mean_iou', 0.0):.4f}, "
            f"mean abs fg rel err={summary_row.get('mean_abs_fg_rel_error', 0.0):.4f}"
        )
        # if "mean_map" in summary_row:
        #     msg += (
        #         f", mean mAP={summary_row['mean_map']:.4f}, "
        #         f"mean mF1={summary_row['mean_mf1']:.4f}, "
        #         f"mean SEG={summary_row['mean_seg']:.4f}, "
        #         f"mean AJI={summary_row['mean_aji']:.4f}, "
        #         f"mean PQ={summary_row['mean_pq']:.4f}"
        #     )
        print(msg)

    summary_rows.sort(key=summary_sort_key)
    best_core = summary_rows[0]

    best_report = {
        "best_by_core_sort_key": best_core,
    }
    for metric_name in ["mean_dice", "mean_iou"]:
        row = best_by(summary_rows, metric_name, higher_is_better=True)
        if row is not None:
            best_report[f"best_by_{metric_name}"] = row

    # ---------------- Optional instance-metric best-by block kept for later -----------
    # for metric_name in ["mean_map", "mean_pq", "mean_seg", "mean_aji"]:
    #     row = best_by(summary_rows, metric_name, higher_is_better=True)
    #     if row is not None:
    #         best_report[f"best_by_{metric_name}"] = row

    write_csv(output_dir / "per_sample_metrics_all_settings.csv", per_sample_rows)
    write_csv(output_dir / "summary_metrics_all_settings.csv", summary_rows)
    save_json(output_dir / "best_settings_by_metric.json", best_report)

    report_lines = [
        f"model_path: {model_path}",
        f"data_dir: {data_dir}",
        f"output_dir: {output_dir}",
        "evaluation: full validation ROIs",
        f"infer_3d: {int(args.infer_3d)}",
        f"anisotropy: {args.anisotropy}",
        f"gt_foreground_mode: {args.gt_foreground_mode}",
        f"gt_foreground_label: {args.gt_foreground_label}",
        "",
        "Best setting by original core ranking (Dice, then IoU, then abs fg error):",
        f"cellprob_threshold: {best_core['cellprob_threshold']}",
        f"flow_threshold: {best_core['flow_threshold']}  (-999 means not used / 3D mode)",
        f"min_size: {best_core['min_size']}",
        f"flow3D_smooth_scalar: {best_core['flow3D_smooth_scalar']}",
        f"mean_dice: {best_core.get('mean_dice', 0.0):.6f}",
        f"mean_iou: {best_core.get('mean_iou', 0.0):.6f}",
        f"mean_precision: {best_core.get('mean_precision', 0.0):.6f}",
        f"mean_recall: {best_core.get('mean_recall', 0.0):.6f}",
        f"mean_specificity: {best_core.get('mean_specificity', 0.0):.6f}",
        f"mean_pred_fg_voxels: {best_core.get('mean_pred_fg_voxels', 0.0):.6f}",
        f"mean_gt_fg_voxels: {best_core.get('mean_gt_fg_voxels', 0.0):.6f}",
        f"mean_fg_rel_error: {best_core.get('mean_fg_rel_error', 0.0):.6f}",
        f"mean_abs_fg_rel_error: {best_core.get('mean_abs_fg_rel_error', 0.0):.6f}",
    ]

    # ---------------- Optional instance-metric report lines kept for later ------------
    # for maybe in [
    #     "mean_map",
    #     "mean_mf1",
    #     "mean_seg",
    #     "mean_aji",
    #     "mean_pq",
    #     "mean_sq",
    #     "mean_rq",
    # ]:
    #     if maybe in best_core:
    #         report_lines.append(f"{maybe}: {best_core[maybe]:.6f}")

    report_lines += [
        "",
        "Saved files:",
        f"- {output_dir / 'per_sample_metrics_all_settings.csv'}",
        f"- {output_dir / 'summary_metrics_all_settings.csv'}",
        f"- {output_dir / 'best_settings_by_metric.json'}",
    ]

    with open(output_dir / "README_results.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()


# ================================================================================
# Commented-out instance metrics reference block
# ================================================================================
#
# The code below is intentionally commented out so the active script remains
# semantic-only. You can re-enable it later if you want instance segmentation
# evaluation again.
#
# def _get_nd_label():
#     """Return an N-D connected-components function with scikit-image or scipy."""
#     try:
#         from skimage.measure import label as sk_label
#
#         def cc_label_nd(arr: np.ndarray, connectivity: int | None = None) -> np.ndarray:
#             if connectivity is None:
#                 connectivity = arr.ndim
#             return sk_label(arr, connectivity=connectivity)
#
#         return cc_label_nd
#     except Exception:
#         try:
#             from scipy.ndimage import label as sp_label
#
#             def cc_label_nd(arr: np.ndarray, connectivity: int | None = None) -> np.ndarray:
#                 if connectivity is None:
#                     connectivity = arr.ndim
#                 if arr.ndim == 2:
#                     if connectivity <= 1:
#                         structure = np.array(
#                             [[0, 1, 0],
#                              [1, 1, 1],
#                              [0, 1, 0]],
#                             dtype=int,
#                         )
#                     else:
#                         structure = np.ones((3, 3), dtype=int)
#                 elif arr.ndim == 3:
#                     if connectivity <= 1:
#                         structure = np.zeros((3, 3, 3), dtype=int)
#                         structure[1, 1, 1] = 1
#                         structure[0, 1, 1] = 1
#                         structure[2, 1, 1] = 1
#                         structure[1, 0, 1] = 1
#                         structure[1, 2, 1] = 1
#                         structure[1, 1, 0] = 1
#                         structure[1, 1, 2] = 1
#                     elif connectivity == 2:
#                         structure = np.ones((3, 3, 3), dtype=int)
#                         corners = [
#                             (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
#                             (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2),
#                         ]
#                         for c in corners:
#                             structure[c] = 0
#                     else:
#                         structure = np.ones((3, 3, 3), dtype=int)
#                 else:
#                     raise ValueError("Connected components helper supports only 2D or 3D arrays")
#                 out, _ = sp_label(arr, structure=structure)
#                 return out
#
#             return cc_label_nd
#         except Exception:
#             def cc_label_nd(_arr: np.ndarray, _connectivity: int | None = None) -> np.ndarray:
#                 raise ImportError(
#                     "Need scikit-image or scipy for connected components.\n"
#                     "Install one of:\n"
#                     "  pip install scikit-image\n"
#                     "  pip install scipy"
#                 )
#             return cc_label_nd
#
#
# cc_label_nd = _get_nd_label()
#
#
# def _get_linear_sum_assignment():
#     try:
#         from scipy.optimize import linear_sum_assignment
#         return linear_sum_assignment
#     except Exception:
#         return None
#
#
# linear_sum_assignment = _get_linear_sum_assignment()
#
#
# def relabel_sequential(labels: np.ndarray) -> np.ndarray:
#     labels = np.asarray(labels)
#     out = np.zeros_like(labels, dtype=np.int32)
#     vals = np.unique(labels)
#     vals = vals[vals > 0]
#     for new_id, old_id in enumerate(vals, start=1):
#         out[labels == old_id] = new_id
#     return out
#
#
# def make_gt_instances(
#     raw_mask: np.ndarray,
#     gt_bin: np.ndarray,
#     gt_instance_mode: str,
#     cc_connectivity: int,
# ) -> np.ndarray:
#     if gt_instance_mode == "none":
#         return np.zeros_like(gt_bin, dtype=np.int32)
#     if gt_instance_mode == "cc":
#         return relabel_sequential(cc_label_nd(gt_bin.astype(np.uint8), connectivity=cc_connectivity))
#     if gt_instance_mode == "raw":
#         raw = np.asarray(raw_mask)
#         if np.any(raw < 0):
#             raw = raw.copy()
#             raw[raw < 0] = 0
#         return relabel_sequential(raw.astype(np.int32))
#     raise ValueError("gt_instance_mode must be one of {'none','cc','raw'}")
#
#
# def encode_pairs(gt_labels: np.ndarray, pred_labels: np.ndarray):
#     gt_flat = gt_labels.ravel().astype(np.int64)
#     pred_flat = pred_labels.ravel().astype(np.int64)
#     gt_areas = np.bincount(gt_flat)
#     pred_areas = np.bincount(pred_flat)
#     pair_ids = (gt_flat << 32) | pred_flat
#     uniq_pair_ids, counts = np.unique(pair_ids, return_counts=True)
#     return uniq_pair_ids, counts, gt_areas, pred_areas
#
#
# def intersection_iou_pairs(gt_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, Any]:
#     gt_labels = relabel_sequential(gt_labels)
#     pred_labels = relabel_sequential(pred_labels)
#     uniq_pair_ids, counts, gt_areas, pred_areas = encode_pairs(gt_labels, pred_labels)
#     gt_ids_all = (uniq_pair_ids >> 32).astype(np.int64)
#     pred_ids_all = (uniq_pair_ids & ((1 << 32) - 1)).astype(np.int64)
#     gt_ids = np.arange(1, len(gt_areas), dtype=np.int64)
#     pred_ids = np.arange(1, len(pred_areas), dtype=np.int64)
#     keep = (gt_ids_all > 0) & (pred_ids_all > 0)
#     pair_gt_ids = gt_ids_all[keep]
#     pair_pred_ids = pred_ids_all[keep]
#     pair_intersections = counts[keep].astype(np.float64)
#
#     if pair_intersections.size == 0:
#         pair_ious = np.zeros((0,), dtype=np.float64)
#     else:
#         unions = gt_areas[pair_gt_ids].astype(np.float64) + pred_areas[pair_pred_ids].astype(np.float64) - pair_intersections
#         pair_ious = np.divide(
#             pair_intersections,
#             unions,
#             out=np.zeros_like(pair_intersections, dtype=np.float64),
#             where=unions > 0,
#         )
#
#     return {
#         "gt_labels": gt_labels,
#         "pred_labels": pred_labels,
#         "gt_ids": gt_ids,
#         "pred_ids": pred_ids,
#         "gt_areas": gt_areas.astype(np.float64),
#         "pred_areas": pred_areas.astype(np.float64),
#         "pair_gt_ids": pair_gt_ids,
#         "pair_pred_ids": pair_pred_ids,
#         "pair_intersections": pair_intersections,
#         "pair_ious": pair_ious,
#     }
#
#
# def build_dense_iou_matrix(overlap: Dict[str, Any]) -> np.ndarray:
#     n_gt = int(len(overlap["gt_ids"]))
#     n_pred = int(len(overlap["pred_ids"]))
#     mat = np.zeros((n_gt, n_pred), dtype=np.float64)
#     if n_gt == 0 or n_pred == 0:
#         return mat
#     gt_pos = overlap["pair_gt_ids"] - 1
#     pred_pos = overlap["pair_pred_ids"] - 1
#     mat[gt_pos, pred_pos] = overlap["pair_ious"]
#     return mat
#
#
# def build_dense_intersection_matrix(overlap: Dict[str, Any]) -> np.ndarray:
#     n_gt = int(len(overlap["gt_ids"]))
#     n_pred = int(len(overlap["pred_ids"]))
#     mat = np.zeros((n_gt, n_pred), dtype=np.float64)
#     if n_gt == 0 or n_pred == 0:
#         return mat
#     gt_pos = overlap["pair_gt_ids"] - 1
#     pred_pos = overlap["pair_pred_ids"] - 1
#     mat[gt_pos, pred_pos] = overlap["pair_intersections"]
#     return mat
#
#
# def hungarian_or_greedy_matches(score_matrix: np.ndarray):
#     if score_matrix.size == 0:
#         return []
#     n_gt, n_pred = score_matrix.shape
#     if linear_sum_assignment is not None:
#         try:
#             rows, cols = linear_sum_assignment(score_matrix, maximize=True)
#         except TypeError:
#             rows, cols = linear_sum_assignment(-score_matrix)
#         return [(int(r), int(c), float(score_matrix[r, c])) for r, c in zip(rows, cols)]
#
#     triples = []
#     for i in range(n_gt):
#         for j in range(n_pred):
#             s = float(score_matrix[i, j])
#             if s > 0:
#                 triples.append((i, j, s))
#     triples.sort(key=lambda t: t[2], reverse=True)
#
#     used_gt = set()
#     used_pred = set()
#     out = []
#     for i, j, s in triples:
#         if i in used_gt or j in used_pred:
#             continue
#         used_gt.add(i)
#         used_pred.add(j)
#         out.append((i, j, s))
#     return out
#
#
# def match_counts_at_threshold(iou_matrix: np.ndarray, thr: float) -> Dict[str, float]:
#     n_gt, n_pred = iou_matrix.shape
#     if n_gt == 0 and n_pred == 0:
#         return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
#     if n_gt == 0:
#         return {"tp": 0.0, "fp": float(n_pred), "fn": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
#     if n_pred == 0:
#         return {"tp": 0.0, "fp": 0.0, "fn": float(n_gt), "precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
#
#     matches = hungarian_or_greedy_matches(iou_matrix)
#     tp = float(sum(1 for _, _, score in matches if score >= thr))
#     fp = float(n_pred) - tp
#     fn = float(n_gt) - tp
#     precision = safe_div(tp, tp + fp)
#     recall = safe_div(tp, tp + fn)
#     f1 = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
#     ap = safe_div(tp, tp + fp + fn)
#     return {
#         "tp": tp,
#         "fp": fp,
#         "fn": fn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "ap": ap,
#     }
#
#
# def seg_metric(iou_matrix: np.ndarray) -> float:
#     if iou_matrix.shape[0] == 0:
#         return 0.0
#     best_per_gt = iou_matrix.max(axis=1) if iou_matrix.shape[1] > 0 else np.zeros((iou_matrix.shape[0],), dtype=np.float64)
#     best_per_gt = np.where(best_per_gt > 0.5, best_per_gt, 0.0)
#     return float(best_per_gt.mean())
#
#
# def aji_metric(overlap: Dict[str, Any], iou_matrix: np.ndarray) -> float:
#     n_gt = iou_matrix.shape[0]
#     n_pred = iou_matrix.shape[1]
#     if n_gt == 0 and n_pred == 0:
#         return 0.0
#
#     inter_mat = build_dense_intersection_matrix(overlap)
#     matches = hungarian_or_greedy_matches(iou_matrix)
#     gt_areas = overlap["gt_areas"]
#     pred_areas = overlap["pred_areas"]
#
#     matched_gt = set()
#     matched_pred = set()
#     numerator = 0.0
#     denominator = 0.0
#
#     for gi, pj, iou_val in matches:
#         if iou_val <= 0:
#             continue
#         g_id = gi + 1
#         p_id = pj + 1
#         inter = float(inter_mat[gi, pj])
#         union = float(gt_areas[g_id] + pred_areas[p_id] - inter)
#         numerator += inter
#         denominator += union
#         matched_gt.add(g_id)
#         matched_pred.add(p_id)
#
#     unmatched_gt = [g for g in range(1, len(gt_areas)) if g not in matched_gt]
#     unmatched_pred = [p for p in range(1, len(pred_areas)) if p not in matched_pred]
#     denominator += float(np.sum(gt_areas[unmatched_gt])) if unmatched_gt else 0.0
#     denominator += float(np.sum(pred_areas[unmatched_pred])) if unmatched_pred else 0.0
#     return safe_div(numerator, denominator)
#
#
# def pq_metrics(iou_matrix: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
#     n_gt = iou_matrix.shape[0]
#     n_pred = iou_matrix.shape[1]
#     if n_gt == 0 and n_pred == 0:
#         return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}
#
#     matches = hungarian_or_greedy_matches(iou_matrix)
#     good = [(gi, pj, s) for gi, pj, s in matches if s > thr]
#     tp = float(len(good))
#     fp = float(n_pred) - tp
#     fn = float(n_gt) - tp
#     sum_iou = float(sum(s for _, _, s in good))
#     sq = safe_div(sum_iou, tp)
#     rq = safe_div(tp, tp + 0.5 * fp + 0.5 * fn)
#     pq = sq * rq
#     return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}
#
#
# def instance_metrics_3d(
#     pred_labels: np.ndarray,
#     gt_labels: np.ndarray,
#     iou_thresholds,
# ) -> Dict[str, float]:
#     overlap = intersection_iou_pairs(gt_labels, pred_labels)
#     iou_matrix = build_dense_iou_matrix(overlap)
#
#     out: Dict[str, float] = {
#         "n_gt_instances": float(iou_matrix.shape[0]),
#         "n_pred_instances": float(iou_matrix.shape[1]),
#     }
#
#     ap_vals = []
#     f1_vals = []
#     prec_vals = []
#     rec_vals = []
#
#     for thr in iou_thresholds:
#         stats = match_counts_at_threshold(iou_matrix, float(thr))
#         thr_tag = f"{int(round(thr * 100)):02d}"
#         out[f"ap_{thr_tag}"] = float(stats["ap"])
#         out[f"inst_f1_{thr_tag}"] = float(stats["f1"])
#         out[f"inst_precision_{thr_tag}"] = float(stats["precision"])
#         out[f"inst_recall_{thr_tag}"] = float(stats["recall"])
#         out[f"inst_tp_{thr_tag}"] = float(stats["tp"])
#         out[f"inst_fp_{thr_tag}"] = float(stats["fp"])
#         out[f"inst_fn_{thr_tag}"] = float(stats["fn"])
#         ap_vals.append(float(stats["ap"]))
#         f1_vals.append(float(stats["f1"]))
#         prec_vals.append(float(stats["precision"]))
#         rec_vals.append(float(stats["recall"]))
#
#     out["map"] = float(np.mean(ap_vals)) if ap_vals else 0.0
#     out["mf1"] = float(np.mean(f1_vals)) if f1_vals else 0.0
#     out["minst_precision"] = float(np.mean(prec_vals)) if prec_vals else 0.0
#     out["minst_recall"] = float(np.mean(rec_vals)) if rec_vals else 0.0
#     out["seg"] = float(seg_metric(iou_matrix))
#     out["aji"] = float(aji_metric(overlap, iou_matrix))
#
#     pq = pq_metrics(iou_matrix, thr=0.5)
#     out["pq"] = float(pq["pq"])
#     out["sq"] = float(pq["sq"])
#     out["rq"] = float(pq["rq"])
#     out["pq_tp_50"] = float(pq["tp"])
#     out["pq_fp_50"] = float(pq["fp"])
#     out["pq_fn_50"] = float(pq["fn"])
#
#     return out
