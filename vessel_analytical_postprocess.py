#!/usr/bin/env python3
"""Calibrate and apply a fixed-window analytical vessel postprocessor.

Local PCA is measured in overlapping 3D windows. A vessel-like window has one
large eigenvalue and two similar smaller eigenvalues. Components are retained
when their local-window score passes a calibrated threshold. Calibration uses
nested folds over existing out-of-fold Cellpose predictions; Cellpose is never
rerun here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


SCALES = ("unscaled", "mid", "coarse")
SCORE_STATISTICS = ("maximum", "top3_mean", "q75")
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class Sample:
    fold: int
    scale: str
    sample_id: str
    pred_path: Path
    gt_path: Path
    gt_voxels: int
    shape: tuple[int, int, int]


def discover_samples(cv_run: Path, n_folds: int) -> list[Sample]:
    samples: list[Sample] = []
    for fold in range(1, n_folds + 1):
        for scale in SCALES:
            inference_dir = cv_run / f"fold_{fold:02d}" / scale / "inference_eval"
            predictions = sorted(inference_dir.glob("*/pred_mask3d.tif"))
            if len(predictions) != 1:
                raise RuntimeError(
                    f"Expected one test prediction in {inference_dir}, found {len(predictions)}"
                )
            pred_path = predictions[0]
            gt_path = pred_path.with_name("gt_mask3d.tif")
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            pred = tifffile.imread(pred_path)
            gt = tifffile.imread(gt_path) > 0
            if pred.ndim != 3 or pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {pred_path.parent.name}: pred={pred.shape}, GT={gt.shape}"
                )
            samples.append(Sample(
                fold=fold,
                scale=scale,
                sample_id=pred_path.parent.name,
                pred_path=pred_path,
                gt_path=gt_path,
                gt_voxels=int(np.count_nonzero(gt)),
                shape=tuple(int(value) for value in pred.shape),
            ))
    return samples


def window_starts(length: int, width: int) -> list[int]:
    if width >= length:
        return [0]
    stride = max(1, width // 2)
    starts = list(range(0, length - width + 1, stride))
    final = length - width
    if starts[-1] != final:
        starts.append(final)
    return starts


def local_tubularity(coords: np.ndarray, min_foreground_voxels: int) -> float:
    if len(coords) < min_foreground_voxels:
        return 0.0
    covariance = np.cov(coords.astype(np.float64), rowvar=False, bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    l1, l2, l3 = (max(0.0, float(value)) for value in eigenvalues)
    epsilon = np.finfo(np.float64).eps
    linearity = max(0.0, min(1.0, (l1 - l2) / max(l1, epsilon)))
    cross_section_roundness = math.sqrt((l3 + epsilon) / (l2 + epsilon))
    cross_section_roundness = max(0.0, min(1.0, cross_section_roundness))
    return linearity * cross_section_roundness


def target_class(overlap_fraction: float, positive: float, negative: float) -> str:
    if overlap_fraction >= positive:
        return "positive"
    if overlap_fraction <= negative:
        return "negative"
    return "ambiguous"


def component_window_features(
    pred: np.ndarray,
    gt: np.ndarray | None,
    window_sizes: list[int],
    min_foreground_voxels: int,
    positive_overlap: float,
    negative_overlap: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels, n_components = ndimage.label(pred > 0, structure=CONNECTIVITY_26)
    component_voxels = np.bincount(labels.ravel(), minlength=n_components + 1)
    if gt is None:
        overlap_voxels = np.zeros(n_components + 1, dtype=np.int64)
    else:
        overlap_voxels = np.bincount(
            labels[np.asarray(gt, dtype=bool)].ravel(), minlength=n_components + 1
        )
    boxes = ndimage.find_objects(labels)
    touches_boundary = np.zeros(n_components + 1, dtype=bool)
    for component_id, box in enumerate(boxes, start=1):
        if box is None:
            continue
        touches_boundary[component_id] = any(
            axis.start == 0 or axis.stop == pred.shape[index]
            for index, axis in enumerate(box)
        )

    scores: dict[tuple[int, int], list[float]] = {
        (component_id, width): []
        for component_id in range(1, n_components + 1)
        for width in window_sizes
    }
    for width in window_sizes:
        z_starts = window_starts(pred.shape[0], width)
        y_starts = window_starts(pred.shape[1], width)
        x_starts = window_starts(pred.shape[2], width)
        for z0 in z_starts:
            z1 = min(pred.shape[0], z0 + width)
            for y0 in y_starts:
                y1 = min(pred.shape[1], y0 + width)
                for x0 in x_starts:
                    x1 = min(pred.shape[2], x0 + width)
                    crop = labels[z0:z1, y0:y1, x0:x1]
                    component_ids = np.unique(crop)
                    component_ids = component_ids[component_ids > 0]
                    for component_id in component_ids:
                        coords = np.argwhere(crop == component_id)
                        scores[(int(component_id), width)].append(
                            local_tubularity(coords, min_foreground_voxels)
                        )

    rows: list[dict] = []
    for component_id in range(1, n_components + 1):
        n_voxels = int(component_voxels[component_id])
        overlap = int(overlap_voxels[component_id])
        overlap_fraction = overlap / n_voxels if n_voxels else 0.0
        for width in window_sizes:
            values = np.asarray(scores[(component_id, width)], dtype=np.float64)
            if len(values):
                ordered = np.sort(values)
                maximum = float(ordered[-1])
                top3_mean = float(np.mean(ordered[-min(3, len(ordered)):]))
                q75 = float(np.quantile(ordered, 0.75))
            else:
                maximum = top3_mean = q75 = 0.0
            rows.append({
                "component_id": component_id,
                "window_size": width,
                "component_voxels": n_voxels,
                "gt_overlap_voxels": overlap,
                "gt_overlap_fraction": overlap_fraction,
                "target_class": target_class(
                    overlap_fraction, positive_overlap, negative_overlap
                ),
                "touches_volume_boundary": bool(touches_boundary[component_id]),
                "n_occupied_windows": int(len(values)),
                "maximum": maximum,
                "top3_mean": top3_mean,
                "q75": q75,
            })
    columns = [
        "component_id", "window_size", "component_voxels", "gt_overlap_voxels",
        "gt_overlap_fraction", "target_class", "touches_volume_boundary",
        "n_occupied_windows", *SCORE_STATISTICS,
    ]
    return labels, pd.DataFrame(rows, columns=columns)


def binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "recall": recall, "dice": dice, "iou": iou,
    }


def sample_metrics_for_rule(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    window_size: int,
    score_statistic: str,
    threshold: float,
) -> pd.DataFrame:
    selected_width = components[components["window_size"] == window_size]
    rows: list[dict] = []
    for sample in samples.itertuples(index=False):
        frame = selected_width[selected_width["sample_key"] == sample.sample_key]
        keep = (
            (frame[score_statistic].to_numpy() >= threshold)
            | frame["touches_volume_boundary"].to_numpy(dtype=bool)
        )
        tp = int(frame.loc[keep, "gt_overlap_voxels"].sum())
        pred_voxels = int(frame.loc[keep, "component_voxels"].sum())
        filtered = binary_metrics(tp, pred_voxels - tp, int(sample.gt_voxels) - tp)
        baseline_tp = int(frame["gt_overlap_voxels"].sum())
        baseline_pred = int(frame["component_voxels"].sum())
        baseline = binary_metrics(
            baseline_tp, baseline_pred - baseline_tp, int(sample.gt_voxels) - baseline_tp
        )
        rows.append({
            "fold": int(sample.fold), "scale": sample.scale,
            "sample_id": sample.sample_id, "sample_key": sample.sample_key,
            "window_size": window_size, "score_statistic": score_statistic,
            "tubularity_threshold": threshold,
            **{f"baseline_{key}": value for key, value in baseline.items()},
            **{f"filtered_{key}": value for key, value in filtered.items()},
            "precision_change": filtered["precision"] - baseline["precision"],
            "recall_change": filtered["recall"] - baseline["recall"],
            "dice_change": filtered["dice"] - baseline["dice"],
        })
    return pd.DataFrame(rows)


def pooled_from_samples(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    return binary_metrics(
        int(frame[f"{prefix}_tp"].sum()),
        int(frame[f"{prefix}_fp"].sum()),
        int(frame[f"{prefix}_fn"].sum()),
    )


def evaluate_grid(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    window_sizes: list[int],
    thresholds: np.ndarray,
    max_recall_loss: float,
    constrained_scales: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    for width in window_sizes:
        width_frame = components[components["window_size"] == width]
        labels = width_frame["target_class"].to_numpy()
        for statistic in SCORE_STATISTICS:
            values = width_frame[statistic].to_numpy()
            boundaries = width_frame["touches_volume_boundary"].to_numpy(dtype=bool)
            for threshold in thresholds:
                keep = (values >= threshold) | boundaries
                metric_rows = sample_metrics_for_rule(
                    components, samples, width, statistic, float(threshold)
                )
                pooled = pooled_from_samples(metric_rows, "filtered")
                recall_losses = []
                row = {
                    "window_size": width,
                    "score_statistic": statistic,
                    "tubularity_threshold": float(threshold),
                    "macro_precision": float(metric_rows["filtered_precision"].mean()),
                    "macro_recall": float(metric_rows["filtered_recall"].mean()),
                    "macro_dice": float(metric_rows["filtered_dice"].mean()),
                    **{f"pooled_{key}": value for key, value in pooled.items()},
                    "positive_component_recall": float(np.mean(keep[labels == "positive"]))
                    if np.any(labels == "positive") else math.nan,
                    "negative_component_rejection": float(np.mean(~keep[labels == "negative"]))
                    if np.any(labels == "negative") else math.nan,
                }
                for scale in constrained_scales:
                    scale_rows = metric_rows[metric_rows["scale"] == scale]
                    baseline = pooled_from_samples(scale_rows, "baseline")
                    filtered = pooled_from_samples(scale_rows, "filtered")
                    loss = float(baseline["recall"] - filtered["recall"])
                    row[f"{scale}_baseline_recall"] = baseline["recall"]
                    row[f"{scale}_filtered_recall"] = filtered["recall"]
                    row[f"{scale}_recall_loss"] = loss
                    recall_losses.append(loss)
                row["max_scale_recall_loss"] = max(recall_losses) if recall_losses else 0.0
                row["eligible"] = row["max_scale_recall_loss"] <= max_recall_loss + 1e-12
                rows.append(row)
    return pd.DataFrame(rows)


def select_rule(search: pd.DataFrame) -> pd.Series:
    eligible = search[search["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No rule satisfies the per-scale recall constraint")
    return eligible.sort_values(
        ["macro_precision", "macro_dice", "pooled_precision", "tubularity_threshold", "window_size"],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def evaluate_coarse_thresholds(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    window_size: int,
    score_statistic: str,
    thresholds: np.ndarray,
    max_pooled_recall_loss: float,
    max_mean_recall_loss: float,
    max_sample_recall_loss: float,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = sample_metrics_for_rule(
            components, samples, window_size, score_statistic, float(threshold)
        )
        baseline = pooled_from_samples(metrics, "baseline")
        filtered = pooled_from_samples(metrics, "filtered")
        sample_losses = metrics["baseline_recall"] - metrics["filtered_recall"]
        pooled_loss = float(baseline["recall"] - filtered["recall"])
        mean_loss = float(sample_losses.mean())
        worst_loss = float(sample_losses.max())
        rows.append({
            "window_size": window_size,
            "score_statistic": score_statistic,
            "tubularity_threshold": float(threshold),
            "macro_precision": float(metrics["filtered_precision"].mean()),
            "macro_recall": float(metrics["filtered_recall"].mean()),
            "macro_dice": float(metrics["filtered_dice"].mean()),
            **{f"pooled_{key}": value for key, value in filtered.items()},
            "pooled_recall_loss": pooled_loss,
            "mean_sample_recall_loss": mean_loss,
            "max_sample_recall_loss": worst_loss,
            "eligible": (
                pooled_loss <= max_pooled_recall_loss + 1e-12
                and mean_loss <= max_mean_recall_loss + 1e-12
                and worst_loss <= max_sample_recall_loss + 1e-12
            ),
        })
    return pd.DataFrame(rows)


def select_coarse_threshold(search: pd.DataFrame) -> pd.Series:
    eligible = search[search["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No coarse threshold satisfies the recall constraints")
    return eligible.sort_values(
        ["macro_precision", "macro_dice", "pooled_precision", "tubularity_threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def rule_from_row(row: pd.Series) -> dict:
    return {
        "window_size": int(row["window_size"]),
        "score_statistic": str(row["score_statistic"]),
        "tubularity_threshold": float(row["tubularity_threshold"]),
    }


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", frame), *list(frame.groupby("scale", sort=False))]
    for scale, group in groups:
        for setting in ("baseline", "filtered"):
            pooled = pooled_from_samples(group, setting)
            rows.append({
                "scale": scale, "setting": setting, "n_samples": len(group),
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"], "pooled_dice": pooled["dice"],
                "mean_sample_precision": float(group[f"{setting}_precision"].mean()),
                "mean_sample_recall": float(group[f"{setting}_recall"].mean()),
                "mean_sample_dice": float(group[f"{setting}_dice"].mean()),
                "tp": pooled["tp"], "fp": pooled["fp"], "fn": pooled["fn"],
            })
    return pd.DataFrame(rows)


def save_filtered_masks(
    cv_run: Path,
    outdir: Path,
    components: pd.DataFrame,
    sample_rows: pd.DataFrame,
    rule: dict,
) -> None:
    width = int(rule["window_size"])
    statistic = rule["score_statistic"]
    threshold = float(rule["tubularity_threshold"])
    for sample in sample_rows.itertuples(index=False):
        pred_path = (
            cv_run / f"fold_{int(sample.fold):02d}" / sample.scale / "inference_eval"
            / sample.sample_id / "pred_mask3d.tif"
        )
        pred = tifffile.imread(pred_path)
        labels, _ = ndimage.label(pred > 0, structure=CONNECTIVITY_26)
        frame = components[
            (components["sample_key"] == sample.sample_key)
            & (components["window_size"] == width)
        ]
        keep = (
            (frame[statistic].to_numpy() >= threshold)
            | frame["touches_volume_boundary"].to_numpy(dtype=bool)
        )
        keep_ids = frame.loc[keep, "component_id"].to_numpy(dtype=np.int64)
        filtered = np.isin(labels, keep_ids).astype(np.uint8)
        destination = outdir / f"fold_{int(sample.fold):02d}" / sample.scale / sample.sample_id
        destination.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(destination / "pred_mask3d_analytical_filtered.tif", filtered)


def run_meta_cv(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    window_sizes: list[int],
    thresholds: np.ndarray,
    max_recall_loss: float,
    n_folds: int,
    search_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    heldout_rows = []
    search_dir.mkdir(parents=True, exist_ok=True)
    for heldout_fold in range(1, n_folds + 1):
        train_samples = samples[samples["fold"] != heldout_fold]
        train_keys = set(train_samples["sample_key"])
        train_components = components[components["sample_key"].isin(train_keys)]
        search = evaluate_grid(
            train_components, train_samples, window_sizes, thresholds,
            max_recall_loss, SCALES,
        )
        search.to_csv(search_dir / f"meta_fold_{heldout_fold:02d}.csv", index=False)
        selected = select_rule(search)
        rule = rule_from_row(selected)
        selected_rows.append({"heldout_fold": heldout_fold, **rule, **selected.to_dict()})
        heldout_samples = samples[samples["fold"] == heldout_fold]
        heldout_rows.append(sample_metrics_for_rule(
            components, heldout_samples, rule["window_size"],
            rule["score_statistic"], rule["tubularity_threshold"],
        ))
        print(f"Meta-fold {heldout_fold:02d}: {rule}", flush=True)
    return pd.DataFrame(selected_rows), pd.concat(heldout_rows, ignore_index=True)


def run_scale_specific_meta_cv(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    window_sizes: list[int],
    thresholds: np.ndarray,
    max_recall_loss: float,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    heldout_rows = []
    for heldout_fold in range(1, n_folds + 1):
        for scale in SCALES:
            train_samples = samples[
                (samples["fold"] != heldout_fold) & (samples["scale"] == scale)
            ]
            train_keys = set(train_samples["sample_key"])
            train_components = components[components["sample_key"].isin(train_keys)]
            search = evaluate_grid(
                train_components, train_samples, window_sizes, thresholds,
                max_recall_loss, (scale,),
            )
            selected = select_rule(search)
            rule = rule_from_row(selected)
            selected_rows.append({
                "heldout_fold": heldout_fold, "scale": scale,
                **rule, **selected.to_dict(),
            })
            heldout_samples = samples[
                (samples["fold"] == heldout_fold) & (samples["scale"] == scale)
            ]
            heldout_rows.append(sample_metrics_for_rule(
                components, heldout_samples, rule["window_size"],
                rule["score_statistic"], rule["tubularity_threshold"],
            ))
    return pd.DataFrame(selected_rows), pd.concat(heldout_rows, ignore_index=True)


def run_coarse_stage2_meta_cv(
    components: pd.DataFrame,
    samples: pd.DataFrame,
    shared_meta_selected: pd.DataFrame,
    thresholds: np.ndarray,
    max_pooled_recall_loss: float,
    max_mean_recall_loss: float,
    max_sample_recall_loss: float,
    search_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    heldout_rows = []
    search_dir.mkdir(parents=True, exist_ok=True)
    for shared_selection in shared_meta_selected.itertuples(index=False):
        heldout_fold = int(shared_selection.heldout_fold)
        window_size = int(shared_selection.window_size)
        score_statistic = str(shared_selection.score_statistic)
        train_samples = samples[
            (samples["fold"] != heldout_fold) & (samples["scale"] == "coarse")
        ]
        train_keys = set(train_samples["sample_key"])
        train_components = components[components["sample_key"].isin(train_keys)]
        search = evaluate_coarse_thresholds(
            train_components, train_samples, window_size, score_statistic, thresholds,
            max_pooled_recall_loss, max_mean_recall_loss, max_sample_recall_loss,
        )
        search.to_csv(search_dir / f"coarse_meta_fold_{heldout_fold:02d}.csv", index=False)
        selected = select_coarse_threshold(search)
        selected_rows.append({"heldout_fold": heldout_fold, **selected.to_dict()})
        heldout_samples = samples[
            (samples["fold"] == heldout_fold) & (samples["scale"] == "coarse")
        ]
        heldout_rows.append(sample_metrics_for_rule(
            components, heldout_samples, window_size, score_statistic,
            float(selected["tubularity_threshold"]),
        ))
        print(
            f"Coarse stage 2 fold {heldout_fold:02d}: window={window_size} "
            f"statistic={score_statistic} threshold={selected['tubularity_threshold']:g}",
            flush=True,
        )
    return pd.DataFrame(selected_rows), pd.concat(heldout_rows, ignore_index=True)


def calibrate(args: argparse.Namespace) -> None:
    cv_run = Path(args.cv_run_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    window_sizes = sorted(set(args.window_sizes))
    if min(window_sizes) < 3:
        raise ValueError("Window sizes must be at least 3")
    thresholds = np.arange(0.0, 1.0 + args.threshold_step / 2, args.threshold_step)
    samples = discover_samples(cv_run, args.n_folds)
    sample_table = pd.DataFrame([{
        "fold": sample.fold, "scale": sample.scale, "sample_id": sample.sample_id,
        "sample_key": f"{sample.fold}:{sample.scale}:{sample.sample_id}",
        "gt_voxels": sample.gt_voxels,
        "shape_z": sample.shape[0], "shape_y": sample.shape[1], "shape_x": sample.shape[2],
        "pred_path": str(sample.pred_path), "gt_path": str(sample.gt_path),
    } for sample in samples])
    sample_table.to_csv(outdir / "samples.csv", index=False)

    feature_tables = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[{index}/{len(samples)}] fixed-window features: fold={sample.fold:02d} "
            f"scale={sample.scale} sample={sample.sample_id} shape={sample.shape}", flush=True,
        )
        pred = tifffile.imread(sample.pred_path)
        gt = tifffile.imread(sample.gt_path) > 0
        _, frame = component_window_features(
            pred, gt, window_sizes, args.min_foreground_voxels,
            args.positive_overlap, args.negative_overlap,
        )
        frame.insert(0, "sample_key", f"{sample.fold}:{sample.scale}:{sample.sample_id}")
        frame.insert(0, "sample_id", sample.sample_id)
        frame.insert(0, "scale", sample.scale)
        frame.insert(0, "fold", sample.fold)
        feature_tables.append(frame)
    components = pd.concat(feature_tables, ignore_index=True)
    components.to_csv(outdir / "component_window_features.csv", index=False)

    selected_meta, meta_metrics = run_meta_cv(
        components, sample_table, window_sizes, thresholds,
        args.max_recall_loss, args.n_folds,
        outdir / "threshold_search_by_meta_fold",
    )
    selected_meta.to_csv(outdir / "meta_cv_selected_thresholds.csv", index=False)
    meta_metrics.to_csv(outdir / "meta_cv_metrics_per_sample.csv", index=False)
    meta_summary = summarize_metrics(meta_metrics)
    meta_summary.to_csv(outdir / "meta_cv_summary_by_scale.csv", index=False)

    scale_meta_selected, scale_meta_metrics = run_scale_specific_meta_cv(
        components, sample_table, window_sizes, thresholds,
        args.max_recall_loss, args.n_folds,
    )
    scale_meta_selected.to_csv(
        outdir / "meta_cv_selected_thresholds_scale_specific.csv", index=False
    )
    scale_meta_metrics.to_csv(
        outdir / "meta_cv_metrics_per_sample_scale_specific.csv", index=False
    )
    scale_meta_summary = summarize_metrics(scale_meta_metrics)
    scale_meta_summary.to_csv(
        outdir / "meta_cv_summary_by_scale_scale_specific.csv", index=False
    )

    coarse_stage2_selected, coarse_stage2_metrics = run_coarse_stage2_meta_cv(
        components, sample_table, selected_meta, thresholds,
        args.coarse_max_pooled_recall_loss,
        args.coarse_max_mean_recall_loss,
        args.coarse_max_sample_recall_loss,
        outdir / "coarse_stage2_threshold_search_by_meta_fold",
    )
    coarse_stage2_selected.to_csv(
        outdir / "coarse_stage2_meta_cv_selected_thresholds.csv", index=False
    )
    coarse_stage2_metrics.to_csv(
        outdir / "coarse_stage2_meta_cv_metrics_per_sample.csv", index=False
    )
    coarse_stage2_summary = summarize_metrics(coarse_stage2_metrics)
    coarse_stage2_summary.to_csv(
        outdir / "coarse_stage2_meta_cv_summary.csv", index=False
    )
    for selected in coarse_stage2_selected.itertuples(index=False):
        heldout_coarse = sample_table[
            (sample_table["fold"] == selected.heldout_fold)
            & (sample_table["scale"] == "coarse")
        ]
        save_filtered_masks(
            cv_run, outdir / "coarse_stage2_meta_cv_filtered_masks",
            components, heldout_coarse,
            {
                "window_size": selected.window_size,
                "score_statistic": selected.score_statistic,
                "tubularity_threshold": selected.tubularity_threshold,
            },
        )

    deployment_search = evaluate_grid(
        components, sample_table, window_sizes, thresholds,
        args.max_recall_loss, SCALES,
    )
    deployment_search.to_csv(outdir / "deployment_threshold_search.csv", index=False)
    shared_rule = rule_from_row(select_rule(deployment_search))
    shared_calibration_metrics = sample_metrics_for_rule(
        components, sample_table, shared_rule["window_size"],
        shared_rule["score_statistic"], shared_rule["tubularity_threshold"],
    )
    shared_calibration_metrics.to_csv(
        outdir / "deployment_calibration_metrics_per_sample.csv", index=False
    )
    shared_calibration_summary = summarize_metrics(shared_calibration_metrics)
    shared_calibration_summary.to_csv(
        outdir / "deployment_calibration_summary_by_scale.csv", index=False
    )

    scale_rules = {}
    scale_calibration_metrics = []
    for scale in SCALES:
        scale_samples = sample_table[sample_table["scale"] == scale]
        scale_keys = set(scale_samples["sample_key"])
        scale_components = components[components["sample_key"].isin(scale_keys)]
        scale_search = evaluate_grid(
            scale_components, scale_samples, window_sizes, thresholds,
            args.max_recall_loss, (scale,),
        )
        scale_search.to_csv(outdir / f"deployment_threshold_search_{scale}.csv", index=False)
        scale_rules[scale] = rule_from_row(select_rule(scale_search))
        rule = scale_rules[scale]
        scale_calibration_metrics.append(sample_metrics_for_rule(
            components, scale_samples, rule["window_size"],
            rule["score_statistic"], rule["tubularity_threshold"],
        ))
    scale_calibration_metrics_table = pd.concat(scale_calibration_metrics, ignore_index=True)
    scale_calibration_metrics_table.to_csv(
        outdir / "deployment_calibration_metrics_per_sample_scale_specific.csv", index=False
    )
    scale_calibration_summary = summarize_metrics(scale_calibration_metrics_table)
    scale_calibration_summary.to_csv(
        outdir / "deployment_calibration_summary_by_scale_scale_specific.csv", index=False
    )

    coarse_samples = sample_table[sample_table["scale"] == "coarse"]
    coarse_keys = set(coarse_samples["sample_key"])
    coarse_components = components[components["sample_key"].isin(coarse_keys)]
    coarse_search = evaluate_coarse_thresholds(
        coarse_components, coarse_samples,
        shared_rule["window_size"], shared_rule["score_statistic"], thresholds,
        args.coarse_max_pooled_recall_loss,
        args.coarse_max_mean_recall_loss,
        args.coarse_max_sample_recall_loss,
    )
    coarse_search.to_csv(outdir / "coarse_stage2_deployment_threshold_search.csv", index=False)
    coarse_selected = select_coarse_threshold(coarse_search)
    coarse_rule = rule_from_row(coarse_selected)
    coarse_calibration_metrics = sample_metrics_for_rule(
        components, coarse_samples, coarse_rule["window_size"],
        coarse_rule["score_statistic"], coarse_rule["tubularity_threshold"],
    )
    coarse_calibration_metrics.to_csv(
        outdir / "coarse_stage2_deployment_calibration_metrics.csv", index=False
    )
    coarse_config = {
        "schema_version": 3,
        "method": "fixed_window_local_pca_tubularity_coarse_stage2",
        "score": "((lambda1-lambda2)/lambda1) * sqrt(lambda3/lambda2)",
        "rule": coarse_rule,
        "min_foreground_voxels_per_window": args.min_foreground_voxels,
        "keep_components_touching_volume_boundary": True,
        "selection_constraints": {
            "max_pooled_recall_loss": args.coarse_max_pooled_recall_loss,
            "max_mean_sample_recall_loss": args.coarse_max_mean_recall_loss,
            "max_individual_sample_recall_loss": args.coarse_max_sample_recall_loss,
        },
        "geometry_source": "shared all-scale OOF deployment rule",
        "threshold_source": "all five OOF coarse predictions",
        "calibration_cv_run": str(cv_run),
        "notes": [
            "Use coarse_stage2_meta_cv_summary.csv as the unbiased performance estimate.",
            "Do not apply independently to bare inference cores.",
        ],
    }
    with open(outdir / "coarse_deployment_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(coarse_config, handle, indent=2)
    save_filtered_masks(
        cv_run, outdir / "coarse_stage2_deployment_filtered_masks",
        components, coarse_samples, coarse_rule,
    )

    config = {
        "schema_version": 2,
        "method": "fixed_window_local_pca_tubularity",
        "score": "((lambda1-lambda2)/lambda1) * sqrt(lambda3/lambda2)",
        "window_overlap_fraction": 0.5,
        "window_sizes_tested_voxels": window_sizes,
        "min_foreground_voxels_per_window": args.min_foreground_voxels,
        "keep_components_touching_volume_boundary": True,
        "positive_overlap_fraction": args.positive_overlap,
        "negative_overlap_fraction": args.negative_overlap,
        "max_absolute_recall_loss_per_scale": args.max_recall_loss,
        "default_rule": "shared",
        "shared_rule": shared_rule,
        "scale_specific_rules": scale_rules,
        "calibration_cv_run": str(cv_run),
        "n_oof_samples": len(sample_table),
        "notes": [
            "Shared-rule selection constrains recall separately for unscaled, mid, and coarse.",
            "Meta-CV metrics are the unbiased postprocessor performance estimate.",
            "Scale-specific rules are diagnostic alternatives; shared_rule is the default.",
            "Do not apply to bare inference cores; use full volumes or sufficient halo.",
        ],
    }
    with open(outdir / "deployment_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for selected in selected_meta.itertuples(index=False):
        fold_samples = sample_table[sample_table["fold"] == selected.heldout_fold]
        save_filtered_masks(
            cv_run, outdir / "meta_cv_filtered_masks", components, fold_samples,
            {
                "window_size": selected.window_size,
                "score_statistic": selected.score_statistic,
                "tubularity_threshold": selected.tubularity_threshold,
            },
        )
    save_filtered_masks(
        cv_run, outdir / "deployment_filtered_masks", components, sample_table, shared_rule,
    )
    classes = components.drop_duplicates(["sample_key", "component_id"])["target_class"]
    report = [
        "Fixed-window analytical vessel postprocessor",
        "==============================================",
        f"OOF samples: {len(sample_table)}",
        f"Components: {len(classes)}",
        f"Strong-overlap positives: {(classes == 'positive').sum()}",
        f"Low-overlap negatives: {(classes == 'negative').sum()}",
        f"Ambiguous components: {(classes == 'ambiguous').sum()}",
        f"Window sizes tested: {window_sizes}",
        "",
        f"Shared deployment rule: {shared_rule}",
        f"Scale-specific diagnostic rules: {scale_rules}",
        "",
        "Unbiased shared-rule meta-CV:",
        meta_summary.to_string(index=False),
        "",
        "Unbiased scale-specific-rule meta-CV:",
        scale_meta_summary.to_string(index=False),
        "",
        "Unbiased nested coarse stage-2 meta-CV:",
        coarse_stage2_summary.to_string(index=False),
        "",
        f"Final coarse stage-2 deployment rule: {coarse_rule}",
        "",
        "All-OOF shared-rule calibration (not an unbiased performance estimate):",
        shared_calibration_summary.to_string(index=False),
        "",
        "All-OOF scale-specific calibration (not an unbiased performance estimate):",
        scale_calibration_summary.to_string(index=False),
    ]
    (outdir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)
    print(f"\nDeployment JSON: {outdir / 'deployment_thresholds.json'}", flush=True)


def apply_filter(args: argparse.Namespace) -> None:
    with open(args.thresholds, encoding="utf-8") as handle:
        config = json.load(handle)
    schema_version = config.get("schema_version")
    if schema_version == 3:
        rule = config["rule"]
    elif schema_version == 2:
        rule = (
            config["shared_rule"] if args.rule == "shared"
            else config["scale_specific_rules"][args.scale]
        )
    else:
        raise ValueError("Unsupported deployment threshold schema")
    pred = tifffile.imread(args.prediction)
    labels, frame = component_window_features(
        pred, None, [int(rule["window_size"])],
        int(config["min_foreground_voxels_per_window"]),
        float(config.get("positive_overlap_fraction", 0.5)),
        float(config.get("negative_overlap_fraction", 0.1)),
    )
    statistic = rule["score_statistic"]
    threshold = float(rule["tubularity_threshold"])
    keep = (
        (frame[statistic].to_numpy() >= threshold)
        | frame["touches_volume_boundary"].to_numpy(dtype=bool)
    )
    keep_ids = frame.loc[keep, "component_id"].to_numpy(dtype=np.int64)
    filtered = np.isin(labels, keep_ids).astype(np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, filtered)
    frame["kept"] = keep
    frame.to_csv(output.with_suffix(".components.csv"), index=False)
    print(f"Rule: {rule}")
    print(f"Components kept: {int(keep.sum())}/{len(frame)}")
    print(f"Foreground voxels: {np.count_nonzero(pred)} -> {np.count_nonzero(filtered)}")
    print(f"Wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--cv_run_dir", required=True)
    calibration.add_argument("--output_dir", required=True)
    calibration.add_argument("--n_folds", type=int, default=5)
    calibration.add_argument(
        "--window_sizes", type=int, nargs="+", default=[10, 12, 15, 20, 25, 30, 35, 40]
    )
    calibration.add_argument("--min_foreground_voxels", type=int, default=20)
    calibration.add_argument("--positive_overlap", type=float, default=0.5)
    calibration.add_argument("--negative_overlap", type=float, default=0.1)
    calibration.add_argument("--max_recall_loss", type=float, default=0.02)
    calibration.add_argument("--threshold_step", type=float, default=0.025)
    calibration.add_argument("--coarse_max_pooled_recall_loss", type=float, default=0.01)
    calibration.add_argument("--coarse_max_mean_recall_loss", type=float, default=0.01)
    calibration.add_argument("--coarse_max_sample_recall_loss", type=float, default=0.02)
    calibration.set_defaults(function=calibrate)

    application = subparsers.add_parser("apply")
    application.add_argument("--prediction", required=True)
    application.add_argument("--output", required=True)
    application.add_argument("--thresholds", required=True)
    application.add_argument("--scale", choices=SCALES, required=True)
    application.add_argument("--rule", choices=("shared", "scale-specific"), default="shared")
    application.set_defaults(function=apply_filter)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.function(parsed)
