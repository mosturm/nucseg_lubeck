from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCALES = ("unscaled", "mid", "coarse")
METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "pred_gt_ratio",
    "fg_rel_error",
    "abs_fg_rel_error",
    "pred_fg_voxels",
    "gt_fg_voxels",
)
COUNT_METRICS = ("tp", "fp", "fn", "tn")


def read_rows(root: Path, stage: str, n_folds: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(1, n_folds + 1):
        for scale in SCALES:
            path = (
                root
                / f"fold_{fold:02d}"
                / scale
                / f"threshold_sweep_{stage}"
                / "per_sample_metrics_all_settings.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing held-out sweep: {path}")
            with path.open(newline="", encoding="utf-8") as handle:
                fold_rows = list(csv.DictReader(handle))
            if not fold_rows:
                raise ValueError(f"Empty held-out sweep: {path}")
            for row in fold_rows:
                row["fold"] = fold
                row["scale"] = scale
                rows.append(row)
    return rows


def key(row: dict[str, object]) -> tuple[str, float, int, float]:
    return (
        str(row["scale"]),
        float(row["cellprob_threshold"]),
        int(float(row["min_size"])),
        float(row["flow3D_smooth_scalar"]),
    )


def summarize(rows: list[dict[str, object]], n_folds: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)

    summaries: list[dict[str, object]] = []
    samples_by_scale: dict[str, set[str]] = {}
    for (scale, cellprob, min_size, smooth), setting_rows in grouped.items():
        sample_ids = [str(row["sample_id"]) for row in setting_rows]
        folds = {int(row["fold"]) for row in setting_rows}
        if len(sample_ids) != n_folds or len(sample_ids) != len(set(sample_ids)):
            raise AssertionError(
                f"{scale} threshold {cellprob:g}: expected {n_folds} unique held-out samples, "
                f"found {len(sample_ids)}"
            )
        if folds != set(range(1, n_folds + 1)):
            raise AssertionError(f"{scale} threshold {cellprob:g}: incomplete folds {folds}")
        if scale not in samples_by_scale:
            samples_by_scale[scale] = set(sample_ids)
        elif set(sample_ids) != samples_by_scale[scale]:
            raise AssertionError(f"{scale}: held-out sample coverage changed across thresholds")

        summary: dict[str, object] = {
            "scale": scale,
            "cellprob_threshold": cellprob,
            "min_size": min_size,
            "flow3D_smooth_scalar": smooth,
            "n": len(setting_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in setting_rows], dtype=float)
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"median_{metric}"] = float(np.median(values))
            summary[f"std_{metric}"] = float(np.std(values, ddof=1))

        counts = {
            name: int(sum(int(float(row[name])) for row in setting_rows))
            for name in COUNT_METRICS
        }
        summary.update({f"total_{name}": value for name, value in counts.items()})
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        summary["micro_dice"] = 2 * counts["tp"] / denominator if denominator else 0.0
        pred = sum(float(row["pred_fg_voxels"]) for row in setting_rows)
        gt = sum(float(row["gt_fg_voxels"]) for row in setting_rows)
        ratio = pred / gt if gt else 0.0
        summary["pooled_pred_gt_ratio"] = ratio
        summary["abs_log_pooled_pred_gt_ratio"] = (
            abs(math.log(ratio)) if ratio > 0 else float("inf")
        )
        summaries.append(summary)
    return summaries


def select_scale(
    rows: list[dict[str, object]], max_dice_drop: float
) -> tuple[dict[str, object], dict[str, object]]:
    best_dice = min(
        rows,
        key=lambda row: (
            -float(row["mean_dice"]),
            -float(row["mean_iou"]),
            float(row["mean_abs_fg_rel_error"]),
            -float(row["cellprob_threshold"]),
        ),
    )
    best_value = float(best_dice["mean_dice"])
    eligible = []
    for row in rows:
        drop = best_value - float(row["mean_dice"])
        row["dice_drop_from_best"] = drop
        row["within_dice_tolerance"] = int(drop <= max_dice_drop + 1e-12)
        if drop <= max_dice_drop + 1e-12:
            eligible.append(row)
    volume_balanced = min(
        eligible,
        key=lambda row: (
            float(row["abs_log_pooled_pred_gt_ratio"]),
            float(row["mean_abs_fg_rel_error"]),
            -float(row["mean_dice"]),
            -float(row["cellprob_threshold"]),
        ),
    )
    return best_dice, volume_balanced


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool cross-fitted vessel threshold sweeps separately by scale"
    )
    parser.add_argument("--calibration_root", type=Path, required=True)
    parser.add_argument("--stage", default="volume_balance")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_dice_drop", type=float, default=0.01)
    args = parser.parse_args()
    if args.max_dice_drop < 0:
        raise ValueError("--max_dice_drop must be non-negative")

    root = args.calibration_root.resolve()
    rows = read_rows(root, args.stage, args.n_folds)
    summaries = summarize(rows, args.n_folds)
    selected: dict[str, object] = {}
    report = ["Vessel deployment threshold calibration", ""]
    for scale in SCALES:
        scale_rows = [row for row in summaries if row["scale"] == scale]
        best_dice, volume_balanced = select_scale(scale_rows, args.max_dice_drop)
        selected[scale] = {
            "selection_scope": f"{args.n_folds} unique cross-fitted held-out {scale} volumes",
            "best_dice": best_dice,
            "volume_balanced": volume_balanced,
            "best": volume_balanced,
        }
        report.extend(
            [
                f"[{scale}]",
                f"Best-Dice threshold: {float(best_dice['cellprob_threshold']):g}",
                f"Best mean Dice: {float(best_dice['mean_dice']):.6f}",
                f"Best pooled pred/GT ratio: {float(best_dice['pooled_pred_gt_ratio']):.6f}",
                f"Selected volume-balanced threshold: {float(volume_balanced['cellprob_threshold']):g}",
                f"Selected mean Dice: {float(volume_balanced['mean_dice']):.6f}",
                f"Dice drop: {float(volume_balanced['dice_drop_from_best']):.6f}",
                f"Selected pooled pred/GT ratio: {float(volume_balanced['pooled_pred_gt_ratio']):.6f}",
                "",
            ]
        )

    summaries.sort(key=lambda row: (SCALES.index(str(row["scale"])), -float(row["cellprob_threshold"])))
    prefix = root / f"pooled_{args.stage}"
    write_csv(prefix.with_name(prefix.name + "_per_sample_all_thresholds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_threshold_summary_by_scale.csv"), summaries)
    prefix.with_name(prefix.name + "_best_by_scale.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_mode": "volume_balanced_within_scale",
                "max_absolute_macro_dice_drop": args.max_dice_drop,
                "selection_rule": (
                    "Within each scale, retain thresholds within the Dice tolerance and "
                    "minimize absolute log pooled predicted/GT volume ratio"
                ),
                "scales": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = prefix.with_name(prefix.name + "_report.txt")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
