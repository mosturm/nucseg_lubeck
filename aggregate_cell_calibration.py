from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def read_stage_rows(calibration_root: Path, stage: str, n_folds: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(1, n_folds + 1):
        path = (
            calibration_root
            / f"fold_{fold:02d}"
            / f"threshold_sweep_{stage}"
            / "per_sample_metrics_all_settings.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold sweep output: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            fold_rows = list(csv.DictReader(handle))
        if not fold_rows:
            raise ValueError(f"No sweep rows in {path}")
        for row in fold_rows:
            row["fold"] = fold
            rows.append(row)
    return rows


def setting_key(row: dict[str, object]) -> tuple[float, int, float]:
    return (
        float(row["cellprob_threshold"]),
        int(float(row["min_size"])),
        float(row["flow3D_smooth_scalar"]),
    )


def summarize(rows: list[dict[str, object]], n_folds: int) -> list[dict[str, object]]:
    grouped: dict[tuple[float, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[setting_key(row)].append(row)

    expected_samples: set[str] | None = None
    summaries: list[dict[str, object]] = []
    for (cellprob, min_size, smooth), setting_rows in grouped.items():
        sample_ids = [str(row["sample_id"]) for row in setting_rows]
        folds = {int(row["fold"]) for row in setting_rows}
        if len(sample_ids) != 12 or len(sample_ids) != len(set(sample_ids)):
            raise AssertionError(
                f"Threshold {cellprob:g}: expected 12 unique held-out samples, got {len(sample_ids)} rows"
            )
        if folds != set(range(1, n_folds + 1)):
            raise AssertionError(f"Threshold {cellprob:g}: incomplete fold coverage {sorted(folds)}")
        if expected_samples is None:
            expected_samples = set(sample_ids)
        elif set(sample_ids) != expected_samples:
            raise AssertionError(f"Threshold {cellprob:g}: held-out sample coverage changed")

        summary: dict[str, object] = {
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
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        summary.update({f"total_{name}": value for name, value in counts.items()})
        summary["micro_dice"] = 2 * counts["tp"] / denominator if denominator else 0.0
        total_pred = float(summary["mean_pred_fg_voxels"]) * len(setting_rows)
        total_gt = float(summary["mean_gt_fg_voxels"]) * len(setting_rows)
        pooled_ratio = total_pred / total_gt if total_gt else 0.0
        summary["pooled_pred_gt_ratio"] = pooled_ratio
        summary["abs_log_pooled_pred_gt_ratio"] = (
            abs(math.log(pooled_ratio)) if pooled_ratio > 0 else float("inf")
        )
        summaries.append(summary)

    return summaries


def select_settings(
    summaries: list[dict[str, object]], max_dice_drop: float
) -> tuple[dict[str, object], dict[str, object]]:
    best_dice = min(
        summaries,
        key=lambda row: (
            -float(row["mean_dice"]),
            -float(row["mean_iou"]),
            float(row["mean_abs_fg_rel_error"]),
            -float(row["cellprob_threshold"]),
        ),
    )
    best_dice_value = float(best_dice["mean_dice"])
    eligible = [
        row
        for row in summaries
        if best_dice_value - float(row["mean_dice"]) <= max_dice_drop + 1e-12
    ]
    volume_balanced = min(
        eligible,
        key=lambda row: (
            float(row["abs_log_pooled_pred_gt_ratio"]),
            float(row["mean_abs_fg_rel_error"]),
            -float(row["mean_dice"]),
            -float(row["cellprob_threshold"]),
        ),
    )
    for row in summaries:
        dice_drop = best_dice_value - float(row["mean_dice"])
        row["dice_drop_from_best"] = dice_drop
        row["within_dice_tolerance"] = int(dice_drop <= max_dice_drop + 1e-12)
    summaries.sort(key=lambda row: -float(row["cellprob_threshold"]))
    return best_dice, volume_balanced


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool cross-fitted cell threshold sweeps over all held-out volumes"
    )
    parser.add_argument("--calibration_root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--selection_mode",
        choices=("dice", "volume_balanced"),
        default="dice",
    )
    parser.add_argument("--max_dice_drop", type=float, default=0.01)
    args = parser.parse_args()

    root = args.calibration_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration root not found: {root}")

    rows = read_stage_rows(root, args.stage, args.n_folds)
    summaries = summarize(rows, args.n_folds)
    if args.max_dice_drop < 0:
        raise ValueError("--max_dice_drop must be non-negative")
    best_dice, volume_balanced = select_settings(summaries, args.max_dice_drop)
    best = best_dice if args.selection_mode == "dice" else volume_balanced
    prefix = root / f"pooled_{args.stage}"

    write_csv(prefix.with_name(prefix.name + "_per_sample_all_thresholds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_threshold_summary.csv"), summaries)
    best_path = prefix.with_name(prefix.name + "_best.json")
    best_path.write_text(
        json.dumps(
            {
                "selection_scope": "12 unique cross-fitted held-out volumes",
                "selection_mode": args.selection_mode,
                "max_absolute_macro_dice_drop": args.max_dice_drop,
                "selection_rule": (
                    "maximize macro mean Dice"
                    if args.selection_mode == "dice"
                    else "among thresholds within the Dice tolerance, minimize absolute log pooled predicted/GT volume ratio"
                ),
                "best_dice": best_dice,
                "volume_balanced": volume_balanced,
                "best": best,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = prefix.with_name(prefix.name + "_report.txt")
    report_path.write_text(
        "\n".join(
            [
                f"stage: {args.stage}",
                "selection scope: 12 unique cross-fitted held-out volumes",
                f"selection mode: {args.selection_mode}",
                f"maximum allowed absolute macro Dice drop: {args.max_dice_drop:.6f}",
                "",
                "Best-Dice setting:",
                f"cellprob_threshold: {float(best_dice['cellprob_threshold']):g}",
                f"mean_dice: {float(best_dice['mean_dice']):.6f}",
                f"pooled_pred_gt_ratio: {float(best_dice['pooled_pred_gt_ratio']):.6f}",
                f"mean_abs_fg_rel_error: {float(best_dice['mean_abs_fg_rel_error']):.6f}",
                "",
                "Volume-balanced setting within Dice tolerance:",
                f"cellprob_threshold: {float(volume_balanced['cellprob_threshold']):g}",
                f"mean_dice: {float(volume_balanced['mean_dice']):.6f}",
                f"dice_drop_from_best: {float(volume_balanced['dice_drop_from_best']):.6f}",
                f"pooled_pred_gt_ratio: {float(volume_balanced['pooled_pred_gt_ratio']):.6f}",
                f"mean_abs_fg_rel_error: {float(volume_balanced['mean_abs_fg_rel_error']):.6f}",
                "",
                "Selected deployment setting:",
                f"cellprob_threshold: {float(best['cellprob_threshold']):g}",
                f"min_size: {int(best['min_size'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
