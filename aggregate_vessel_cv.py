from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCALES = ("unscaled", "mid", "coarse")
METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "pred_gt_ratio",
    "pred_fg_voxels",
    "gt_fg_voxels",
)


def read_single_test_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing test metrics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one test ROI in {path}, found {len(rows)}")
    return rows[0]


def load_rows(cv_run_dir: Path, n_folds: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for fold in range(1, n_folds + 1):
        for scale in SCALES:
            eval_dir = cv_run_dir / f"fold_{fold:02d}" / scale / "inference_eval"
            metric_row = read_single_test_row(eval_dir / "metrics_per_sample.csv")
            meta_path = eval_dir / "run_meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"Missing inference metadata: {meta_path}")
            with meta_path.open(encoding="utf-8") as handle:
                run_meta = json.load(handle)

            row: dict[str, object] = {
                "fold": fold,
                "scale": scale,
                "model": run_meta["model"],
                "cellprob_threshold": float(run_meta["cellprob_threshold"]),
                "min_size": int(run_meta["min_size"]),
            }
            row.update(metric_row)
            rows.append(row)

    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        if len(scale_rows) != n_folds:
            raise AssertionError(f"Expected {n_folds} {scale} rows, found {len(scale_rows)}")
        sample_ids = [str(row["sample_id"]) for row in scale_rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise AssertionError(f"{scale} test samples are not unique across folds")

    return rows


def write_all_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "fold",
        "scale",
        "sample_id",
        "model",
        "cellprob_threshold",
        "min_size",
        *METRICS,
        "tp",
        "fp",
        "fn",
        "tn",
        "shape_z",
        "shape_y",
        "shape_x",
    ]
    available = list(rows[0].keys())
    fieldnames = [name for name in preferred if name in available]
    fieldnames.extend(name for name in available if name not in fieldnames)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in scale_rows], dtype=float)
            summary_rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": int(len(values)),
                }
            )
    return summary_rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["scale", "metric", "mean", "median", "std", "min", "max", "n"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_violin_plot(cv_run_dir: Path, rows: list[dict[str, object]]) -> None:
    values = [
        np.asarray([float(row["dice"]) for row in rows if row["scale"] == scale])
        for scale in SCALES
    ]
    colors = ("#4C78A8", "#59A14F", "#E15759")
    positions = np.arange(1, len(SCALES) + 1)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    violins = axis.violinplot(
        values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        violins[key].set_color("#222222")
        violins[key].set_linewidth(1.2)

    for position, scale_values, color in zip(positions, values, colors):
        offsets = np.linspace(-0.09, 0.09, len(scale_values))
        axis.scatter(
            position + offsets,
            scale_values,
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    axis.set_xticks(positions, ["Unscaled", "Mid", "Coarse"])
    axis.set_ylabel("Test Dice")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Out-of-fold vessel segmentation performance")
    axis.text(
        0.99,
        0.02,
        "n=5 independent test ROIs per scale",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(cv_run_dir / "test_dice_violin_by_scale.png", dpi=300)
    fig.savefig(cv_run_dir / "test_dice_violin_by_scale.pdf")
    plt.close(fig)


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    lines = [
        "Vessel five-fold cross-validation report",
        "",
        "Test Dice by scale",
        "scale\tn\tmean\tmedian\tstd\tmin\tmax",
    ]

    for scale in SCALES:
        summary = next(
            row for row in summary_rows if row["scale"] == scale and row["metric"] == "dice"
        )
        lines.append(
            f"{scale}\t{summary['n']}\t{summary['mean']:.6f}\t"
            f"{summary['median']:.6f}\t{summary['std']:.6f}\t"
            f"{summary['min']:.6f}\t{summary['max']:.6f}"
        )

    lines.extend(["", "Selected validation thresholds and test samples"])
    lines.append("fold\tscale\tsample_id\tcellprob_threshold\tdice")
    for row in rows:
        lines.append(
            f"{row['fold']}\t{row['scale']}\t{row['sample_id']}\t"
            f"{float(row['cellprob_threshold']):g}\t{float(row['dice']):.6f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate scale-specific vessel cross-validation test metrics"
    )
    parser.add_argument("--cv_run_dir", type=Path, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    cv_run_dir = args.cv_run_dir.resolve()
    if not cv_run_dir.is_dir():
        raise FileNotFoundError(f"CV run directory not found: {cv_run_dir}")

    rows = load_rows(cv_run_dir, args.n_folds)
    summary_rows = summarize(rows)

    write_all_metrics(cv_run_dir / "test_metrics_all_scales.csv", rows)
    write_summary(cv_run_dir / "test_metrics_summary_by_scale.csv", summary_rows)
    make_violin_plot(cv_run_dir, rows)
    write_report(cv_run_dir / "cv_report.txt", rows, summary_rows)

    print(f"Wrote aggregate CV outputs to {cv_run_dir}")
    for scale in SCALES:
        dice = next(
            row for row in summary_rows if row["scale"] == scale and row["metric"] == "dice"
        )
        print(
            f"{scale}: Dice mean={dice['mean']:.4f}, "
            f"median={dice['median']:.4f}, std={dice['std']:.4f}, n={dice['n']}"
        )


if __name__ == "__main__":
    main()
