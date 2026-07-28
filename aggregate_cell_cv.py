from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def load_rows(cv_run_dir: Path, n_folds: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(1, n_folds + 1):
        eval_dir = cv_run_dir / f"fold_{fold:02d}" / "inference_eval"
        metrics_path = eval_dir / "metrics_per_sample.csv"
        meta_path = eval_dir / "run_meta.json"
        if not metrics_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing inference outputs in {eval_dir}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))
        with meta_path.open(encoding="utf-8") as handle:
            run_meta = json.load(handle)
        for metric_row in metric_rows:
            row: dict[str, object] = {
                "fold": fold,
                "model": run_meta["model"],
                "cellprob_threshold": float(run_meta["cellprob_threshold"]),
                "min_size": int(run_meta["min_size"]),
            }
            row.update(metric_row)
            rows.append(row)

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != 12:
        raise AssertionError(f"Expected 12 out-of-fold test rows, found {len(sample_ids)}")
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("Cell test samples are not unique across folds")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "fold",
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
    available = list(rows[0])
    fieldnames = [name for name in preferred if name in available]
    fieldnames.extend(name for name in available if name not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "mean", "median", "std", "min", "max", "n"]
        )
        writer.writeheader()
        writer.writerows(rows)


def make_violin_plot(cv_run_dir: Path, rows: list[dict[str, object]]) -> None:
    values = np.asarray([float(row["dice"]) for row in rows])
    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    violin = axis.violinplot(
        [values], positions=[1], widths=0.65, showmedians=True, showextrema=True
    )
    for body in violin["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#222222")
        body.set_alpha(0.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        violin[key].set_color("#222222")
        violin[key].set_linewidth(1.2)
    axis.scatter(
        1 + np.linspace(-0.12, 0.12, len(values)),
        values,
        s=38,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axis.set_xticks([1], ["Cells"])
    axis.set_ylabel("Test Dice")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Out-of-fold cell segmentation performance")
    axis.text(
        0.98,
        0.02,
        "n=12 independent test volumes",
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
    fig.savefig(cv_run_dir / "test_dice_violin.png", dpi=300)
    fig.savefig(cv_run_dir / "test_dice_violin.pdf")
    plt.close(fig)


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> None:
    dice = next(row for row in summary if row["metric"] == "dice")
    lines = [
        "Cell five-fold cross-validation report",
        "",
        (
            f"Dice: n={dice['n']}, mean={dice['mean']:.6f}, "
            f"median={dice['median']:.6f}, std={dice['std']:.6f}, "
            f"min={dice['min']:.6f}, max={dice['max']:.6f}"
        ),
        "",
        "fold\tsample_id\tcellprob_threshold\tdice",
    ]
    for row in rows:
        lines.append(
            f"{row['fold']}\t{row['sample_id']}\t"
            f"{float(row['cellprob_threshold']):g}\t{float(row['dice']):.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cell CV test metrics")
    parser.add_argument("--cv_run_dir", type=Path, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    cv_run_dir = args.cv_run_dir.resolve()
    if not cv_run_dir.is_dir():
        raise FileNotFoundError(f"CV run directory not found: {cv_run_dir}")
    rows = load_rows(cv_run_dir, args.n_folds)
    summary = summarize(rows)
    write_csv(cv_run_dir / "test_metrics_all.csv", rows)
    write_summary(cv_run_dir / "test_metrics_summary.csv", summary)
    make_violin_plot(cv_run_dir, rows)
    write_report(cv_run_dir / "cv_report.txt", rows, summary)
    print(f"Wrote aggregate CV outputs to {cv_run_dir}")


if __name__ == "__main__":
    main()
