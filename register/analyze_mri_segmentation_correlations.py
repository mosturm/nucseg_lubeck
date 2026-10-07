#!/usr/bin/env python3
"""Create correlation tables, plots, and spatially blocked linear-model estimates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PREDICTORS = {
    "sandi": ["De", "Din", "Rsoma", "fextra", "fneurite", "fsoma"],
    "mpm": ["R1", "R2s", "A"],
    "r2s": ["R2s"],
}
RADIUS_TARGETS = [
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r7_r3_um",
    "cell_radius_effective_r5_r3_um",
]
TARGETS = [
    "f_soma_seg",
    "cell_density_per_mm3",
    *RADIUS_TARGETS,
    "f_vessel_seg",
]
PRIMARY_TARGETS = [
    "f_soma_seg",
    "cell_density_per_mm3",
    "cell_radius_effective_r7_r3_um",
    "f_vessel_seg",
]


def finite_pair(frame: pd.DataFrame, x: str, y: str, min_cells_radius: int) -> pd.DataFrame:
    columns = [x, y, "mri_i", "mri_j", "mri_k", "cell_count"]
    selected = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if y in RADIUS_TARGETS:
        selected = selected[selected["cell_count"] >= min_cells_radius]
    if y in ("f_soma_seg", "f_vessel_seg"):
        selected = selected[selected[y].between(0.0, 1.0)]
    return selected


def safe_correlation(function, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    result = function(x, y)
    return float(result.statistic), float(result.pvalue)


def bh_adjust(pvalues: pd.Series) -> np.ndarray:
    values = pvalues.to_numpy(float)
    output = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return output
    order = finite_indices[np.argsort(values[finite_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(ranked, 1.0)
    return output


def spatial_groups(frame: pd.DataFrame, block_size: int) -> np.ndarray:
    coordinates = frame[["mri_i", "mri_j", "mri_k"]].to_numpy(np.int64)
    blocks = np.floor_divide(coordinates, block_size)
    return np.array([f"{i}:{j}:{k}" for i, j, k in blocks], dtype=object)


def balanced_group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < folds:
        raise ValueError(f"Only {len(unique)} spatial groups for {folds} folds")
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(unique))
    order = np.lexsort((tie_break, -counts))
    loads = np.zeros(folds, dtype=int)
    assignments = np.zeros(len(unique), dtype=int)
    for group_index in order:
        fold = int(np.argmin(loads))
        assignments[group_index] = fold
        loads[fold] += counts[group_index]
    return assignments[inverse]


def regression_and_cv(
    selected: pd.DataFrame,
    predictor: str,
    target: str,
    block_size: int,
    folds: int,
    seed: int,
) -> dict[str, float | int]:
    x = selected[predictor].to_numpy(float)
    y = selected[target].to_numpy(float)
    pearson, pearson_p = safe_correlation(stats.pearsonr, x, y)
    spearman, spearman_p = safe_correlation(stats.spearmanr, x, y)
    if len(x) < 3 or np.ptp(x) == 0:
        return {
            "n": int(len(x)),
            "pearson_r": pearson,
            "pearson_p": pearson_p,
            "spearman_rho": spearman,
            "spearman_p": spearman_p,
        }
    fit = stats.linregress(x, y)
    groups = spatial_groups(selected, block_size)
    fold_ids = balanced_group_folds(groups, folds, seed)
    prediction = np.full(len(y), np.nan)
    for fold in range(folds):
        train = fold_ids != fold
        test = ~train
        if train.sum() < 3 or np.ptp(x[train]) == 0:
            prediction[test] = np.median(y[train])
        else:
            train_fit = stats.linregress(x[train], y[train])
            prediction[test] = train_fit.intercept + train_fit.slope * x[test]
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    cv_r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else math.nan
    cv_pearson, _ = safe_correlation(stats.pearsonr, y, prediction)
    cv_spearman, _ = safe_correlation(stats.spearmanr, y, prediction)
    return {
        "n": int(len(x)),
        "n_spatial_groups": int(len(np.unique(groups))),
        "linear_slope": float(fit.slope),
        "linear_slope_stderr": float(fit.stderr),
        "linear_slope_ci95_low": float(fit.slope - 1.96 * fit.stderr),
        "linear_slope_ci95_high": float(fit.slope + 1.96 * fit.stderr),
        "linear_intercept": float(fit.intercept),
        "linear_r2_in_sample": float(fit.rvalue**2),
        "linear_p": float(fit.pvalue),
        "pearson_r": pearson,
        "pearson_p": pearson_p,
        "spearman_rho": spearman,
        "spearman_p": spearman_p,
        "spatial_cv_r2": cv_r2,
        "spatial_cv_mae": float(np.mean(np.abs(residual))),
        "spatial_cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "spatial_cv_pearson": cv_pearson,
        "spatial_cv_spearman": cv_spearman,
    }


def plot_pair(
    selected: pd.DataFrame,
    predictor: str,
    target: str,
    result: dict[str, object],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = selected[predictor].to_numpy(float)
    y = selected[target].to_numpy(float)
    figure, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    if len(x) > 3000:
        artist = axis.hexbin(x, y, gridsize=45, mincnt=1, bins="log", cmap="viridis")
        figure.colorbar(artist, ax=axis, label="log10 count")
    else:
        axis.scatter(x, y, s=13, alpha=0.38, edgecolors="none", color="#176B87")
    if np.ptp(x) > 0 and "linear_slope" in result:
        line_x = np.linspace(np.min(x), np.max(x), 200)
        line_y = result["linear_intercept"] + result["linear_slope"] * line_x
        axis.plot(line_x, line_y, color="#C73E1D", linewidth=2)
    axis.set_xlabel(predictor)
    axis.set_ylabel(target)
    axis.set_title(f"{predictor} vs {target}")
    text = (
        f"n={len(x):,}\n"
        f"Spearman rho={result.get('spearman_rho', math.nan):.3f}\n"
        f"Linear R2={result.get('linear_r2_in_sample', math.nan):.3f}\n"
        f"Spatial-CV R2={result.get('spatial_cv_r2', math.nan):.3f}"
    )
    axis.text(
        0.03,
        0.97,
        text,
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
    )
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_coverage(frame: pd.DataFrame, modality: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = frame[["coverage_percent", "reconstructed_bounds_coverage_percent"]].dropna()
    figure, axis = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    axis.hexbin(
        selected["coverage_percent"],
        selected["reconstructed_bounds_coverage_percent"],
        gridsize=35,
        mincnt=1,
        bins="log",
        cmap="magma",
    )
    axis.plot([0, 100], [0, 100], color="white", linewidth=1.5, linestyle="--")
    axis.set(xlabel="Collaborator coverage (%)", ylabel="Reconstructed bounds coverage (%)")
    axis.set_title(f"{modality.upper()} coverage QC")
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def should_plot(modality: str, predictor: str, target: str) -> bool:
    if modality == "sandi":
        return (predictor == "Rsoma" and target in RADIUS_TARGETS) or (
            predictor == "fsoma" and target in ("f_soma_seg", "cell_density_per_mm3")
        )
    return target in PRIMARY_TARGETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-min", type=float, default=25.0)
    parser.add_argument("--min-cells-radius", type=int, default=5)
    parser.add_argument("--spatial-block-size", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--near-zero-abs-spearman",
        type=float,
        default=0.10,
        help="Save and plot associations with absolute Spearman rho at or below this value",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    near_zero_plot_dir = plot_dir / "near_zero_associations"
    near_zero_plot_dir.mkdir(exist_ok=True)
    all_results: list[dict[str, object]] = []
    modality_summaries: dict[str, object] = {}
    for modality, predictors in PREDICTORS.items():
        matches = list(args.features_dir.glob(f"hippo002_{modality}_segmentation_features.csv.gz"))
        if not matches:
            continue
        frame = pd.read_csv(matches[0], low_memory=False)
        eligible = frame[
            (frame["local_basis_valid"] == 1)
            & (frame["coverage_percent"] >= args.coverage_min)
            & (frame["covered_footprint_volume_um3"] > 0)
        ].copy()
        plot_coverage(frame, modality, plot_dir / f"{modality}_coverage_qc.png")
        for predictor in predictors:
            for target in TARGETS:
                selected = finite_pair(eligible, predictor, target, args.min_cells_radius)
                if len(selected) < max(20, args.folds * 3):
                    continue
                try:
                    result = regression_and_cv(
                        selected,
                        predictor,
                        target,
                        args.spatial_block_size,
                        args.folds,
                        args.seed,
                    )
                except ValueError:
                    continue
                record = {"modality": modality, "predictor": predictor, "target": target, **result}
                all_results.append(record)
                if should_plot(modality, predictor, target):
                    plot_pair(
                        selected,
                        predictor,
                        target,
                        result,
                        plot_dir / f"{modality}_{predictor}_vs_{target}.png",
                    )
                if abs(float(result.get("spearman_rho", math.nan))) <= args.near_zero_abs_spearman:
                    plot_pair(
                        selected,
                        predictor,
                        target,
                        result,
                        near_zero_plot_dir / f"{modality}_{predictor}_vs_{target}.png",
                    )
        modality_summaries[modality] = {
            "input_rows": int(len(frame)),
            "eligible_rows": int(len(eligible)),
            "coverage_min": args.coverage_min,
        }

    results = pd.DataFrame(all_results)
    if results.empty:
        raise RuntimeError("No eligible regression pairs were produced")
    for column in ("linear_p", "pearson_p", "spearman_p"):
        results[f"{column}_fdr_bh"] = bh_adjust(results[column])
    results = results.sort_values(["modality", "target", "predictor"])
    results.to_csv(args.output_dir / "all_univariate_correlations_and_spatial_cv.csv", index=False)
    near_zero = results[
        results["spearman_rho"].abs() <= args.near_zero_abs_spearman
    ].sort_values("spearman_rho", key=lambda values: values.abs())
    near_zero.to_csv(
        args.output_dir / "near_zero_associations_abs_spearman_le_0p10.csv", index=False
    )
    non_significant = results[
        results["spearman_p_fdr_bh"].isna() | (results["spearman_p_fdr_bh"] >= 0.05)
    ].sort_values("spearman_p_fdr_bh", ascending=False)
    non_significant.to_csv(
        args.output_dir / "non_significant_associations_spearman_fdr_ge_0p05.csv", index=False
    )
    for modality in results["modality"].unique():
        results[results["modality"] == modality].to_csv(
            args.output_dir / f"{modality}_correlations_and_spatial_cv.csv", index=False
        )
    radius_comparison = results[
        (results["modality"] == "sandi")
        & (results["predictor"] == "Rsoma")
        & (results["target"].isin(RADIUS_TARGETS))
    ].sort_values("spearman_rho", ascending=False)
    radius_comparison.to_csv(args.output_dir / "sandi_rsoma_radius_definition_comparison.csv", index=False)
    report = {
        "schema_version": 1,
        "coverage_min": args.coverage_min,
        "min_cells_for_radius_statistics": args.min_cells_radius,
        "spatial_block_size_in_native_mri_voxels": args.spatial_block_size,
        "folds": args.folds,
        "seed": args.seed,
        "near_zero_absolute_spearman_threshold": args.near_zero_abs_spearman,
        "n_near_zero_associations": int(len(near_zero)),
        "n_non_significant_spearman_fdr": int(len(non_significant)),
        "modalities": modality_summaries,
        "n_regressions": int(len(results)),
        "interpretation_warning": (
            "Spatial CV estimates interpolation within Hippo002. It is not an independent "
            "specimen-level test, and p-values do not account fully for residual spatial autocorrelation."
        ),
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
