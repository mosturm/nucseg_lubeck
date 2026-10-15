#!/usr/bin/env python3
"""Relate MPM R1 and r23 to empirical per-voxel soma-radius distributions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


R23_COLUMN = "cell_surface_to_volume_radius_inverse_um_inv"


def parse_quantiles(value: str) -> list[float]:
    quantiles = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not quantiles or any(q <= 0 or q >= 1 for q in quantiles):
        raise argparse.ArgumentTypeError("Quantiles must be comma-separated values between 0 and 1")
    return quantiles


def summarize_radius_group(values: pd.Series) -> pd.Series:
    radii = values.to_numpy(float)
    q25, median, q75, p90, p95 = np.quantile(radii, [0.25, 0.5, 0.75, 0.9, 0.95])
    return pd.Series(
        {
            "distribution_cell_count": len(radii),
            "radius_mean_um": float(np.mean(radii)),
            "radius_std_um": float(np.std(radii, ddof=1)) if len(radii) > 1 else math.nan,
            "radius_median_um_empirical": float(median),
            "radius_iqr_um": float(q75 - q25),
            "radius_p90_um": float(p90),
            "radius_p95_um": float(p95),
            "radius_max_um": float(np.max(radii)),
            "radius_skewness": float(stats.skew(radii, bias=False)) if len(radii) > 2 else math.nan,
            "radius_tail_ratio_p95_median": float(p95 / median) if median > 0 else math.nan,
        }
    )


def choose_representative_rows(
    frame: pd.DataFrame,
    quantiles: list[float],
    examples_per_quantile: int,
) -> pd.DataFrame:
    chosen: list[int] = []
    records: list[dict[str, object]] = []
    r23 = frame[R23_COLUMN].to_numpy(float)
    category_names = ["low", "middle", "high"] if len(quantiles) == 3 else [
        f"q{quantile:g}" for quantile in quantiles
    ]
    for category, quantile in zip(category_names, quantiles):
        target = float(np.quantile(r23, quantile))
        order = np.argsort(np.abs(r23 - target))
        available = [int(index) for index in order if int(index) not in chosen]
        if len(available) < examples_per_quantile:
            raise RuntimeError("Could not select enough distinct representative MRI voxels")
        for rank, selected in enumerate(available[:examples_per_quantile], start=1):
            chosen.append(selected)
            record = frame.iloc[selected].to_dict()
            record.update(
                {
                    "selection_label": f"{category.capitalize()} {rank}",
                    "selection_category": category,
                    "selection_rank": rank,
                    "target_r23_quantile": quantile,
                    "target_r23_value": target,
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def fit_gev(radii: np.ndarray) -> dict[str, float]:
    if len(radii) < 10 or np.ptp(radii) == 0:
        return {"gev_shape_scipy": math.nan, "gev_location_um": math.nan, "gev_scale_um": math.nan}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, location, scale = stats.genextreme.fit(radii)
        if not np.isfinite([shape, location, scale]).all() or scale <= 0:
            raise ValueError("invalid fitted parameters")
        return {
            "gev_shape_scipy": float(shape),
            "gev_location_um": float(location),
            "gev_scale_um": float(scale),
        }
    except Exception:
        return {"gev_shape_scipy": math.nan, "gev_location_um": math.nan, "gev_scale_um": math.nan}


def density_panel(axis, radii: np.ndarray, row: pd.Series, color: str) -> dict[str, float]:
    bins = np.histogram_bin_edges(radii, bins="fd")
    if len(bins) < 6:
        bins = np.linspace(np.min(radii), np.max(radii), 8)
    axis.hist(radii, bins=bins, density=True, color=color, alpha=0.28, edgecolor="white")
    x = np.linspace(max(0.0, np.min(radii) * 0.85), np.max(radii) * 1.08, 500)
    if len(radii) >= 3 and np.ptp(radii) > 0:
        kde = stats.gaussian_kde(radii)
        axis.plot(x, kde(x), color=color, linewidth=2.4, label="empirical KDE")
    gev = fit_gev(radii)
    if np.isfinite(gev["gev_scale_um"]):
        density = stats.genextreme.pdf(
            x,
            gev["gev_shape_scipy"],
            loc=gev["gev_location_um"],
            scale=gev["gev_scale_um"],
        )
        density[~np.isfinite(density)] = np.nan
        axis.plot(x, density, color="#333333", linestyle="--", linewidth=1.7, label="GEV fit")
    axis.axvline(row["radius_mean_um"], color=color, linestyle=":", linewidth=2)
    axis.set_xlim(0.0, 11.0)
    axis.set_xlabel("Sphere-equivalent soma radius (um)")
    axis.set_ylabel("Density")
    axis.set_title(
        f"{row['selection_label']}: r23={row[R23_COLUMN]:.3f} 1/um, "
        f"R1={row['R1']:.3f}\n"
        f"n={int(row['distribution_cell_count'])}, mean={row['radius_mean_um']:.2f}, "
        f"SD={row['radius_std_um']:.2f}, p95/median={row['radius_tail_ratio_p95_median']:.2f}"
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=9)
    return gev


def add_association(axis, x: np.ndarray, y: np.ndarray, xlabel: str) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    rho, pvalue = stats.spearmanr(x, y)
    fit = stats.linregress(x, y)
    axis.scatter(x, y, s=16, alpha=0.42, edgecolors="none", color="#287271")
    line_x = np.linspace(np.min(x), np.max(x), 250)
    axis.plot(line_x, fit.intercept + fit.slope * line_x, color="#C73E1D", linewidth=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"$r_{23}=\sum r^2/\sum r^3$  [$\mu m^{-1}$]")
    axis.text(
        0.04,
        0.96,
        f"n={len(x):,}\nSpearman rho={rho:.3f}\nLinear R2={fit.rvalue**2:.3f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#777777"},
    )
    axis.grid(alpha=0.18)
    return {"n": len(x), "spearman_rho": float(rho), "spearman_p": float(pvalue), "linear_r2": float(fit.rvalue**2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-mpm", type=Path, required=True)
    parser.add_argument("--per-cell-radii", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--quantiles", type=parse_quantiles, default=parse_quantiles("0.10,0.50,0.90"))
    parser.add_argument("--examples-per-quantile", type=int, default=3)
    args = parser.parse_args()

    if args.examples_per_quantile < 1:
        raise ValueError("--examples-per-quantile must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw_mpm, low_memory=False)
    radii = pd.read_csv(args.per_cell_radii, low_memory=False)
    required_raw = {"mri_table_row", "R1", R23_COLUMN, "cell_count", "included_in_base_analysis"}
    missing = sorted(required_raw - set(raw.columns))
    if missing:
        raise ValueError(f"Raw MPM table is missing columns: {missing}")
    if not {"mri_table_row", "cell_radius_um"}.issubset(radii.columns):
        raise ValueError("Per-cell radius table lacks mri_table_row or cell_radius_um")

    summaries = (
        radii.groupby("mri_table_row", sort=False)["cell_radius_um"]
        .apply(summarize_radius_group)
        .unstack()
        .reset_index()
    )
    merged = raw.merge(summaries, on="mri_table_row", how="left", validate="one_to_one")
    eligible = merged[
        (merged["included_in_base_analysis"] == 1)
        & (merged["distribution_cell_count"] >= args.min_cells)
        & np.isfinite(merged["R1"])
        & np.isfinite(merged[R23_COLUMN])
    ].copy()
    required_representatives = len(args.quantiles) * args.examples_per_quantile
    if len(eligible) < max(10, required_representatives):
        raise RuntimeError(f"Only {len(eligible)} MPM voxels have at least {args.min_cells} cells")
    eligible.to_csv(args.output_dir / "mpm_r23_radius_distribution_summary.csv", index=False)

    selected = choose_representative_rows(
        eligible.reset_index(drop=True), args.quantiles, args.examples_per_quantile
    )
    category_colors = {"low": "#2A9D8F", "middle": "#D99B2B", "high": "#D1495B"}
    colors = [category_colors.get(str(category), "#6A4C93") for category in selected["selection_category"]]

    overview_figure, scatter_axis = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    scatter_axis.scatter(
        eligible[R23_COLUMN], eligible["R1"], s=18, alpha=0.32, edgecolors="none", color="#778899"
    )
    scatter_fit = stats.linregress(eligible[R23_COLUMN], eligible["R1"])
    line_x = np.linspace(eligible[R23_COLUMN].min(), eligible[R23_COLUMN].max(), 300)
    scatter_axis.plot(
        line_x, scatter_fit.intercept + scatter_fit.slope * line_x, color="#C73E1D", linewidth=2.3
    )
    for (_, row), color in zip(selected.iterrows(), colors):
        scatter_axis.scatter(row[R23_COLUMN], row["R1"], s=115, color=color, edgecolor="black", zorder=4)
        scatter_axis.annotate(
            row["selection_label"],
            (row[R23_COLUMN], row["R1"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )
    rho, _ = stats.spearmanr(eligible[R23_COLUMN], eligible["R1"])
    scatter_axis.set(
        xlabel=r"$r_{23}=\sum r^2/\sum r^3$  [$\mu m^{-1}$]",
        ylabel=r"MPM $R_1$",
        title="MPM R1 versus soma surface/volume radius marker",
    )
    scatter_axis.text(
        0.02,
        0.97,
        f"n={len(eligible):,}\nSpearman rho={rho:.3f}\nLinear R2={scatter_fit.rvalue**2:.3f}",
        transform=scatter_axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#777777"},
    )
    scatter_axis.grid(alpha=0.18)

    for suffix in ("png", "pdf"):
        overview_figure.savefig(
            args.output_dir / f"mpm_R1_vs_r23_selected_voxels.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(overview_figure)

    figure, axes = plt.subplots(
        len(args.quantiles),
        args.examples_per_quantile,
        figsize=(18.0, 14.0),
        constrained_layout=True,
        squeeze=False,
    )
    selected_records = []
    for position, ((_, row), color) in enumerate(zip(selected.iterrows(), colors)):
        grid_row, grid_column = divmod(position, args.examples_per_quantile)
        voxel_radii = radii.loc[
            radii["mri_table_row"] == row["mri_table_row"], "cell_radius_um"
        ].to_numpy(float)
        gev = density_panel(axes[grid_row, grid_column], voxel_radii, row, color)
        selected_records.append({**row.to_dict(), **gev})
    figure.suptitle(
        "Soma-radius distributions at low, middle, and high r23\n"
        "Three representative MRI voxels per category; GEV overlays are exploratory",
        fontsize=16,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"mpm_R1_vs_r23_with_radius_distributions.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    pd.DataFrame(selected_records).to_csv(
        args.output_dir / "mpm_r23_selected_voxels.csv", index=False
    )

    driver_figure, driver_axes = plt.subplots(1, 3, figsize=(16.5, 5.3), constrained_layout=True)
    driver_specs = [
        ("radius_mean_um", "Mean soma radius (um)"),
        ("radius_std_um", "Soma-radius width, SD (um)"),
        ("radius_tail_ratio_p95_median", "Upper-tail ratio, p95/median"),
    ]
    driver_results = {}
    for axis, (column, label) in zip(driver_axes, driver_specs):
        driver_results[column] = add_association(
            axis,
            eligible[column].to_numpy(float),
            eligible[R23_COLUMN].to_numpy(float),
            label,
        )
    driver_figure.suptitle("Which features of the soma-radius distribution determine r23?", fontsize=16)
    for suffix in ("png", "pdf"):
        driver_figure.savefig(
            args.output_dir / f"mpm_r23_distribution_drivers.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(driver_figure)

    report = {
        "schema_version": 1,
        "eligible_mpm_voxels": int(len(eligible)),
        "minimum_cells_per_distribution": args.min_cells,
        "selection_quantiles": args.quantiles,
        "examples_per_quantile": args.examples_per_quantile,
        "r23_definition": "sum(radius^2)/sum(radius^3) [1/um]",
        "gev_note": "Exploratory overlay only; GEV suitability for soma radii is not assumed.",
        "distribution_driver_associations": driver_results,
    }
    (args.output_dir / "mpm_r23_distribution_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote r23 distribution figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
