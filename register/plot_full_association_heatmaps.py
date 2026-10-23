#!/usr/bin/env python3
"""Plot pooled-protocol, primary, exploratory, and per-specimen heatmaps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


TARGET_ORDER = [
    "f_soma_seg",
    "cell_density_per_mm3",
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r5_r3_um",
    "cell_radius_effective_r7_r3_um",
    "cell_surface_to_volume_r23_um_inv",
    "cell_surface_density_um_inv",
    "f_vessel_seg",
]
TARGET_LABELS = {
    "f_soma_seg": "Soma fraction",
    "cell_density_per_mm3": "Cell density",
    "cell_radius_mean_um": "Mean radius",
    "cell_radius_median_um": "Median radius",
    "cell_radius_volume_weighted_um": "Volume-weighted\nradius r43",
    "cell_radius_effective_r5_r3_um": "Effective radius\nr53",
    "cell_radius_effective_r7_r3_um": "Effective radius\nr73",
    "cell_surface_to_volume_r23_um_inv": "Surface/volume\nr23",
    "cell_surface_density_um_inv": "Cell surface\ndensity",
    "f_vessel_seg": "Vessel fraction",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def draw(
    frame: pd.DataFrame,
    value_column: str,
    output: Path,
    title: str,
    significant: bool = False,
    footer: str | None = None,
):
    predictors = sorted(frame["predictor"].unique())
    lookup = frame.set_index(["predictor", "target"])
    values = np.array(
        [
            [lookup[value_column].get((predictor, target), np.nan) for target in TARGET_ORDER]
            for predictor in predictors
        ]
    )
    qvalues = np.array(
        [
            [lookup["spearman_p_fdr_bh"].get((predictor, target), np.nan) for target in TARGET_ORDER]
            for predictor in predictors
        ]
    )
    finite = values[np.isfinite(values)]
    if not len(finite):
        return
    if value_column == "spearman_rho":
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        colorbar = r"Spearman $\rho$"
    else:
        norm = TwoSlopeNorm(vmin=min(-0.25, float(finite.min())), vcenter=0, vmax=max(0.5, float(finite.max())))
        colorbar = r"Held-out $R^2$"
    height = max(5.5, 0.6 * len(predictors) + 2.5)
    figure, axis = plt.subplots(figsize=(18, height), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    axis.set_xticks(range(len(TARGET_ORDER)))
    axis.set_xticklabels([TARGET_LABELS[target] for target in TARGET_ORDER], rotation=25, ha="right")
    axis.set_yticks(range(len(predictors)))
    axis.set_yticklabels(predictors)
    axis.set_title(title, fontsize=16)
    for row in range(len(predictors)):
        for column in range(len(TARGET_ORDER)):
            value = values[row, column]
            label = "NA" if not np.isfinite(value) else f"{value:+.2f}"
            rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            weight = "bold" if significant and np.isfinite(qvalues[row, column]) and qvalues[row, column] < 0.05 else "normal"
            axis.text(column, row, label, ha="center", va="center", fontsize=10, fontweight=weight, color="black" if luminance > 0.56 else "white")
    axis.set_xticks(np.arange(-0.5, len(TARGET_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(predictors), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0, alpha=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(image, ax=axis, shrink=0.8, label=colorbar)
    if footer is None:
        footer = (
            "Bold: Spearman q < 0.05"
            if significant
            else "Leave-one-specimen-out validation where multiple specimens are available"
        )
    figure.text(0.5, 0.01, footer, ha="center")
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-exploratory", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    pooled = results[results["scope"] == "pooled_protocol"].copy()
    if not args.include_exploratory:
        pooled = pooled[pooled["primary_analysis"] == 1]
    index_rows = []
    for scope_id, frame in pooled.groupby("scope_id", sort=True):
        acquisition = frame["acquisition"].iloc[0]
        protocol = frame["protocol"].iloc[0]
        is_primary = bool(frame["primary_analysis"].iloc[0])
        qualifier = "" if is_primary else " (exploratory: registration flagged)"
        base = args.output_dir / safe_name(scope_id)
        draw(
            frame,
            "spearman_rho",
            base.with_name(base.name + "__spearman"),
            f"{acquisition} | {protocol}{qualifier}: MRI-histology Spearman associations",
            significant=True,
        )
        draw(
            frame,
            "heldout_r2",
            base.with_name(base.name + "__heldout_r2"),
            f"{acquisition} | {protocol}{qualifier}: held-out prediction",
        )
        index_rows.append(
            {
                "scope_id": scope_id,
                "acquisition": acquisition,
                "protocol": protocol,
                "primary_analysis": int(is_primary),
                "interpretation": "primary" if is_primary else "exploratory_registration_flagged",
                "predictors": ";".join(sorted(frame["predictor"].unique())),
                "associations": len(frame),
                "specimens": int(frame["n_specimens"].max()),
            }
        )

    primary = pooled[pooled["primary_analysis"] == 1].copy()
    if not primary.empty:
        primary["predictor"] = (
            primary["acquisition"].astype(str)
            + " | "
            + primary["protocol"].astype(str)
            + " | "
            + primary["predictor"].astype(str)
        )
        association_count = len(primary)
        draw(
            primary,
            "spearman_rho",
            args.output_dir / "all_primary_associations__spearman",
            f"All primary MRI-histology associations ({association_count} estimable pairs)",
            significant=True,
        )
        draw(
            primary,
            "heldout_r2",
            args.output_dir / "all_primary_associations__heldout_r2",
            f"All primary held-out MRI-histology predictions ({association_count} estimable pairs)",
        )

    if args.include_exploratory and not pooled.empty:
        inclusive = pooled.copy()
        qualifier = np.where(
            inclusive["primary_analysis"].eq(1),
            "",
            "[exploratory] ",
        )
        inclusive["predictor"] = (
            pd.Series(qualifier, index=inclusive.index)
            + inclusive["acquisition"].astype(str)
            + " | "
            + inclusive["protocol"].astype(str)
            + " | "
            + inclusive["predictor"].astype(str)
        )
        association_count = len(inclusive)
        draw(
            inclusive,
            "spearman_rho",
            args.output_dir / "all_associations_including_exploratory__spearman",
            f"All MRI-histology associations, including flagged exploratory data ({association_count} estimable pairs)",
            significant=True,
            footer=(
                "Bold: Spearman q < 0.05; [exploratory] rows are registration-flagged "
                "and are not part of the primary inference"
            ),
        )
        draw(
            inclusive,
            "heldout_r2",
            args.output_dir / "all_associations_including_exploratory__heldout_r2",
            f"All held-out predictions, including flagged exploratory data ({association_count} estimable pairs)",
            footer=(
                "[exploratory] rows are registration-flagged and are not part of the "
                "primary inference"
            ),
        )

    dataset = results[
        (results["scope"] == "dataset") & (results["primary_analysis"] == 1)
    ].copy()
    if not dataset.empty:
        dataset["specimen"] = dataset["scope_id"].astype(str).str.split("__").str[1]
        for specimen, frame in dataset.groupby("specimen", sort=True):
            frame = frame.copy()
            frame["predictor"] = (
                frame["acquisition"].astype(str)
                + " | "
                + frame["protocol"].astype(str)
                + " | "
                + frame["predictor"].astype(str)
            )
            count = len(frame)
            base = args.output_dir / f"{safe_name(specimen)}__all_primary_associations"
            draw(
                frame,
                "spearman_rho",
                base.with_name(base.name + "__spearman"),
                f"{specimen}: all primary MRI-histology associations ({count} estimable pairs)",
                significant=True,
                footer="Bold: Spearman q < 0.05 (BH correction across primary dataset-level tests)",
            )
            draw(
                frame,
                "heldout_r2",
                base.with_name(base.name + "__spatial_cv_r2"),
                f"{specimen}: spatially held-out MRI-histology prediction ({count} estimable pairs)",
                footer="Five-fold spatial-block validation within this specimen",
            )

    pd.DataFrame(index_rows).to_csv(args.output_dir / "heatmap_index.csv", index=False)
    print(
        f"Wrote {len(index_rows)} protocol heatmap pairs, primary and inclusive pooled overviews, "
        f"and {dataset['specimen'].nunique() if not dataset.empty else 0} specimen overviews "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
