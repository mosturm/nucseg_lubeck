#!/usr/bin/env python3
"""Plot compact overview matrices for all MRI-histology associations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROW_ORDER = [
    ("sandi", "De"),
    ("sandi", "Din"),
    ("sandi", "Rsoma"),
    ("sandi", "fextra"),
    ("sandi", "fneurite"),
    ("sandi", "fsoma"),
    ("mpm", "R1"),
    ("mpm", "R2s"),
    ("mpm", "A"),
    ("r2s", "R2s"),
]

ROW_LABELS = {
    ("sandi", "De"): r"SANDI  $D_e$",
    ("sandi", "Din"): r"SANDI  $D_{in}$",
    ("sandi", "Rsoma"): r"SANDI  $R_{soma}$",
    ("sandi", "fextra"): r"SANDI  $f_{extra}$",
    ("sandi", "fneurite"): r"SANDI  $f_{neurite}$",
    ("sandi", "fsoma"): r"SANDI  $f_{soma}$",
    ("mpm", "R1"): r"MPM  $R_1$",
    ("mpm", "R2s"): r"MPM  $R_2^*$",
    ("mpm", "A"): "MPM  A",
    ("r2s", "R2s"): r"R2* map  $R_2^*$",
}

COLUMN_ORDER = [
    "f_soma_seg",
    "cell_density_per_mm3",
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r5_r3_um",
    "cell_radius_effective_r7_r3_um",
    "cell_surface_to_volume_radius_inverse_um_inv",
    "cell_volume_to_surface_radius_um",
    "f_vessel_seg",
]

COLUMN_LABELS = {
    "f_soma_seg": "Soma fraction\n" + r"$f_{soma}^{seg}$",
    "cell_density_per_mm3": "Cell density\n(cells/mm3)",
    "cell_radius_mean_um": "Mean sphere-\nequivalent radius",
    "cell_radius_median_um": "Median sphere-\nequivalent radius",
    "cell_radius_volume_weighted_um": "Volume-weighted radius\n" + r"$\Sigma r^4/\Sigma r^3$",
    "cell_radius_effective_r5_r3_um": "Effective radius\n" + r"$(\Sigma r^5/\Sigma r^3)^{1/2}$",
    "cell_radius_effective_r7_r3_um": "Effective radius\n" + r"$(\Sigma r^7/\Sigma r^3)^{1/4}$",
    "cell_surface_to_volume_radius_inverse_um_inv": "Surface/volume marker\n" + r"$\Sigma r^2/\Sigma r^3$",
    "cell_volume_to_surface_radius_um": "Volume/surface marker\n" + r"$\Sigma r^3/\Sigma r^2$",
    "f_vessel_seg": "Vessel fraction\n" + r"$f_{vessel}^{seg}$",
}


def matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    lookup = frame.set_index(["modality", "predictor", "target"])[value]
    return np.array(
        [
            [lookup.get((modality, predictor, target), np.nan) for target in COLUMN_ORDER]
            for modality, predictor in ROW_ORDER
        ],
        dtype=float,
    )


def draw_heatmap(
    values: np.ndarray,
    output_base: Path,
    title: str,
    colorbar_label: str,
    norm: TwoSlopeNorm,
    significant: np.ndarray | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(24, 12))
    figure.subplots_adjust(left=0.12, right=0.91, top=0.90, bottom=0.27)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")

    axis.set_xticks(range(len(COLUMN_ORDER)))
    axis.set_xticklabels(
        [COLUMN_LABELS[column] for column in COLUMN_ORDER], rotation=24, ha="right"
    )
    axis.set_yticks(range(len(ROW_ORDER)))
    axis.set_yticklabels([ROW_LABELS[row] for row in ROW_ORDER])
    axis.tick_params(axis="both", labelsize=12, length=0, pad=8)
    axis.set_title(title, fontsize=22, pad=22)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "NA"
                weight = "normal"
            else:
                label = f"{value:+.2f}" if colorbar_label.startswith("Spearman") else f"{value:.2f}"
                weight = (
                    "bold"
                    if significant is not None and bool(significant[row, column])
                    else "normal"
                )
            rgba = image.cmap(image.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight=weight,
                color="black" if luminance > 0.56 else "white",
            )

    # Visually retain the three acquisition families without splitting the matrix.
    axis.axhline(5.5, color="black", linewidth=2.2)
    axis.axhline(8.5, color="black", linewidth=2.2)
    axis.set_xticks(np.arange(-0.5, len(COLUMN_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ROW_ORDER), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2, alpha=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)

    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=14)
    colorbar.ax.tick_params(labelsize=11)
    if significant is not None:
        figure.text(
            0.5,
            0.035,
            f"Bold values: significant after Benjamini-Hochberg correction across all {values.size} tests (q < 0.05).",
            ha="center",
            fontsize=12,
        )
    else:
        figure.text(
            0.5,
            0.035,
            "Held-out spatial-block CV. Negative R2 means worse prediction than the held-out mean baseline.",
            ha="center",
            fontsize=12,
        )

    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "analysis" / "all_univariate_correlations_and_spatial_cv.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)

    observed = set(zip(frame["modality"], frame["predictor"], frame["target"]))
    expected = {
        (modality, predictor, target)
        for modality, predictor in ROW_ORDER
        for target in COLUMN_ORDER
    }
    missing = sorted(expected - observed)
    if missing:
        print(
            f"Warning: {len(missing)} of {len(expected)} planned associations were not estimable; "
            "they will be shown as NA."
        )

    association_count = len(expected)

    rho = matrix(frame, "spearman_rho")
    qvalue = matrix(frame, "spearman_p_fdr_bh")
    draw_heatmap(
        rho,
        output_dir / f"all_{association_count}_associations_spearman_heatmap",
        "MRI-Histology Associations in Hippo002",
        r"Spearman $\rho$",
        TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        significant=np.isfinite(qvalue) & (qvalue < 0.05),
    )

    cv_r2 = matrix(frame, "spatial_cv_r2")
    finite = cv_r2[np.isfinite(cv_r2)]
    lower = min(-0.1, float(np.min(finite)))
    upper = max(0.5, float(np.max(finite)))
    draw_heatmap(
        cv_r2,
        output_dir / f"all_{association_count}_associations_spatial_cv_r2_heatmap",
        "Held-Out Spatial Prediction in Hippo002",
        r"Spatial-CV $R^2$",
        TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper),
    )

    print(f"Wrote overview heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
