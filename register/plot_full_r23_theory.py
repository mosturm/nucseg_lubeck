#!/usr/bin/env python3
"""Plot R1 against the requested r23 surface-to-volume marker."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--theory-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-min", type=float, default=80.0)
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.features_dir / "feature_manifest.csv")
    theory = pd.read_csv(args.theory_results)
    theory = theory[(theory["scope"] == "pooled_protocol") & (theory["primary_analysis"] == 1)]
    colors = {"P2": "#277DA1", "P5": "#D1495B", "P7": "#2A9D8F"}
    index = []
    for result in theory.itertuples(index=False):
        acquisition, protocol = result.acquisition, result.protocol
        selected_manifest = manifest[
            (manifest["acquisition"] == acquisition) & (manifest["protocol"] == protocol)
        ]
        frames = [pd.read_csv(path, low_memory=False) for path in selected_manifest["feature_table"]]
        frame = pd.concat(frames, ignore_index=True)
        r1 = result.r1_column
        source_r1 = (
            f"{r1}_uncorrected"
            if acquisition == "HH_exvivo" and f"{r1}_uncorrected" in frame.columns
            else r1
        )
        required = [source_r1, "cell_surface_to_volume_r23_um_inv", "specimen", "coord_usable", "coverage_percent", "local_basis_valid", "cell_count"]
        selected = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
        selected = selected[
            (selected["coord_usable"] == 1)
            & (selected["coverage_percent"] >= args.coverage_min)
            & (selected["local_basis_valid"] == 1)
            & (selected["cell_count"] >= args.min_cells)
        ]
        figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
        for specimen, group in selected.groupby("specimen"):
            axis.scatter(
                group["cell_surface_to_volume_r23_um_inv"],
                group[source_r1],
                s=20,
                alpha=0.5,
                edgecolors="none",
                color=colors.get(specimen, "#777777"),
                label=f"{specimen} (n={len(group)})",
            )
        xline = np.linspace(selected["cell_surface_to_volume_r23_um_inv"].min(), selected["cell_surface_to_volume_r23_um_inv"].max(), 300)
        axis.plot(xline, result.r1_free_intercept + result.r23_slope * xline, color="black", linewidth=2.2)
        axis.set_xlabel(r"$r_{23}=\sum r^2/\sum r^3$  [$\mu m^{-1}$]")
        axis.set_ylabel(r1)
        axis.set_title(f"{acquisition} | {protocol}: {r1} versus r23")
        axis.text(
            0.03,
            0.97,
            f"n={len(selected):,}\nintercept={result.r1_free_intercept:.4g}\nslope={result.r23_slope:.4g}\nLOSO R2={result.heldout_r2:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#777777"},
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        base = args.output_dir / f"{safe_name(result.scope_id)}__{safe_name(r1)}_vs_r23"
        for suffix in ("png", "pdf"):
            figure.savefig(base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
        plt.close(figure)
        index.append({"scope_id": result.scope_id, "r1_column": r1, "n": len(selected), "plot": str(base.with_suffix('.png'))})
    pd.DataFrame(index).to_csv(args.output_dir / "r23_plot_index.csv", index=False)
    print(f"Wrote {len(index)} R1-r23 plot pairs")


if __name__ == "__main__":
    main()
