#!/usr/bin/env python3
"""Create cross-specimen diagnostics from the completed full MRI analysis."""

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
from scipy import stats


SPECIMENS = ("P2", "P5", "P7")
COLORS = {"P2": "#277DA1", "P5": "#D1495B", "P7": "#2A9D8F"}
TARGET_LABELS = {
    "f_soma_seg": "Soma fraction",
    "cell_density_per_mm3": "Cell density",
    "cell_radius_mean_um": "Mean radius",
    "cell_radius_median_um": "Median radius",
    "cell_radius_volume_weighted_um": "Volume-weighted radius",
    "cell_radius_effective_r5_r3_um": "Effective radius r53",
    "cell_radius_effective_r7_r3_um": "Effective radius r73",
    "cell_surface_to_volume_r23_um_inv": "Surface/volume r23",
    "cell_surface_density_um_inv": "Cell surface density",
    "f_vessel_seg": "Vessel fraction",
}
RADIUS_TARGETS = {
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r5_r3_um",
    "cell_radius_effective_r7_r3_um",
    "cell_surface_to_volume_r23_um_inv",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def specimen_from_scope(scope_id: str) -> str:
    parts = str(scope_id).split("__")
    return parts[1] if len(parts) >= 3 else ""


def pair_label(row: pd.Series, include_protocol: bool = False) -> str:
    core = f"{row['predictor']} -> {TARGET_LABELS.get(row['target'], row['target'])}"
    if include_protocol:
        return f"{row['acquisition']} | {row['protocol']} | {core}"
    return core


def annotated_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    output: Path,
    title: str,
    colorbar_label: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    if values.size == 0 or not np.isfinite(values).any():
        return
    height = max(5.5, 0.34 * len(row_labels) + 2.5)
    width = max(8.5, 1.6 * len(column_labels) + 4.5)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    shown = np.clip(values, vmin, vmax)
    image = axis.imshow(
        shown,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
        aspect="auto",
    )
    axis.set_xticks(range(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=25, ha="right")
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=8)
    axis.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "NA" if not np.isfinite(value) else f"{value:+.2f}"
            color = "black" if not np.isfinite(value) or abs(shown[i, j]) < 0.45 else "white"
            axis.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
    axis.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(image, ax=axis, shrink=0.75, label=colorbar_label)
    for suffix in ("png", "pdf"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_validation_ladder(results: pd.DataFrame) -> pd.DataFrame:
    pooled = results[
        (results["scope"] == "pooled_protocol")
        & (pd.to_numeric(results["primary_analysis"], errors="coerce") == 1)
    ].copy()
    datasets = results[
        (results["scope"] == "dataset")
        & (pd.to_numeric(results["primary_analysis"], errors="coerce") == 1)
    ].copy()
    datasets["specimen"] = datasets["scope_id"].map(specimen_from_scope)
    key = ["acquisition", "protocol", "predictor", "target"]
    columns = key + [
        "n_specimens",
        "spearman_rho",
        "linear_r2_in_sample",
        "heldout_r2",
    ]
    ladder = pooled[columns].rename(
        columns={
            "spearman_rho": "pooled_spearman_rho",
            "linear_r2_in_sample": "pooled_in_sample_r2",
            "heldout_r2": "pooled_loso_r2",
        }
    )
    for specimen in SPECIMENS:
        subset = datasets[datasets["specimen"] == specimen][
            key + ["n", "spearman_rho", "heldout_r2"]
        ].rename(
            columns={
                "n": f"{specimen}_n",
                "spearman_rho": f"{specimen}_spearman_rho",
                "heldout_r2": f"{specimen}_spatial_cv_r2",
            }
        )
        ladder = ladder.merge(subset, on=key, how="left", validate="one_to_one")
    return ladder.sort_values(key).reset_index(drop=True)


def plot_validation_ladders(ladder: pd.DataFrame, output_dir: Path) -> int:
    directory = output_dir / "validation_ladders"
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    columns = [f"{specimen}_spatial_cv_r2" for specimen in SPECIMENS] + [
        "pooled_in_sample_r2",
        "pooled_loso_r2",
    ]
    labels = [f"{specimen} spatial CV" for specimen in SPECIMENS] + [
        "Pooled in-sample",
        "Pooled LOSO",
    ]
    for (acquisition, protocol), frame in ladder.groupby(["acquisition", "protocol"], sort=True):
        frame = frame.copy()
        frame["label"] = frame.apply(pair_label, axis=1)
        values = frame[columns].to_numpy(float)
        annotated_heatmap(
            values,
            frame["label"].tolist(),
            labels,
            directory / f"{safe_name(acquisition)}__{safe_name(protocol)}__validation_ladder",
            f"{acquisition} | {protocol}: within-specimen versus cross-specimen prediction",
            r"$R^2$ (values below -1 clipped only in color)",
        )
        count += 1
    return count


def plot_effect_consistency(ladder: pd.DataFrame, output_dir: Path) -> None:
    columns = [f"{specimen}_spearman_rho" for specimen in SPECIMENS] + [
        "pooled_spearman_rho"
    ]
    rank = ladder[columns].abs().max(axis=1, skipna=True)
    selected = ladder.assign(_rank=rank).sort_values("_rank", ascending=False).head(40)
    selected = selected.sort_values("pooled_spearman_rho")
    annotated_heatmap(
        selected[columns].to_numpy(float),
        [pair_label(row, include_protocol=True) for _, row in selected.iterrows()],
        [*SPECIMENS, "Pooled"],
        output_dir / "specimen_effect_consistency_top40",
        "Strongest associations: consistency of Spearman direction among specimens",
        r"Spearman $\rho$",
    )


def plot_generalization_scatter(ladder: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 6.3), constrained_layout=True)
    acquisitions = sorted(ladder["acquisition"].unique())
    cmap = plt.get_cmap("tab10")
    for index, acquisition in enumerate(acquisitions):
        frame = ladder[ladder["acquisition"] == acquisition]
        axis.scatter(
            frame["pooled_spearman_rho"].abs(),
            frame["pooled_loso_r2"],
            s=34,
            alpha=0.72,
            edgecolors="none",
            color=cmap(index % 10),
            label=acquisition,
        )
    axis.axhline(0, color="black", linewidth=1)
    minimum = float(ladder["pooled_loso_r2"].min(skipna=True))
    axis.set_ylim(bottom=min(-0.05, max(-2.0, minimum - 0.05)))
    axis.set_xlabel(r"Absolute pooled Spearman $|\rho|$")
    axis.set_ylabel(r"Leave-one-specimen-out $R^2$")
    axis.set_title("Association strength does not imply cross-specimen prediction")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(
            (output_dir / "pooled_association_strength_vs_loso_r2").with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def load_pair_data(raw_dir: Path, row: pd.Series) -> pd.DataFrame:
    frames = []
    pattern = f"{row['acquisition']}__P*__{row['protocol']}_raw_voxel_values.csv.gz"
    flag = (
        "included_for_radius_regressions"
        if row["target"] in RADIUS_TARGETS
        else "included_in_base_analysis"
    )
    for path in sorted(raw_dir.glob(pattern)):
        columns = ["specimen", row["predictor"], row["target"], flag]
        try:
            frame = pd.read_csv(path, usecols=columns)
        except (ValueError, KeyError):
            continue
        frame = frame[pd.to_numeric(frame[flag], errors="coerce") == 1]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[row["predictor"], row["target"], "specimen"]
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_selected_scatter_grid(ladder: pd.DataFrame, raw_dir: Path, output_dir: Path) -> int:
    candidates = ladder[ladder["n_specimens"] >= 2].copy()
    candidates["rank"] = candidates["pooled_spearman_rho"].abs()
    candidates = (
        candidates.sort_values("rank", ascending=False)
        .drop_duplicates(["acquisition", "protocol", "target"])
        .head(9)
    )
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 13.5), constrained_layout=True)
    flat_axes = list(axes.flat)
    for axis in flat_axes:
        axis.set_visible(False)
    rng = np.random.default_rng(46)
    plotted = 0
    for axis, (_, row) in zip(flat_axes, candidates.iterrows()):
        frame = load_pair_data(raw_dir, row)
        if frame.empty:
            continue
        axis.set_visible(True)
        for specimen, group in frame.groupby("specimen", sort=True):
            if len(group) > 1500:
                group = group.iloc[rng.choice(len(group), 1500, replace=False)]
            x = group[row["predictor"]].to_numpy(float)
            y = group[row["target"]].to_numpy(float)
            axis.scatter(
                x,
                y,
                s=8,
                alpha=0.25,
                edgecolors="none",
                color=COLORS.get(specimen, "#777777"),
                label=specimen,
            )
            if len(group) >= 3 and np.ptp(x) > 0:
                fit = stats.linregress(x, y)
                line = np.linspace(np.min(x), np.max(x), 100)
                axis.plot(line, fit.intercept + fit.slope * line, color=COLORS.get(specimen), linewidth=1.8)
        pooled_x = frame[row["predictor"]].to_numpy(float)
        pooled_y = frame[row["target"]].to_numpy(float)
        if len(frame) >= 3 and np.ptp(pooled_x) > 0:
            pooled_fit = stats.linregress(pooled_x, pooled_y)
            pooled_line = np.linspace(np.min(pooled_x), np.max(pooled_x), 150)
            axis.plot(
                pooled_line,
                pooled_fit.intercept + pooled_fit.slope * pooled_line,
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Pooled",
            )
        axis.set_title(
            f"{row['acquisition']} | {row['protocol']}\n"
            f"{row['predictor']} vs {TARGET_LABELS.get(row['target'], row['target'])}\n"
            f"pooled rho={row['pooled_spearman_rho']:+.2f}, LOSO R2={row['pooled_loso_r2']:+.2f}",
            fontsize=9,
        )
        axis.set_xlabel(row["predictor"])
        axis.set_ylabel(TARGET_LABELS.get(row["target"], row["target"]))
        axis.grid(alpha=0.15)
        plotted += 1
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLORS[s], label=s)
        for s in SPECIMENS
    ]
    handles.append(
        plt.Line2D([], [], color="black", linestyle="--", linewidth=2.2, label="Pooled fit")
    )
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(
            (output_dir / "selected_associations_by_specimen").with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    return plotted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results, low_memory=False)
    ladder = build_validation_ladder(results)
    ladder.to_csv(args.output_dir / "validation_ladder_all_associations.csv", index=False)
    ladder_count = plot_validation_ladders(ladder, args.output_dir)
    plot_effect_consistency(ladder, args.output_dir)
    plot_generalization_scatter(ladder, args.output_dir)
    scatter_count = plot_selected_scatter_grid(ladder, args.raw_dir, args.output_dir)

    summary = pd.DataFrame(
        [
            {"artifact": "validation_ladders", "count": ladder_count},
            {"artifact": "specimen_effect_consistency", "count": 1},
            {"artifact": "association_vs_generalization", "count": 1},
            {"artifact": "selected_specimen_scatter_panels", "count": scatter_count},
        ]
    )
    summary.to_csv(args.output_dir / "diagnostic_plot_index.csv", index=False)
    print(f"Wrote generalization diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
