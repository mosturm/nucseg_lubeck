#!/usr/bin/env python3
"""Build a compact interactive 3D view of cells and all vessel scales."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from visualize_cell_vessel_overlay_ome_zarr import (
    accumulate_foreground_on_grid,
    apply_scale_postprocess,
    boundary_coordinates,
    open_array,
    upsample_nearest_on_grid,
)


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_capacity(arrays: tuple[np.ndarray, ...], required: int):
    if required <= arrays[0].size:
        return arrays
    new_size = max(required, arrays[0].size * 2)
    grown = []
    for array in arrays:
        expanded = np.zeros(new_size, dtype=array.dtype)
        expanded[: array.size] = array
        grown.append(expanded)
    return tuple(grown)


def scan_cell_instances(label_array, block_zyx: tuple[int, int, int]):
    """Stream instance labels once and return IDs, sizes, and centroids."""
    shape = tuple(int(value) for value in label_array.shape)
    counts = np.zeros(1024, dtype=np.uint64)
    sums = tuple(np.zeros(1024, dtype=np.float64) for _ in range(3))
    total_blocks = math.prod(
        math.ceil(shape[axis] / block_zyx[axis]) for axis in range(3)
    )
    processed = 0
    for z0 in range(0, shape[0], block_zyx[0]):
        z1 = min(shape[0], z0 + block_zyx[0])
        for y0 in range(0, shape[1], block_zyx[1]):
            y1 = min(shape[1], y0 + block_zyx[1])
            for x0 in range(0, shape[2], block_zyx[2]):
                x1 = min(shape[2], x0 + block_zyx[2])
                processed += 1
                block = np.asarray(label_array[z0:z1, y0:y1, x0:x1])
                foreground = block > 0
                if foreground.any():
                    local = np.nonzero(foreground)
                    labels = block[foreground].astype(np.int64, copy=False)
                    unique, inverse, local_counts = np.unique(
                        labels, return_inverse=True, return_counts=True
                    )
                    required = int(unique[-1]) + 1
                    counts, *grown_sums = ensure_capacity(
                        (counts, *sums), required
                    )
                    sums = tuple(grown_sums)
                    counts[unique] += local_counts.astype(np.uint64)
                    offsets = (z0, y0, x0)
                    for axis in range(3):
                        sums[axis][unique] += np.bincount(
                            inverse,
                            weights=(local[axis] + offsets[axis]).astype(np.float64),
                            minlength=unique.size,
                        )
                if processed == 1 or processed % 100 == 0 or processed == total_blocks:
                    progress(f"cell instance scan {processed}/{total_blocks}")

    instance_ids = np.flatnonzero(counts).astype(np.int64)
    if instance_ids.size < 2:
        raise RuntimeError("Cell input does not contain usable instance IDs")
    instance_counts = counts[instance_ids]
    centroids = np.column_stack([axis[instance_ids] for axis in sums])
    centroids /= instance_counts[:, None]
    return instance_ids, instance_counts, centroids


def save_log_radius_histogram(
    outdir: Path, radii_um: np.ndarray, bins: int
) -> dict[str, float | int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = float(np.mean(radii_um))
    std = float(np.std(radii_um))
    mean_minus_sigma = max(0.0, mean - std)
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(
        radii_um,
        bins=bins,
        color="#D62728",
        edgecolor="white",
        linewidth=0.35,
    )
    axis.set_yscale("log", base=10)
    axis.axvline(mean, color="black", linewidth=1.2, label="Mean")
    axis.axvline(
        mean_minus_sigma,
        color="#555555",
        linewidth=1.2,
        linestyle="--",
        label="Mean - 1 SD",
    )
    axis.set_xlabel("Equivalent-sphere cell radius (um)")
    axis.set_ylabel("Number of segmented instances (log10 scale)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "cell_radius_histogram_log10.png", dpi=220)
    fig.savefig(outdir / "cell_radius_histogram_log10.pdf")
    plt.close(fig)
    return {
        "n_cells": int(radii_um.size),
        "mean_radius_um": mean,
        "std_radius_um": std,
        "mean_minus_sigma_radius_um": mean_minus_sigma,
        "mean_minus_sigma_diameter_um": 2.0 * mean_minus_sigma,
        "n_radius_gt_3um": int(np.count_nonzero(radii_um > 3.0)),
    }


def cylinder_mask(shape: tuple[int, int, int]) -> np.ndarray:
    _, ny, nx = shape
    y = ((np.arange(ny, dtype=np.float32) + 0.5) / ny - 0.5) * 2.0
    x = ((np.arange(nx, dtype=np.float32) + 0.5) / nx - 0.5) * 2.0
    return y[:, None] ** 2 + x[None, :] ** 2 <= 1.0


def physical_coordinates(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int, int],
    extent_um_zyx: tuple[float, float, float],
):
    values = []
    for axis in range(3):
        values.append(
            ((coordinates[axis].astype(np.float32) + 0.5) / shape[axis] - 0.5)
            * extent_um_zyx[axis]
            / 1000.0
        )
    return values[2], values[1], values[0]


def vessel_points(
    grid: np.ndarray,
    max_points: int,
    seed: int,
    extent_um_zyx: tuple[float, float, float],
):
    grid &= cylinder_mask(grid.shape)[None, :, :]
    coordinates, total = boundary_coordinates(grid, max_points, seed)
    return physical_coordinates(coordinates, grid.shape, extent_um_zyx), total


def make_cylinder_traces(go, diameter_mm: float, height_mm: float):
    theta = np.linspace(0.0, 2.0 * math.pi, 129)
    radius = diameter_mm / 2.0
    traces = []
    for z_value, name in ((-height_mm / 2.0, "Cylinder"), (height_mm / 2.0, None)):
        traces.append(
            go.Scatter3d(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                z=np.full(theta.shape, z_value),
                mode="lines",
                name=name,
                showlegend=name is not None,
                line=dict(color="rgba(70,70,70,0.5)", width=3),
                hoverinfo="skip",
            )
        )
    for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        traces.append(
            go.Scatter3d(
                x=[radius * math.cos(angle)] * 2,
                y=[radius * math.sin(angle)] * 2,
                z=[-height_mm / 2.0, height_mm / 2.0],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(90,90,90,0.22)", width=2),
                hoverinfo="skip",
            )
        )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--unscaled-vessels", type=Path, required=True)
    parser.add_argument("--mid-vessels", type=Path, required=True)
    parser.add_argument("--coarse-vessels", type=Path, required=True)
    parser.add_argument("--mid-thresholds", type=Path, required=True)
    parser.add_argument("--coarse-thresholds", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--physical-extent-um", default="1200,1200,1200")
    parser.add_argument("--cell-scan-block", default="40,256,256")
    parser.add_argument("--max-vessel-points", type=int, default=120000)
    parser.add_argument("--histogram-bins", type=int, default=60)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    extent_um_zyx = tuple(float(value) for value in args.physical_extent_um.split(","))
    scan_block = tuple(int(value) for value in args.cell_scan_block.split(","))
    if len(extent_um_zyx) != 3 or len(scan_block) != 3:
        raise ValueError("Expected three Z,Y,X values")
    args.outdir.mkdir(parents=True, exist_ok=True)

    cells = open_array(args.cells, args.array_key)
    unscaled = open_array(args.unscaled_vessels, args.array_key)
    mid = open_array(args.mid_vessels, args.array_key)
    coarse = open_array(args.coarse_vessels, args.array_key)
    source_shape = tuple(int(value) for value in cells.shape)
    if tuple(int(value) for value in unscaled.shape) != source_shape:
        raise ValueError("Cell and unscaled-vessel grids must match")
    display_shape = tuple(int(value) for value in mid.shape)
    progress(f"Native mid display grid ZYX={display_shape}")

    ids, cell_voxels, centroids = scan_cell_instances(cells, scan_block)
    voxel_um_zyx = np.asarray(extent_um_zyx) / np.asarray(source_shape)
    radii_um = np.cbrt(
        3.0 * cell_voxels.astype(np.float64) * float(np.prod(voxel_um_zyx))
        / (4.0 * math.pi)
    )
    radius_summary = save_log_radius_histogram(
        args.outdir, radii_um, args.histogram_bins
    )

    display_indices = np.floor(
        (centroids + 0.5)
        * np.asarray(display_shape, dtype=np.float64)
        / np.asarray(source_shape, dtype=np.float64)
    ).astype(np.int64)
    for axis in range(3):
        display_indices[:, axis] = np.clip(
            display_indices[:, axis], 0, display_shape[axis] - 1
        )
    cell_x, cell_y, cell_z = physical_coordinates(
        tuple(display_indices[:, axis] for axis in range(3)),
        display_shape,
        extent_um_zyx,
    )
    inside_cells = cell_x**2 + cell_y**2 <= (extent_um_zyx[2] / 2000.0) ** 2
    large_cells = (radii_um > 3.0) & inside_cells
    small_cells = (~large_cells) & inside_cells

    progress("Preparing fine/unscaled vessel display without analytical filtering")
    fine_grid = np.zeros(display_shape, dtype=bool)
    fine_source_foreground = accumulate_foreground_on_grid(unscaled, fine_grid)
    fine_xyz, fine_boundary_total = vessel_points(
        fine_grid, args.max_vessel_points, args.seed, extent_um_zyx
    )
    del fine_grid

    progress("Applying the saved native-scale mid analytical rule")
    filtered_mid, mid_summary = apply_scale_postprocess(mid, args.mid_thresholds, "mid")
    mid_xyz, mid_boundary_total = vessel_points(
        np.asarray(filtered_mid, dtype=bool),
        args.max_vessel_points,
        args.seed + 1,
        extent_um_zyx,
    )
    del filtered_mid

    progress("Applying the saved native-scale coarse analytical rule")
    filtered_coarse, coarse_summary = apply_scale_postprocess(
        coarse, args.coarse_thresholds, "coarse"
    )
    coarse_grid = np.zeros(display_shape, dtype=bool)
    upsample_nearest_on_grid(filtered_coarse, coarse_grid)
    coarse_xyz, coarse_boundary_total = vessel_points(
        coarse_grid, args.max_vessel_points, args.seed + 2, extent_um_zyx
    )
    del filtered_coarse, coarse_grid

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit("Install plotly in the active environment") from exc

    traces = [
        go.Scatter3d(
            x=cell_x[small_cells], y=cell_y[small_cells], z=cell_z[small_cells],
            mode="markers", name="Cells <= 3 um",
            marker=dict(size=1.0, color="#E53935", opacity=0.38), hoverinfo="skip",
        ),
        go.Scatter3d(
            x=cell_x[large_cells], y=cell_y[large_cells], z=cell_z[large_cells],
            mode="markers", name="Cells > 3 um",
            marker=dict(size=2.2, color="#A80000", opacity=0.82), hoverinfo="skip",
        ),
        go.Scatter3d(
            x=fine_xyz[0], y=fine_xyz[1], z=fine_xyz[2], mode="markers",
            name="Fine vessels (unscaled; no postprocessing)",
            marker=dict(size=1.25, color="#8CCBFF", opacity=0.62), hoverinfo="skip",
        ),
        go.Scatter3d(
            x=mid_xyz[0], y=mid_xyz[1], z=mid_xyz[2], mode="markers",
            name="Mid vessels (postprocessed)",
            marker=dict(size=1.5, color="#3182CE", opacity=0.76), hoverinfo="skip",
        ),
        go.Scatter3d(
            x=coarse_xyz[0], y=coarse_xyz[1], z=coarse_xyz[2], mode="markers",
            name="Coarse vessels (postprocessed)",
            marker=dict(size=1.8, color="#084C9E", opacity=0.88), hoverinfo="skip",
        ),
    ]
    cylinder_traces = make_cylinder_traces(
        go, extent_um_zyx[2] / 1000.0, extent_um_zyx[0] / 1000.0
    )
    figure = go.Figure(data=[*traces, *cylinder_traces])
    cylinder_visible = [True] * len(cylinder_traces)
    modes = [
        ("All", [True, True, True, True, True]),
        ("Cells only", [True, True, False, False, False]),
        ("Cells > 3 um", [False, True, False, False, False]),
        ("All vessels", [False, False, True, True, True]),
        ("Fine vessels", [False, False, True, False, False]),
        ("Mid vessels", [False, False, False, True, False]),
        ("Coarse vessels", [False, False, False, False, True]),
    ]
    figure.update_layout(
        title="Cells and multiscale vessel segmentation",
        template="plotly_white",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            aspectmode="data", camera=dict(eye=dict(x=1.45, y=1.45, z=1.05)),
        ),
        updatemenus=[dict(
            type="dropdown", direction="down", x=0.5, xanchor="center", y=1.08,
            buttons=[dict(
                label=label, method="update",
                args=[{"visible": [*visible, *cylinder_visible]}],
            ) for label, visible in modes],
        )],
        margin=dict(l=0, r=0, b=0, t=55),
        legend=dict(x=0.01, y=0.99),
    )
    html_path = args.outdir / "cells_vessels_all_scales_3d.html"
    figure.write_html(
        str(html_path), include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )

    eligible_cells = np.flatnonzero(inside_cells)
    sample_count = min(
        eligible_cells.size,
        max(1, int(round(0.05 * eligible_cells.size))),
    )
    rng = np.random.default_rng(args.seed)
    sampled_cells = np.sort(
        rng.choice(eligible_cells, size=sample_count, replace=False)
    )
    overview_traces = [
        go.Scatter3d(
            x=cell_x[sampled_cells],
            y=cell_y[sampled_cells],
            z=cell_z[sampled_cells],
            mode="markers",
            name="Cells (fixed random 5% sample)",
            marker=dict(size=1.5, color="#D73027", opacity=0.58),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=mid_xyz[0],
            y=mid_xyz[1],
            z=mid_xyz[2],
            mode="markers",
            name="Mid vessels (postprocessed)",
            marker=dict(size=1.5, color="#2166AC", opacity=0.80),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=coarse_xyz[0],
            y=coarse_xyz[1],
            z=coarse_xyz[2],
            mode="markers",
            name="Coarse vessels (postprocessed)",
            marker=dict(size=1.8, color="#2166AC", opacity=0.80),
            hoverinfo="skip",
        ),
    ]
    overview_cylinder = make_cylinder_traces(
        go, extent_um_zyx[2] / 1000.0, extent_um_zyx[0] / 1000.0
    )
    overview = go.Figure(data=[*overview_traces, *overview_cylinder])
    overview_modes = [
        ("All", [True, True, True]),
        ("Sampled cells", [True, False, False]),
        ("All vessels", [False, True, True]),
        ("Mid vessels", [False, True, False]),
        ("Coarse vessels", [False, False, True]),
    ]
    overview.update_layout(
        title="Mid/coarse vessels with a fixed random 5% cell sample",
        template="plotly_white",
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.45, y=1.45, z=1.05),
                projection=dict(type="orthographic"),
            ),
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.5,
            xanchor="center",
            y=1.08,
            buttons=[dict(
                label=label,
                method="update",
                args=[{
                    "visible": [
                        *visible,
                        *([True] * len(overview_cylinder)),
                    ]
                }],
            ) for label, visible in overview_modes],
        )],
        margin=dict(l=0, r=0, b=0, t=55),
        legend=dict(x=0.01, y=0.99),
    )
    overview_html_path = (
        args.outdir / "cells5pct_vessels_mid_coarse_3d.html"
    )
    overview.write_html(
        str(overview_html_path),
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )

    metadata = {
        "html": str(html_path.resolve()),
        "mid_coarse_5pct_cells_html": str(overview_html_path.resolve()),
        "cell_instances": str(args.cells.resolve()),
        "vessel_inputs": {
            "fine_unscaled": str(args.unscaled_vessels.resolve()),
            "mid": str(args.mid_vessels.resolve()),
            "coarse": str(args.coarse_vessels.resolve()),
        },
        "source_shape_zyx": list(source_shape),
        "display_shape_zyx": list(display_shape),
        "source_voxels_per_display_voxel_zyx": (
            np.asarray(source_shape) / np.asarray(display_shape)
        ).tolist(),
        "display_voxel_um_zyx": (
            np.asarray(extent_um_zyx) / np.asarray(display_shape)
        ).tolist(),
        "cell_radius_summary": radius_summary,
        "cells_inside_cylinder": int(np.count_nonzero(inside_cells)),
        "cells_gt_3um_inside_cylinder": int(np.count_nonzero(large_cells)),
        "overview_cell_sample": {
            "population_inside_cylinder": int(eligible_cells.size),
            "sample_count": int(sample_count),
            "requested_fraction": 0.05,
            "realized_fraction": float(sample_count / eligible_cells.size),
            "sampling_without_replacement": True,
        },
        "cell_marker_policy": "one minimum-size centroid point per instance, snapped to the native mid grid",
        "vessel_boundary_points": {
            "fine_before_sampling": fine_boundary_total,
            "mid_before_sampling": mid_boundary_total,
            "coarse_before_sampling": coarse_boundary_total,
            "maximum_rendered_per_scale": args.max_vessel_points,
        },
        "fine_source_foreground_voxels": fine_source_foreground,
        "postprocessing": {
            "fine": "none",
            "mid": mid_summary,
            "coarse": coarse_summary,
        },
        "vessel_colors": {
            "fine": "#8CCBFF", "mid": "#3182CE", "coarse": "#084C9E"
        },
        "sampling_seed": args.seed,
    }
    (args.outdir / "cells_vessels_all_scales_3d_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    progress(f"Wrote interactive HTML: {html_path}")
    progress(f"Wrote 5% cell/mid/coarse HTML: {overview_html_path}")


if __name__ == "__main__":
    main()
