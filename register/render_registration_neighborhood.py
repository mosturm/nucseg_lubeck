#!/usr/bin/env python3
"""Render one selected MRI-to-XPCT neighborhood from the real registered data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

from full_registration_config import SPECIMENS, cell_mask_path


CORNER_NAMES = ("000", "001", "010", "011", "100", "101", "110", "111")
FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
BITS = np.array([[int(bit) for bit in name] for name in CORNER_NAMES])
TOMO_SHAPE_IJK = np.array([2560, 2560, 2156], dtype=np.int64)
TARGET = "cell_radius_mean_um"
TARGET_PANEL_TITLE = r"Mean cell radius ($\mu$m)"
TARGET_AXIS_LABEL = r"Mean sphere-equivalent cell radius ($\mu$m)"


def open_array(path: Path, key: str):
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if key not in root:
        raise KeyError(f"Missing array {key!r} in {path}")
    return root[key]


def raw_path(data_root: Path, specimen: str, tomo: str) -> Path:
    sample = SPECIMENS[specimen]["sample_id"]
    stem = f"tomo_reco_id{sample}_t{tomo}"
    candidates = []
    if specimen == "P5" and tomo == "0022":
        candidates.append(data_root / f"{stem}_raw_correct_axrot.ome.zarr")
    candidates.append(data_root / f"{stem}_raw.ome.zarr")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("No raw OME-Zarr found; checked: " + ", ".join(map(str, candidates)))


def corners_for_row(row: pd.Series, tomo: str) -> np.ndarray:
    return np.array(
        [
            [float(row[f"c{name}_tomo{tomo}_{axis}_650nm"]) for axis in "ijk"]
            for name in CORNER_NAMES
        ],
        dtype=float,
    )


def center_basis(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = corners.mean(axis=0)
    basis = np.empty((3, 3), dtype=float)
    for axis in range(3):
        high = corners[BITS[:, axis] == 1].mean(axis=0)
        low = corners[BITS[:, axis] == 0].mean(axis=0)
        basis[:, axis] = high - low
    return center, basis, np.linalg.inv(basis)


def mri_aligned_display_transform(corner_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return a rigid world-to-display transform aligned with the MRI voxel axes."""
    bases = np.stack([center_basis(corners)[1] for corners in corner_sets])
    mean_basis = bases.mean(axis=0)
    u, _, vt = np.linalg.svd(mean_basis)
    world_from_mri = u @ vt
    origin = np.concatenate(corner_sets, axis=0).mean(axis=0)
    return origin, world_from_mri


def to_display(points: np.ndarray, origin: np.ndarray, world_from_mri: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (points - origin) @ world_from_mri


def style_3d(axis, bounds: tuple[np.ndarray, np.ndarray], title: str) -> None:
    low, high = bounds
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_box_aspect(np.maximum(high - low, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=24, azim=-55)
    axis.set_axis_off()
    axis.set_facecolor("black")
    axis.set_title(title, color="white", fontsize=11, pad=2)


def add_wireframe(axis, corners: np.ndarray, color="#F05A5A", linewidth=0.8, alpha=0.68) -> None:
    for face in FACES:
        ring = np.vstack([corners[list(face)], corners[face[0]]])
        axis.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=linewidth, alpha=alpha)


def plot_registered_voxels(axis, rows: pd.DataFrame, value: str, corner_sets, bounds, title: str, cmap_name: str) -> None:
    values = pd.to_numeric(rows[value], errors="coerce").to_numpy(float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if math.isclose(low, high):
        high = low + 1.0
    norm = colors.Normalize(low, high)
    cmap = matplotlib.colormaps[cmap_name]
    for number, ((_, row), corners) in enumerate(zip(rows.iterrows(), corner_sets), 1):
        faces = [corners[list(face)] for face in FACES]
        collection = Poly3DCollection(
            faces,
            facecolors=[cmap(norm(float(row[value])))] * 6,
            edgecolors="#BFBFBF" if cmap_name == "gray" else "#202020",
            linewidths=0.8,
            alpha=0.88,
        )
        axis.add_collection3d(collection)
        axis.text(*corners.mean(axis=0), str(number), color="#FF6464", fontsize=7, ha="center", va="center")
    style_3d(axis, bounds, title)


def plot_mri_voxels(axis, rows: pd.DataFrame, predictor: str, corner_sets, bounds) -> None:
    plot_registered_voxels(axis, rows, predictor, corner_sets, bounds, f"MRI: {predictor}", "gray")


def face_grid(corners: np.ndarray, face: tuple[int, ...], samples: int) -> np.ndarray:
    p00, p10, p11, p01 = corners[list(face)]
    u, v = np.meshgrid(np.linspace(0.0, 1.0, samples), np.linspace(0.0, 1.0, samples))
    return (
        (1 - u)[..., None] * (1 - v)[..., None] * p00
        + u[..., None] * (1 - v)[..., None] * p10
        + u[..., None] * v[..., None] * p11
        + (1 - u)[..., None] * v[..., None] * p01
    )


def sample_exterior_xpct_faces(raw_array, rows: pd.DataFrame, corner_sets, samples: int = 48):
    coordinates = [tuple(int(row[column]) for column in ("mri_i", "mri_j", "mri_k")) for _, row in rows.iterrows()]
    occupied = set(coordinates)
    surfaces = []
    for coordinate, corners in zip(coordinates, corner_sets):
        for axis in range(3):
            for side in (0, 1):
                neighbor = list(coordinate)
                neighbor[axis] += -1 if side == 0 else 1
                if tuple(neighbor) in occupied:
                    continue
                points = face_grid(corners, FACES[2 * axis + side], samples)
                indices_ijk = np.rint(points.reshape(-1, 3)).astype(np.int64)
                indices_ijk = np.clip(indices_ijk, 0, TOMO_SHAPE_IJK - 1)
                selection = (indices_ijk[:, 2], indices_ijk[:, 1], indices_ijk[:, 0])
                if hasattr(raw_array, "vindex"):
                    values = np.asarray(raw_array.vindex[selection])
                else:
                    values = np.asarray(raw_array.get_coordinate_selection(selection))
                surfaces.append((points, values.reshape(samples, samples)))
    return surfaces


def plot_xpct_surfaces(axis, surfaces, bounds) -> None:
    gray = matplotlib.colormaps["gray"]
    all_values = np.concatenate([values.ravel() for _, values in surfaces]).astype(np.float32)
    finite = all_values[np.isfinite(all_values)]
    low, high = np.percentile(finite, [1.0, 99.5]) if len(finite) else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    for points, values in surfaces:
        normalized = np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        axis.plot_surface(
            points[..., 0], points[..., 1], points[..., 2],
            facecolors=gray(normalized), shade=False, linewidth=0, antialiased=False,
            rcount=points.shape[0], ccount=points.shape[1],
        )
    style_3d(axis, bounds, "Registered XPCT")


def points_inside_registered_voxels(points_ijk: np.ndarray, geometries) -> np.ndarray:
    inside_any = np.zeros(len(points_ijk), dtype=bool)
    for center, inverse in geometries:
        local = (points_ijk - center) @ inverse.T
        inside_any |= np.max(np.abs(local), axis=1) <= 0.500001
    return inside_any


def plot_cells(
    axis, mask: np.ndarray, sampled_ijk, geometries, bounds,
    display_origin: np.ndarray, world_from_mri: np.ndarray,
    id_step: int, max_points: int,
) -> dict:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    kept = labels[::id_step]
    if not len(kept):
        style_3d(axis, bounds, "Cellpose instances (no sampled IDs)")
        return {"sampled_instance_ids": 0, "rendered_points": 0}
    foreground = np.isin(mask, kept)
    zyx = np.column_stack(np.nonzero(foreground))
    label_values = mask[foreground]
    x, y, z = sampled_ijk
    points = np.column_stack([x[zyx[:, 2]], y[zyx[:, 1]], z[zyx[:, 0]]])
    inside = points_inside_registered_voxels(points, geometries)
    points, label_values = points[inside], label_values[inside]
    if len(points) > max_points:
        rng = np.random.default_rng(46)
        selected = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points, label_values = points[selected], label_values[selected]
    points = to_display(points, display_origin, world_from_mri)
    mapped = np.searchsorted(kept, label_values)
    palette = matplotlib.colormaps["turbo"](np.linspace(0.03, 0.97, max(len(kept), 2)))
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=palette[mapped], s=2.2, alpha=0.72, linewidths=0, depthshade=False,
    )
    style_3d(axis, bounds, f"Cellpose instances (every {id_step}th ID)")
    return {"sampled_instance_ids": int(len(kept)), "rendered_points": int(len(points))}


def plot_correlation(axis, full: pd.DataFrame, selected: pd.DataFrame, predictor: str) -> None:
    include = pd.to_numeric(full["included_for_radius_regressions"], errors="coerce").eq(1)
    include &= pd.to_numeric(full["coverage_percent"], errors="coerce").ge(80)
    x = pd.to_numeric(full.loc[include, TARGET], errors="coerce")
    y = pd.to_numeric(full.loc[include, predictor], errors="coerce")
    finite = x.notna() & y.notna()
    x, y = x[finite].to_numpy(float), y[finite].to_numpy(float)
    axis.scatter(x, y, s=10, color="#A8A8A8", alpha=0.35, linewidths=0, label="All eligible MRI voxels")
    selected_x = pd.to_numeric(selected[TARGET], errors="coerce").to_numpy(float)
    selected_y = pd.to_numeric(selected[predictor], errors="coerce").to_numpy(float)
    selected_finite = np.isfinite(selected_x) & np.isfinite(selected_y)
    selected_x, selected_y = selected_x[selected_finite], selected_y[selected_finite]
    if not len(selected_x):
        raise ValueError("The selected voxels have no finite MRI/radius pairs")
    axis.scatter(
        selected_x, selected_y, s=62, color="#FFCB47",
        edgecolor="black", linewidth=0.6, zorder=5, label="Highlighted voxels",
    )
    finite_rows = selected.loc[selected_finite]
    for number, (_, row) in enumerate(finite_rows.iterrows(), 1):
        axis.annotate(str(number), (row[TARGET], row[predictor]), xytext=(3, 3), textcoords="offset points", color="white", fontsize=7)

    def padded_limits(
        values: np.ndarray, lower_fraction: float = 0.30, upper_fraction: float = 0.30,
    ) -> tuple[float, float]:
        low, high = float(values.min()), float(values.max())
        span = high - low
        if span > 0:
            return low - lower_fraction * span, high + upper_fraction * span
        padding = max(abs(low) * max(lower_fraction, upper_fraction), 0.1)
        return low - padding, high + padding

    x_limits = padded_limits(selected_x)
    y_limits = padded_limits(selected_y)
    y_limits = (y_limits[0] - 0.10 * abs(y_limits[0]), y_limits[1])
    visible = (
        (x >= x_limits[0]) & (x <= x_limits[1])
        & (y >= y_limits[0]) & (y <= y_limits[1])
    )
    if visible.sum() >= 2 and np.ptp(x[visible]) > 0:
        slope, intercept = np.polyfit(x[visible], y[visible], 1)
        line_x = np.linspace(x_limits[0], x_limits[1], 300)
        axis.plot(
            line_x, intercept + slope * line_x,
            color="#E84A5F", linewidth=2.4, zorder=3,
        )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel(TARGET_AXIS_LABEL, color="white")
    axis.set_ylabel(predictor, color="white")
    axis.set_title("Observed MRI-histology association", color="white", fontsize=11)
    axis.set_facecolor("black")
    axis.tick_params(colors="white")
    for spine in axis.spines.values():
        spine.set_color("#777777")
    legend = axis.legend(loc="upper right", frameon=False, fontsize=7)
    for text in legend.get_texts():
        text.set_color("white")


def save_panel(figure, output_stem: Path, dpi: int = 300) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, facecolor="black", bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="black", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--max-render-dim", type=int, default=180)
    parser.add_argument("--cell-id-step", type=int, default=10)
    parser.add_argument("--max-cell-points", type=int, default=250_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = pd.read_csv(args.selection_dir / "selected_neighborhoods.csv")
    matching = summaries[summaries["rank"].eq(args.rank)]
    if len(matching) != 1:
        raise ValueError(f"Expected one selection for rank {args.rank}, found {len(matching)}")
    summary = matching.iloc[0]
    selected_all = pd.read_csv(args.selection_dir / "selected_neighborhood_voxels.csv.gz", low_memory=False)
    selected = selected_all[selected_all["rank"].eq(args.rank)].copy()
    selected = selected.sort_values(["mri_i", "mri_j", "mri_k"])
    dataset = str(summary["dataset_id"])
    predictor = str(summary["predictor"])
    specimen = str(summary["specimen"])
    tomo = str(summary["analysis_tomogram"]).zfill(4)

    raw_table = args.analysis_dir / "raw_voxel_tables" / f"{dataset}_raw_voxel_values.csv.gz"
    full = pd.read_csv(raw_table, low_memory=False)
    raw_store = raw_path(args.data_root, specimen, tomo)
    mask_store = cell_mask_path(args.data_root, specimen, tomo, merged=True)
    if not mask_store.is_dir():
        raise FileNotFoundError(f"Missing canonical merged instance mask: {mask_store}")
    raw_array = open_array(raw_store, args.array_key)
    mask_array = open_array(mask_store, args.array_key)
    if tuple(raw_array.shape) != tuple(mask_array.shape):
        raise ValueError(f"Raw/mask shape mismatch: {raw_array.shape} versus {mask_array.shape}")

    world_corner_sets = [corners_for_row(row, tomo) for _, row in selected.iterrows()]
    display_origin, world_from_mri = mri_aligned_display_transform(world_corner_sets)
    display_corner_sets = [
        to_display(corners, display_origin, world_from_mri)
        for corners in world_corner_sets
    ]
    all_world_corners = np.concatenate(world_corner_sets, axis=0)
    low = np.maximum(np.floor(all_world_corners.min(axis=0)).astype(int) - 3, 0)
    high = np.minimum(np.ceil(all_world_corners.max(axis=0)).astype(int) + 4, TOMO_SHAPE_IJK)
    extent = high - low
    step = np.maximum(np.ceil(extent / args.max_render_dim).astype(int), 1)
    x = np.arange(low[0], high[0], step[0])
    y = np.arange(low[1], high[1], step[1])
    z = np.arange(low[2], high[2], step[2])
    selection_zyx = (slice(low[2], high[2], step[2]), slice(low[1], high[1], step[1]), slice(low[0], high[0], step[0]))
    world_xpct_surfaces = sample_exterior_xpct_faces(raw_array, selected, world_corner_sets)
    xpct_surfaces = [
        (to_display(points, display_origin, world_from_mri), values)
        for points, values in world_xpct_surfaces
    ]
    mask = np.asarray(mask_array[selection_zyx])
    geometries = []
    for corners in world_corner_sets:
        center, _, inverse = center_basis(corners)
        geometries.append((center, inverse))
    all_display_corners = np.concatenate(display_corner_sets, axis=0)
    display_span = np.ptp(all_display_corners, axis=0)
    display_pad = np.maximum(display_span * 0.04, 1.0)
    bounds = (
        all_display_corners.min(axis=0) - display_pad,
        all_display_corners.max(axis=0) + display_pad,
    )

    figure_mri = plt.figure(figsize=(7.2, 6.2), facecolor="black")
    mri_axis = figure_mri.add_subplot(111, projection="3d")
    plot_mri_voxels(mri_axis, selected, predictor, display_corner_sets, bounds)
    save_panel(figure_mri, args.output_dir / f"neighborhood_{args.rank:02d}_mri_voxels")

    figure_xpct = plt.figure(figsize=(7.2, 6.2), facecolor="black")
    xpct_axis = figure_xpct.add_subplot(111, projection="3d")
    plot_xpct_surfaces(xpct_axis, xpct_surfaces, bounds)
    for corners in display_corner_sets:
        add_wireframe(xpct_axis, corners)
    save_panel(figure_xpct, args.output_dir / f"neighborhood_{args.rank:02d}_xpct_volume")

    figure_cells = plt.figure(figsize=(7.2, 6.2), facecolor="black")
    cell_axis = figure_cells.add_subplot(111, projection="3d")
    cell_report = plot_cells(
        cell_axis, mask, (x, y, z), geometries, bounds,
        display_origin, world_from_mri,
        args.cell_id_step, args.max_cell_points,
    )
    for corners in display_corner_sets:
        add_wireframe(cell_axis, corners, alpha=0.30)
    save_panel(figure_cells, args.output_dir / f"neighborhood_{args.rank:02d}_cell_instances")

    figure_radius = plt.figure(figsize=(7.2, 6.2), facecolor="black")
    radius_axis = figure_radius.add_subplot(111, projection="3d")
    plot_registered_voxels(
        radius_axis, selected, TARGET, display_corner_sets, bounds,
        TARGET_PANEL_TITLE, "viridis",
    )
    save_panel(
        figure_radius,
        args.output_dir / f"neighborhood_{args.rank:02d}_mean_cell_radius",
    )

    figure_corr, corr_axis = plt.subplots(figsize=(7.2, 6.2), facecolor="black")
    plot_correlation(corr_axis, full, selected, predictor)
    figure_corr.tight_layout()
    save_panel(figure_corr, args.output_dir / f"neighborhood_{args.rank:02d}_correlation")

    overview = plt.figure(figsize=(19.2, 4.4), facecolor="black")
    axes = [overview.add_subplot(1, 5, index, projection="3d") for index in (1, 2, 3, 4)]
    corr = overview.add_subplot(1, 5, 5)
    plot_mri_voxels(axes[0], selected, predictor, display_corner_sets, bounds)
    plot_xpct_surfaces(axes[1], xpct_surfaces, bounds)
    plot_cells(
        axes[2], mask, (x, y, z), geometries, bounds,
        display_origin, world_from_mri,
        args.cell_id_step, args.max_cell_points,
    )
    for corners in display_corner_sets:
        add_wireframe(axes[1], corners)
        add_wireframe(axes[2], corners, alpha=0.30)
    plot_registered_voxels(
        axes[3], selected, TARGET, display_corner_sets, bounds,
        TARGET_PANEL_TITLE, "viridis",
    )
    plot_correlation(corr, full, selected, predictor)
    overview.suptitle(
        f"{dataset} | tomogram {tomo} | illustrative registration (not independent validation)",
        color="white", fontsize=13,
    )
    overview.tight_layout(rect=(0, 0, 1, 0.94))
    save_panel(overview, args.output_dir / f"neighborhood_{args.rank:02d}_overview", dpi=200)

    report = {
        "rank": args.rank,
        "dataset_id": dataset,
        "predictor": predictor,
        "specimen": specimen,
        "tomogram": tomo,
        "raw_store": str(raw_store),
        "merged_cell_mask": str(mask_store),
        "bbox_low_ijk": low.tolist(),
        "bbox_high_ijk_exclusive": high.tolist(),
        "render_stride_ijk": step.tolist(),
        "display_coordinate_frame": "MRI-axis-aligned rigid transform",
        "display_projection": "orthographic",
        "display_camera_elevation_deg": 24,
        "display_camera_azimuth_deg": -55,
        "display_origin_xpct_ijk": display_origin.tolist(),
        "display_world_from_mri_orthogonal": world_from_mri.tolist(),
        "display_transform_determinant": float(np.linalg.det(world_from_mri)),
        "displayed_histology_target": TARGET,
        "displayed_histology_label": TARGET_AXIS_LABEL,
        "xpct_exterior_face_count": len(xpct_surfaces),
        "xpct_samples_per_face_axis": 48,
        "xpct_interior_loaded": False,
        "cell_id_step": args.cell_id_step,
        **cell_report,
        "selection_note": "Selected for visual concordance; not an independent performance estimate.",
    }
    (args.output_dir / f"neighborhood_{args.rank:02d}_render_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
