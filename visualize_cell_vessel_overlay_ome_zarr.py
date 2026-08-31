#!/usr/bin/env python3
"""Render matching raw, cell, and vessel center sections from OME-Zarr."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def open_array(path: Path, key: str):
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr first: python -m pip install zarr") from exc

    root = zarr.open_group(str(path), mode="r")
    if key not in root:
        raise KeyError(f"Array key {key!r} not found in {path}; keys={list(root.keys())}")
    array = root[key]
    if len(array.shape) != 3:
        raise ValueError(f"Expected a 3D ZYX array, got {array.shape} from {path}")
    return array


def parse_color(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise ValueError("Colors must contain three integers from 0 to 255")
    return values


def extract_section(array, plane: str, index: int) -> np.ndarray:
    if plane == "xz":
        return np.asarray(array[:, index, :])
    if plane == "yz":
        return np.asarray(array[:, :, index])
    raise ValueError(f"Unsupported plane: {plane}")


def normalize_raw(raw: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Raw section contains no finite values")
    low, high = (float(value) for value in np.percentile(finite, [low_pct, high_pct]))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8)


def resize(array: np.ndarray, scale: float, resample: int) -> np.ndarray:
    height, width = array.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(array).resize(size, resample))


def tint_from_raw(
    raw_u8: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    gray = np.stack([raw_u8, raw_u8, raw_u8], axis=-1).astype(np.float32)
    result = gray.copy()
    result[mask] = (
        (1.0 - alpha) * gray[mask]
        + alpha * np.asarray(color, dtype=np.float32)
    )
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def combined_overlay(
    raw_u8: np.ndarray,
    cells: np.ndarray,
    vessels: np.ndarray,
    cell_color: tuple[int, int, int],
    vessel_color: tuple[int, int, int],
    cell_alpha: float,
    vessel_alpha: float,
) -> np.ndarray:
    # Vessel pixels are rendered from the raw image, so blue fully takes
    # precedence over a red cell overlay where the semantic masks overlap.
    gray = np.stack([raw_u8, raw_u8, raw_u8], axis=-1).astype(np.float32)
    result = gray.copy()
    cell_only = cells & ~vessels
    result[cell_only] = (
        (1.0 - cell_alpha) * gray[cell_only]
        + cell_alpha * np.asarray(cell_color, dtype=np.float32)
    )
    result[vessels] = (
        (1.0 - vessel_alpha) * gray[vessels]
        + vessel_alpha * np.asarray(vessel_color, dtype=np.float32)
    )
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def save_panels(path: Path, panels: list[np.ndarray]) -> None:
    rgb_panels = [
        np.stack([panel, panel, panel], axis=-1) if panel.ndim == 2 else panel
        for panel in panels
    ]
    separator = np.full((rgb_panels[0].shape[0], 6, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for index, panel in enumerate(rgb_panels):
        if index:
            pieces.append(separator)
        pieces.append(panel)
    Image.fromarray(np.concatenate(pieces, axis=1), mode="RGB").save(path)


def render_plane(
    raw_array,
    cell_array,
    vessel_array,
    plane: str,
    index: int,
    args: argparse.Namespace,
    cell_color: tuple[int, int, int],
    vessel_color: tuple[int, int, int],
) -> dict[str, object]:
    progress(f"Reading {plane.upper()} section at index {index}")
    raw = extract_section(raw_array, plane, index)
    cells = extract_section(cell_array, plane, index) > 0
    vessels = extract_section(vessel_array, plane, index) > 0
    if raw.shape != cells.shape or raw.shape != vessels.shape:
        raise ValueError(
            f"{plane.upper()} section mismatch: raw={raw.shape}, "
            f"cells={cells.shape}, vessels={vessels.shape}"
        )

    raw_u8 = normalize_raw(raw, args.low_percentile, args.high_percentile)
    raw_display = resize(raw_u8, args.scale, Image.Resampling.LANCZOS)
    cells_display = resize(
        cells.astype(np.uint8) * 255, args.scale, Image.Resampling.NEAREST
    ) > 0
    vessels_display = resize(
        vessels.astype(np.uint8) * 255, args.scale, Image.Resampling.NEAREST
    ) > 0
    cells_rgb = tint_from_raw(
        raw_display, cells_display, cell_color, args.cell_alpha
    )
    combined_rgb = combined_overlay(
        raw_display,
        cells_display,
        vessels_display,
        cell_color,
        vessel_color,
        args.cell_alpha,
        args.vessel_alpha,
    )

    stem = f"{plane}_index{index:05d}"
    Image.fromarray(raw_display, mode="L").save(args.outdir / f"raw_{stem}.png")
    Image.fromarray(cells_rgb, mode="RGB").save(
        args.outdir / f"cells_red_{stem}.png"
    )
    Image.fromarray(combined_rgb, mode="RGB").save(
        args.outdir / f"cells_red_vessels_blue_{stem}.png"
    )
    Image.fromarray(cells_display.astype(np.uint8) * 255, mode="L").save(
        args.outdir / f"cell_mask_{stem}.png"
    )
    Image.fromarray(vessels_display.astype(np.uint8) * 255, mode="L").save(
        args.outdir / f"vessel_mask_{stem}.png"
    )
    save_panels(
        args.outdir / f"raw_vs_cells_vs_cells_vessels_{stem}.png",
        [raw_display, cells_rgb, combined_rgb],
    )
    save_panels(
        args.outdir / f"raw_vs_cells_vessels_{stem}.png",
        [raw_display, combined_rgb],
    )
    return {
        "plane": plane,
        "fixed_axis": "y" if plane == "xz" else "x",
        "fixed_index": index,
        "source_shape": list(raw.shape),
        "display_shape": list(raw_display.shape),
        "cell_foreground_pixels": int(np.count_nonzero(cells)),
        "vessel_foreground_pixels": int(np.count_nonzero(vessels)),
        "overlap_pixels": int(np.count_nonzero(cells & vessels)),
    }


def accumulate_foreground_on_grid(source, destination: np.ndarray) -> int:
    """Max-pool a semantic ZYX array onto a compact display grid."""
    source_shape = tuple(int(value) for value in source.shape)
    target_shape = destination.shape
    z_chunk = int(source.chunks[0]) if getattr(source, "chunks", None) else 8
    source_foreground = 0
    for z0 in range(0, source_shape[0], z_chunk):
        z1 = min(source_shape[0], z0 + z_chunk)
        block = np.asarray(source[z0:z1, :, :]) > 0
        local_z, local_y, local_x = np.nonzero(block)
        source_foreground += int(local_z.size)
        if local_z.size:
            target_z = np.minimum(
                ((local_z + z0) * target_shape[0]) // source_shape[0],
                target_shape[0] - 1,
            )
            target_y = np.minimum(
                (local_y * target_shape[1]) // source_shape[1],
                target_shape[1] - 1,
            )
            target_x = np.minimum(
                (local_x * target_shape[2]) // source_shape[2],
                target_shape[2] - 1,
            )
            flat = np.ravel_multi_index(
                (target_z, target_y, target_x), target_shape
            )
            destination.reshape(-1)[np.unique(flat)] = True
        progress(
            f"3D display grid: source z={z0}:{z1}/{source_shape[0]}, "
            f"foreground={source_foreground}"
        )
    return source_foreground


def upsample_nearest_on_grid(source, destination: np.ndarray) -> int:
    """Nearest-neighbor upsample a semantic source onto the destination grid."""
    source_shape = tuple(int(value) for value in source.shape)
    target_shape = destination.shape
    source_binary = np.asarray(source[:, :, :]) > 0
    source_foreground = int(np.count_nonzero(source_binary))
    source_indices = tuple(
        np.minimum(
            np.floor(
                (np.arange(target_shape[axis], dtype=np.float64) + 0.5)
                * source_shape[axis]
                / target_shape[axis]
            ).astype(np.int64),
            source_shape[axis] - 1,
        )
        for axis in range(3)
    )
    target_z_chunk = 8
    for z0 in range(0, target_shape[0], target_z_chunk):
        z1 = min(target_shape[0], z0 + target_z_chunk)
        destination[z0:z1] |= source_binary[np.ix_(
            source_indices[0][z0:z1], source_indices[1], source_indices[2]
        )]
        progress(f"3D coarse-to-mid nearest upsampling z={z0}:{z1}/{target_shape[0]}")
    return source_foreground


def apply_scale_postprocess(
    source_array, thresholds_path: Path, scale: str
) -> tuple[np.ndarray, dict[str, object]]:
    from vessel_analytical_postprocess import component_window_features

    config = json.loads(thresholds_path.read_text(encoding="utf-8"))
    schema_version = config.get("schema_version")
    if schema_version == 3 and scale == "coarse":
        rule = config["rule"]
        rule_source = "coarse stage-2 deployment rule"
    elif schema_version == 2:
        try:
            rule = config["scale_specific_rules"][scale]
        except KeyError as exc:
            raise ValueError(
                f"No scale-specific {scale} rule in {thresholds_path}"
            ) from exc
        rule_source = f"scale-specific {scale} deployment rule"
    else:
        raise ValueError(
            f"Unsupported schema {schema_version} for {scale} postprocessing"
        )
    source = np.asarray(source_array[:, :, :]) > 0
    progress(
        f"Applying calibrated {scale} postprocessor: "
        f"window={rule['window_size']} statistic={rule['score_statistic']} "
        f"threshold={rule['tubularity_threshold']}"
    )
    labels, frame = component_window_features(
        source,
        None,
        [int(rule["window_size"])],
        int(config["min_foreground_voxels_per_window"]),
        0.5,
        0.1,
    )
    keep = frame[rule["score_statistic"]].to_numpy() >= float(
        rule["tubularity_threshold"]
    )
    if bool(config.get("keep_components_touching_volume_boundary", True)):
        keep |= frame["touches_volume_boundary"].to_numpy(dtype=bool)
    keep_ids = frame.loc[keep, "component_id"].to_numpy(dtype=np.int64)
    filtered = np.isin(labels, keep_ids)
    summary = {
        "thresholds": str(thresholds_path.resolve()),
        "scale": scale,
        "rule_source": rule_source,
        "schema_version": config["schema_version"],
        "method": config.get("method"),
        "rule": rule,
        "min_foreground_voxels_per_window": config[
            "min_foreground_voxels_per_window"
        ],
        "keep_components_touching_volume_boundary": bool(
            config.get("keep_components_touching_volume_boundary", True)
        ),
        "components_before": int(len(frame)),
        "components_kept": int(np.count_nonzero(keep)),
        "foreground_voxels_before": int(np.count_nonzero(source)),
        "foreground_voxels_after": int(np.count_nonzero(filtered)),
    }
    progress(
        f"{scale.capitalize()} postprocessor kept {summary['components_kept']}/"
        f"{summary['components_before']} components and "
        f"{summary['foreground_voxels_after']}/"
        f"{summary['foreground_voxels_before']} foreground voxels"
    )
    return filtered, summary


def boundary_coordinates(
    grid: np.ndarray, max_points: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    from scipy.ndimage import binary_erosion

    boundary = grid & ~binary_erosion(grid)
    coordinates = np.nonzero(boundary)
    total = int(coordinates[0].size)
    if total == 0:
        raise RuntimeError("A mid+coarse union contains no vessel boundary points")
    if total > max_points:
        selected = np.sort(
            np.random.default_rng(seed).choice(total, size=max_points, replace=False)
        )
        coordinates = tuple(axis[selected] for axis in coordinates)
    return coordinates, total


def vessel_trace(
    go,
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int, int],
    cylinder_diameter_mm: float,
    cylinder_height_mm: float,
    name: str,
    color: str,
    opacity: float,
    visible: bool | str,
):
    z_idx, y_idx, x_idx = coordinates
    nz, ny, nx = shape
    x_mm = ((x_idx.astype(np.float32) + 0.5) / nx - 0.5) * cylinder_diameter_mm
    y_mm = ((y_idx.astype(np.float32) + 0.5) / ny - 0.5) * cylinder_diameter_mm
    z_mm = ((z_idx.astype(np.float32) + 0.5) / nz - 0.5) * cylinder_height_mm
    return go.Scatter3d(
        x=x_mm,
        y=y_mm,
        z=z_mm,
        mode="markers",
        name=name,
        visible=visible,
        marker=dict(size=1.6, color=color, opacity=opacity),
        hovertemplate="x=%{x:.3f} mm<br>y=%{y:.3f} mm<br>z=%{z:.3f} mm<extra></extra>",
    )


def write_vessel_html(
    mid_array,
    coarse_array,
    outdir: Path,
    max_points: int,
    cylinder_diameter_mm: float,
    cylinder_height_mm: float,
    seed: int,
    coarse_postprocess_thresholds: Path | None,
    mid_postprocess_thresholds: Path | None,
) -> dict[str, object]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "The 3D HTML output requires plotly and scipy in the active environment"
        ) from exc

    mid_shape = tuple(int(value) for value in mid_array.shape)
    display_shape = tuple(max(2, value) for value in mid_shape)
    mid_grid = np.zeros(display_shape, dtype=bool)
    coarse_grid = np.zeros(display_shape, dtype=bool)
    progress(
        f"Building mid+coarse 3D display grid ZYX={display_shape}; "
        "unscaled vessels are intentionally excluded"
    )
    mid_foreground = accumulate_foreground_on_grid(mid_array, mid_grid)
    coarse_foreground = upsample_nearest_on_grid(coarse_array, coarse_grid)
    before_grid = mid_grid | coarse_grid

    coarse_postprocess_summary = None
    mid_postprocess_summary = None
    after_grid = None
    if (
        coarse_postprocess_thresholds is not None
        or mid_postprocess_thresholds is not None
    ):
        if mid_postprocess_thresholds is not None:
            filtered_mid, mid_postprocess_summary = apply_scale_postprocess(
                mid_array, mid_postprocess_thresholds, "mid"
            )
            after_grid = np.asarray(filtered_mid, dtype=bool).copy()
        else:
            after_grid = mid_grid.copy()
        if coarse_postprocess_thresholds is not None:
            filtered_coarse, coarse_postprocess_summary = apply_scale_postprocess(
                coarse_array, coarse_postprocess_thresholds, "coarse"
            )
        else:
            filtered_coarse = np.asarray(coarse_array[:, :, :]) > 0
        filtered_coarse_grid = np.zeros(display_shape, dtype=bool)
        upsample_nearest_on_grid(filtered_coarse, filtered_coarse_grid)
        after_grid |= filtered_coarse_grid

    nz, ny, nx = display_shape
    y_normalized = ((np.arange(ny, dtype=np.float32) + 0.5) / ny - 0.5) * 2.0
    x_normalized = ((np.arange(nx, dtype=np.float32) + 0.5) / nx - 0.5) * 2.0
    inside_xy = (
        y_normalized[:, np.newaxis] ** 2 + x_normalized[np.newaxis, :] ** 2
    ) <= 1.0
    outside_before = int(np.count_nonzero(before_grid & ~inside_xy[np.newaxis]))
    before_grid &= inside_xy[np.newaxis]
    before_coordinates, before_boundary_points = boundary_coordinates(
        before_grid, max_points, seed
    )
    vessel_traces = [
        vessel_trace(
            go,
            before_coordinates,
            display_shape,
            cylinder_diameter_mm,
            cylinder_height_mm,
            "Before postprocessing",
            "#4D9BFF",
            0.72,
            after_grid is None,
        )
    ]
    outside_after = None
    after_boundary_points = None
    if after_grid is not None:
        outside_after = int(np.count_nonzero(after_grid & ~inside_xy[np.newaxis]))
        after_grid &= inside_xy[np.newaxis]
        after_coordinates, after_boundary_points = boundary_coordinates(
            after_grid, max_points, seed + 1
        )
        vessel_traces.append(
            vessel_trace(
                go,
                after_coordinates,
                display_shape,
                cylinder_diameter_mm,
                cylinder_height_mm,
                "After mid + coarse postprocessing",
                "#0057D9",
                0.86,
                True,
            )
        )

    theta = np.linspace(0.0, 2.0 * math.pi, 129)
    radius = cylinder_diameter_mm / 2.0
    cylinder_traces = []
    for z_value, name in (
        (-cylinder_height_mm / 2.0, "Cylinder"),
        (cylinder_height_mm / 2.0, None),
    ):
        cylinder_traces.append(
            go.Scatter3d(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                z=np.full(theta.shape, z_value),
                mode="lines",
                name=name,
                showlegend=name is not None,
                line=dict(color="rgba(70,70,70,0.55)", width=3),
                hoverinfo="skip",
            )
        )
    for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        cylinder_traces.append(
            go.Scatter3d(
                x=[radius * math.cos(angle)] * 2,
                y=[radius * math.sin(angle)] * 2,
                z=[-cylinder_height_mm / 2.0, cylinder_height_mm / 2.0],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(100,100,100,0.25)", width=2),
                hoverinfo="skip",
            )
        )

    figure = go.Figure(data=[*vessel_traces, *cylinder_traces])
    if after_grid is not None:
        cylinder_visible = [True] * len(cylinder_traces)
        figure.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    xanchor="center",
                    y=1.08,
                    buttons=[
                        dict(
                            label="Before",
                            method="update",
                            args=[{"visible": [True, False, *cylinder_visible]}],
                        ),
                        dict(
                            label="After",
                            method="update",
                            args=[{"visible": [False, True, *cylinder_visible]}],
                        ),
                        dict(
                            label="Both",
                            method="update",
                            args=[{"visible": [True, True, *cylinder_visible]}],
                        ),
                    ],
                )
            ]
        )
    figure.update_layout(
        title="Mid + coarse vessel segmentation: before/after scale-specific postprocessing",
        template="plotly_white",
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05)),
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(x=0.01, y=0.99),
    )
    html_path = outdir / "vessels_mid_coarse_3d.html"
    figure.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    progress(f"Wrote interactive 3D HTML: {html_path}")
    return {
        "html": str(html_path.resolve()),
        "display_grid_shape_zyx": list(display_shape),
        "display_union_voxels_inside_cylinder_before": int(np.count_nonzero(before_grid)),
        "display_union_voxels_inside_cylinder_after": int(np.count_nonzero(after_grid))
        if after_grid is not None
        else None,
        "outside_cylinder_voxels_removed_before": outside_before,
        "outside_cylinder_voxels_removed_after": outside_after,
        "boundary_points_before_postprocess_before_sampling": before_boundary_points,
        "boundary_points_after_postprocess_before_sampling": after_boundary_points,
        "boundary_points_rendered_per_trace_max": max_points,
        "max_points": max_points,
        "sampling_seed": seed,
        "mid_source_shape_zyx": list(mid_array.shape),
        "coarse_source_shape_zyx": list(coarse_array.shape),
        "mid_source_foreground_voxels": mid_foreground,
        "coarse_source_foreground_voxels": coarse_foreground,
        "mid_postprocess": mid_postprocess_summary,
        "coarse_postprocess": coarse_postprocess_summary,
        "cylinder_diameter_mm": cylinder_diameter_mm,
        "cylinder_height_mm": cylinder_height_mm,
        "representation": "before/after boundary-point rendering on the native mid grid; mid and coarse are filtered on their native grids, then filtered coarse is upsampled to mid by nearest-neighbor center-coordinate mapping",
        "excluded_scale": "unscaled",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--vessels", type=Path, required=True)
    parser.add_argument("--raw-key", default="0")
    parser.add_argument("--cell-key", default="0")
    parser.add_argument("--vessel-key", default="0")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plane", choices=("xz", "yz", "both"), default="both")
    parser.add_argument("--x-index", type=int, default=None)
    parser.add_argument("--y-index", type=int, default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--cell-color", default="255,20,20")
    parser.add_argument("--vessel-color", default="20,100,255")
    parser.add_argument("--cell-alpha", type=float, default=0.62)
    parser.add_argument("--vessel-alpha", type=float, default=0.72)
    parser.add_argument("--mid-vessels", type=Path, default=None)
    parser.add_argument("--coarse-vessels", type=Path, default=None)
    parser.add_argument("--scale-mask-key", default="0")
    parser.add_argument("--html-max-points", type=int, default=250000)
    parser.add_argument("--cylinder-diameter-mm", type=float, default=1.2)
    parser.add_argument("--cylinder-height-mm", type=float, default=1.2)
    parser.add_argument("--html-seed", type=int, default=46)
    parser.add_argument("--coarse-postprocess-thresholds", type=Path, default=None)
    parser.add_argument("--mid-postprocess-thresholds", type=Path, default=None)
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("scale must be greater than zero")
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if not 0 <= args.cell_alpha <= 1 or not 0 <= args.vessel_alpha <= 1:
        raise ValueError("overlay alpha values must be between 0 and 1")
    if (args.mid_vessels is None) != (args.coarse_vessels is None):
        raise ValueError("Pass both --mid-vessels and --coarse-vessels, or neither")
    if args.html_max_points <= 0:
        raise ValueError("html-max-points must be positive")
    if args.cylinder_diameter_mm <= 0 or args.cylinder_height_mm <= 0:
        raise ValueError("Cylinder dimensions must be positive")

    raw_array = open_array(args.raw, args.raw_key)
    cell_array = open_array(args.cells, args.cell_key)
    vessel_array = open_array(args.vessels, args.vessel_key)
    shape = tuple(int(value) for value in raw_array.shape)
    for name, array in (("cells", cell_array), ("vessels", vessel_array)):
        other_shape = tuple(int(value) for value in array.shape)
        if other_shape != shape:
            raise ValueError(f"Raw/{name} volume mismatch: {shape} vs {other_shape}")

    x_index = shape[2] // 2 if args.x_index is None else args.x_index
    y_index = shape[1] // 2 if args.y_index is None else args.y_index
    if not 0 <= x_index < shape[2] or not 0 <= y_index < shape[1]:
        raise ValueError(f"Requested indices outside volume shape {shape}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cell_color = parse_color(args.cell_color)
    vessel_color = parse_color(args.vessel_color)
    planes = ("xz", "yz") if args.plane == "both" else (args.plane,)
    sections = [
        render_plane(
            raw_array,
            cell_array,
            vessel_array,
            plane,
            y_index if plane == "xz" else x_index,
            args,
            cell_color,
            vessel_color,
        )
        for plane in planes
    ]
    html_3d = None
    if args.mid_vessels is not None and args.coarse_vessels is not None:
        mid_array = open_array(args.mid_vessels, args.scale_mask_key)
        coarse_array = open_array(args.coarse_vessels, args.scale_mask_key)
        html_3d = write_vessel_html(
            mid_array,
            coarse_array,
            args.outdir,
            args.html_max_points,
            args.cylinder_diameter_mm,
            args.cylinder_height_mm,
            args.html_seed,
            args.coarse_postprocess_thresholds,
            args.mid_postprocess_thresholds,
        )
    metadata = {
        "raw": str(args.raw.resolve()),
        "cells": str(args.cells.resolve()),
        "vessels": str(args.vessels.resolve()),
        "volume_shape_zyx": list(shape),
        "x_index": x_index,
        "y_index": y_index,
        "scale": args.scale,
        "cell_color_rgb": list(cell_color),
        "vessel_color_rgb": list(vessel_color),
        "cell_alpha": args.cell_alpha,
        "vessel_alpha": args.vessel_alpha,
        "overlap_policy": "vessel blue takes precedence over cell red",
        "sections": sections,
        "mid_coarse_3d": html_3d,
    }
    (args.outdir / "overlay_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    progress(f"Wrote overlays to {args.outdir}")


if __name__ == "__main__":
    main()
