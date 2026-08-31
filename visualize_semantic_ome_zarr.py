#!/usr/bin/env python3
"""Visualize semantic OME-Zarr labels and derive physical soma statistics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def open_array(path: Path, array_key: str):
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr first: python -m pip install zarr") from exc

    root = zarr.open_group(str(path), mode="r")
    if array_key not in root:
        raise KeyError(
            f"Array key {array_key!r} not found in {path}; "
            f"available keys: {list(root.keys())}"
        )
    arr = root[array_key]
    if len(arr.shape) != 3:
        raise ValueError(f"Expected a 3D ZYX array, got {arr.shape} from {path}")
    return arr


def parse_positive_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("Expected three positive integers in Z,Y,X order")
    return values


def parse_color(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise ValueError("color must contain three integers from 0 to 255")
    return values


def normalize_raw(
    raw: np.ndarray, low_pct: float, high_pct: float
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(raw, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Raw section contains no finite values")

    low, high = np.percentile(finite, [low_pct, high_pct])
    low, high = float(low), float(high)
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), low, high

    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8), low, high


def semantic_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return binary & ~interior


def resize_2d(array: np.ndarray, scale: float, resample: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    height, width = array.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(array).resize(size, resample))


def color_overlay(
    raw: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    gray = np.stack([raw, raw, raw], axis=-1).astype(np.float32)
    result = gray.copy()
    tint = np.asarray(color, dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    result[mask] = (1.0 - a) * gray[mask] + a * tint
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def blend_grayscale(base_rgb: np.ndarray, scalar_u8: np.ndarray, alpha: float) -> np.ndarray:
    scalar_rgb = np.stack([scalar_u8, scalar_u8, scalar_u8], axis=-1).astype(np.float32)
    base = np.asarray(base_rgb, dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.rint(np.clip((1.0 - a) * base + a * scalar_rgb, 0, 255)).astype(np.uint8)


def save_comparison(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    if left.ndim == 2:
        left = np.stack([left, left, left], axis=-1)
    if right.ndim == 2:
        right = np.stack([right, right, right], axis=-1)
    separator = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([left, separator, right], axis=1), mode="RGB").save(path)


def extract_section(arr, plane: str, index: int) -> np.ndarray:
    if plane == "xz":
        return np.asarray(arr[:, index, :])
    if plane == "yz":
        return np.asarray(arr[:, :, index])
    raise ValueError(f"Unsupported plane: {plane}")


def ensure_capacity(arrays: tuple[np.ndarray, ...], required: int) -> tuple[np.ndarray, ...]:
    if required <= arrays[0].size:
        return arrays
    new_size = max(required, arrays[0].size * 2)
    grown = []
    for array in arrays:
        expanded = np.zeros(new_size, dtype=array.dtype)
        expanded[: array.size] = array
        grown.append(expanded)
    return tuple(grown)


def axis_bin_indices(length: int, spacing_um: float, bin_um: float) -> np.ndarray:
    centers_um = (np.arange(length, dtype=np.float64) + 0.5) * spacing_um
    return np.floor(centers_um / bin_um).astype(np.int64)


def scan_labels(
    label_arr,
    voxel_um_zyx: tuple[float, float, float],
    spatial_bin_um: float,
    scan_block_zyx: tuple[int, int, int],
) -> dict[str, np.ndarray | tuple[int, int, int] | int]:
    """Stream labels once to obtain instance sizes/centroids and soma occupancy."""
    shape = tuple(int(value) for value in label_arr.shape)
    axis_bins = tuple(
        axis_bin_indices(shape[axis], voxel_um_zyx[axis], spatial_bin_um)
        for axis in range(3)
    )
    grid_shape = tuple(int(indices.max()) + 1 for indices in axis_bins)
    soma_voxels_grid = np.zeros(grid_shape, dtype=np.uint64)

    counts = np.zeros(1024, dtype=np.uint64)
    sum_z = np.zeros(1024, dtype=np.float64)
    sum_y = np.zeros(1024, dtype=np.float64)
    sum_x = np.zeros(1024, dtype=np.float64)
    processed = 0
    total_blocks = math.prod(
        math.ceil(shape[axis] / scan_block_zyx[axis]) for axis in range(3)
    )

    for z0 in range(0, shape[0], scan_block_zyx[0]):
        z1 = min(shape[0], z0 + scan_block_zyx[0])
        for y0 in range(0, shape[1], scan_block_zyx[1]):
            y1 = min(shape[1], y0 + scan_block_zyx[1])
            for x0 in range(0, shape[2], scan_block_zyx[2]):
                x1 = min(shape[2], x0 + scan_block_zyx[2])
                processed += 1
                block = np.asarray(label_arr[z0:z1, y0:y1, x0:x1])
                foreground = block > 0
                if foreground.any():
                    local_z, local_y, local_x = np.nonzero(foreground)
                    labels = block[foreground].astype(np.int64, copy=False)
                    unique, inverse, local_counts = np.unique(
                        labels, return_inverse=True, return_counts=True
                    )
                    required = int(unique[-1]) + 1
                    counts, sum_z, sum_y, sum_x = ensure_capacity(
                        (counts, sum_z, sum_y, sum_x), required
                    )
                    counts[unique] += local_counts.astype(np.uint64)
                    sum_z[unique] += np.bincount(
                        inverse, weights=(local_z + z0).astype(np.float64), minlength=unique.size
                    )
                    sum_y[unique] += np.bincount(
                        inverse, weights=(local_y + y0).astype(np.float64), minlength=unique.size
                    )
                    sum_x[unique] += np.bincount(
                        inverse, weights=(local_x + x0).astype(np.float64), minlength=unique.size
                    )

                    bz = axis_bins[0][local_z + z0]
                    by = axis_bins[1][local_y + y0]
                    bx = axis_bins[2][local_x + x0]
                    flat_bins = np.ravel_multi_index((bz, by, bx), grid_shape)
                    soma_voxels_grid += np.bincount(
                        flat_bins, minlength=soma_voxels_grid.size
                    ).reshape(grid_shape).astype(np.uint64)

                if processed == 1 or processed % 100 == 0 or processed == total_blocks:
                    progress(
                        f"label scan block {processed}/{total_blocks}; "
                        f"foreground={int(np.count_nonzero(foreground))}"
                    )

    instance_ids = np.flatnonzero(counts).astype(np.int64)
    if instance_ids.size < 2:
        raise RuntimeError(
            "Fewer than two positive instance IDs were found. Equivalent cell-radius "
            "statistics require the original Cellpose instance-label output, not a binary mask."
        )

    instance_counts = counts[instance_ids]
    centroids = np.column_stack(
        [sum_z[instance_ids], sum_y[instance_ids], sum_x[instance_ids]]
    ) / instance_counts[:, None]

    axis_voxel_counts = tuple(
        np.bincount(indices, minlength=grid_shape[axis]).astype(np.uint64)
        for axis, indices in enumerate(axis_bins)
    )
    bin_voxel_capacity = (
        axis_voxel_counts[0][:, None, None]
        * axis_voxel_counts[1][None, :, None]
        * axis_voxel_counts[2][None, None, :]
    )

    return {
        "instance_ids": instance_ids,
        "instance_voxels": instance_counts,
        "centroids_zyx": centroids,
        "soma_voxels_grid": soma_voxels_grid,
        "bin_voxel_capacity": bin_voxel_capacity,
        "axis_bins": axis_bins,
        "grid_shape": grid_shape,
        "total_foreground_voxels": int(instance_counts.sum()),
    }


def derive_soma_fields(
    stats: dict[str, object],
    voxel_um_zyx: tuple[float, float, float],
    spatial_bin_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(stats["instance_ids"])
    counts = np.asarray(stats["instance_voxels"], dtype=np.float64)
    centroids = np.asarray(stats["centroids_zyx"], dtype=np.float64)
    grid_shape = tuple(int(v) for v in stats["grid_shape"])

    voxel_volume_um3 = float(np.prod(voxel_um_zyx))
    volumes_um3 = counts * voxel_volume_um3
    radii_um = np.cbrt(3.0 * volumes_um3 / (4.0 * math.pi))

    centroid_bins = np.floor(
        centroids * np.asarray(voxel_um_zyx)[None, :] / spatial_bin_um
    ).astype(np.int64)
    for axis in range(3):
        centroid_bins[:, axis] = np.clip(centroid_bins[:, axis], 0, grid_shape[axis] - 1)
    flat = np.ravel_multi_index(centroid_bins.T, grid_shape)
    radius_sum = np.bincount(flat, weights=radii_um, minlength=math.prod(grid_shape))
    cell_count = np.bincount(flat, minlength=math.prod(grid_shape))
    r_soma = np.zeros(math.prod(grid_shape), dtype=np.float64)
    np.divide(radius_sum, cell_count, out=r_soma, where=cell_count > 0)
    r_soma = r_soma.reshape(grid_shape)

    soma_voxels = np.asarray(stats["soma_voxels_grid"], dtype=np.float64)
    capacity = np.asarray(stats["bin_voxel_capacity"], dtype=np.float64)
    f_soma = np.zeros(grid_shape, dtype=np.float64)
    np.divide(soma_voxels, capacity, out=f_soma, where=capacity > 0)
    return ids, volumes_um3, radii_um, r_soma, f_soma


def save_radius_outputs(
    outdir: Path,
    instance_ids: np.ndarray,
    volumes_um3: np.ndarray,
    radii_um: np.ndarray,
    histogram_bins: int,
) -> dict[str, float | int]:
    per_instance = outdir / "cell_radius_per_instance.csv.gz"
    with gzip.open(per_instance, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_id", "volume_um3", "equivalent_sphere_radius_um"])
        writer.writerows(zip(instance_ids.tolist(), volumes_um3.tolist(), radii_um.tolist()))

    hist_counts, hist_edges = np.histogram(radii_um, bins=histogram_bins)
    with (outdir / "cell_radius_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius_bin_left_um", "radius_bin_right_um", "count"])
        writer.writerows(zip(hist_edges[:-1], hist_edges[1:], hist_counts))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(radii_um, bins=histogram_bins, color="#D62728", edgecolor="white", linewidth=0.4)
    axis.set_yscale("log", base=10)
    axis.axvline(float(np.median(radii_um)), color="black", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Equivalent-sphere cell radius (µm)")
    axis.set_ylabel("Number of segmented instances (log10 scale)")
    axis.set_title("Cell-radius distribution")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "cell_radius_histogram.png", dpi=220)
    fig.savefig(outdir / "cell_radius_histogram.pdf")
    plt.close(fig)

    return {
        "n_instances": int(radii_um.size),
        "radius_mean_um": float(np.mean(radii_um)),
        "radius_median_um": float(np.median(radii_um)),
        "radius_std_um": float(np.std(radii_um)),
        "radius_min_um": float(np.min(radii_um)),
        "radius_max_um": float(np.max(radii_um)),
    }


def save_f_soma_outputs(
    outdir: Path,
    f_soma: np.ndarray,
    soma_voxels: np.ndarray,
    bin_voxel_capacity: np.ndarray,
    voxel_volume_um3: float,
    histogram_bins: int,
) -> dict[str, float | int]:
    valid = bin_voxel_capacity > 0
    occupied = valid & (soma_voxels > 0)
    all_values = np.asarray(f_soma[valid], dtype=np.float64)
    occupied_values = np.asarray(f_soma[occupied], dtype=np.float64)

    with (outdir / "f_soma_250um_per_bin.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grid_z",
                "grid_y",
                "grid_x",
                "sampled_bin_volume_um3",
                "soma_volume_um3",
                "f_soma",
                "occupied",
            ]
        )
        for z, y, x in np.argwhere(valid):
            capacity = int(bin_voxel_capacity[z, y, x])
            foreground = int(soma_voxels[z, y, x])
            writer.writerow(
                [
                    int(z),
                    int(y),
                    int(x),
                    capacity * voxel_volume_um3,
                    foreground * voxel_volume_um3,
                    float(f_soma[z, y, x]),
                    int(foreground > 0),
                ]
            )

    hist_range = (0.0, max(1.0, float(all_values.max()) if all_values.size else 1.0))
    all_counts, edges = np.histogram(all_values, bins=histogram_bins, range=hist_range)
    occupied_counts, _ = np.histogram(
        occupied_values, bins=histogram_bins, range=hist_range
    )
    with (outdir / "f_soma_250um_histogram.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "f_soma_bin_left",
                "f_soma_bin_right",
                "all_valid_cubes_count",
                "occupied_cubes_count",
            ]
        )
        writer.writerows(zip(edges[:-1], edges[1:], all_counts, occupied_counts))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    axes[0].hist(
        all_values,
        bins=histogram_bins,
        range=hist_range,
        color="#4C4C4C",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_title("All sampled 250 um cubes")
    axes[1].hist(
        occupied_values,
        bins=histogram_bins,
        range=hist_range,
        color="#D62728",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set_title("Cubes with segmented soma")
    for axis in axes:
        axis.set_xlabel("Soma volume fraction, f_soma")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Number of spatial cubes")
    fig.suptitle("Local soma fraction on the 250 um grid")
    fig.tight_layout()
    fig.savefig(outdir / "f_soma_250um_histogram.png", dpi=220)
    fig.savefig(outdir / "f_soma_250um_histogram.pdf")
    plt.close(fig)

    return {
        "n_valid_cubes": int(all_values.size),
        "n_occupied_cubes": int(occupied_values.size),
        "n_zero_cubes": int(all_values.size - occupied_values.size),
        "zero_cube_fraction": float(1.0 - occupied_values.size / all_values.size)
        if all_values.size
        else 0.0,
        "mean_all_cubes": float(np.mean(all_values)) if all_values.size else 0.0,
        "median_all_cubes": float(np.median(all_values)) if all_values.size else 0.0,
        "mean_occupied_cubes": float(np.mean(occupied_values))
        if occupied_values.size
        else 0.0,
        "median_occupied_cubes": float(np.median(occupied_values))
        if occupied_values.size
        else 0.0,
        "max": float(np.max(all_values)) if all_values.size else 0.0,
    }


def scalar_plane_to_pixels(
    field: np.ndarray,
    plane: str,
    fixed_index: int,
    axis_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    if plane == "xz":
        grid_plane = field[:, axis_bins[1][fixed_index], :]
        return grid_plane[np.ix_(axis_bins[0], axis_bins[2])]
    grid_plane = field[:, :, axis_bins[2][fixed_index]]
    return grid_plane[np.ix_(axis_bins[0], axis_bins[1])]


def scalar_to_uint8(field: np.ndarray, maximum: float) -> np.ndarray:
    if maximum <= 0:
        return np.zeros(field.shape, dtype=np.uint8)
    return np.rint(np.clip(field / maximum, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_plane(
    raw_arr,
    label_arr,
    plane: str,
    index: int,
    outdir: Path,
    scale: float,
    low_pct: float,
    high_pct: float,
    color: tuple[int, int, int],
    mask_alpha: float,
    field_alpha: float,
    r_soma: np.ndarray,
    f_soma: np.ndarray,
    axis_bins: tuple[np.ndarray, np.ndarray, np.ndarray],
    r_soma_max: float,
    f_soma_max: float,
) -> dict[str, object]:
    progress(f"Reading central {plane.upper()} section at index {index}")
    raw_section = extract_section(raw_arr, plane, index)
    label_section = extract_section(label_arr, plane, index)
    if raw_section.shape != label_section.shape:
        raise ValueError(
            f"{plane.upper()} raw/label mismatch: {raw_section.shape} vs {label_section.shape}"
        )

    raw_u8, display_low, display_high = normalize_raw(raw_section, low_pct, high_pct)
    semantic = label_section > 0
    raw_display = resize_2d(raw_u8, scale, Image.Resampling.LANCZOS)
    semantic_display = resize_2d(
        semantic.astype(np.uint8) * 255, scale, Image.Resampling.NEAREST
    ) > 0
    raw_rgb = np.stack([raw_display, raw_display, raw_display], axis=-1)
    masked_rgb = color_overlay(raw_display, semantic_display, color, mask_alpha)
    boundary_rgb = color_overlay(raw_display, semantic_boundary(semantic_display), color, 1.0)

    r_pixels = scalar_plane_to_pixels(r_soma, plane, index, axis_bins)
    f_pixels = scalar_plane_to_pixels(f_soma, plane, index, axis_bins)
    r_u8 = resize_2d(scalar_to_uint8(r_pixels, r_soma_max), scale, Image.Resampling.NEAREST)
    f_u8 = resize_2d(scalar_to_uint8(f_pixels, f_soma_max), scale, Image.Resampling.NEAREST)
    r_overlay = blend_grayscale(masked_rgb, r_u8, field_alpha)
    f_overlay = blend_grayscale(masked_rgb, f_u8, field_alpha)

    stem = f"{plane}_index{index:05d}"
    Image.fromarray(raw_display, mode="L").save(outdir / f"raw_{stem}.png")
    Image.fromarray(semantic_display.astype(np.uint8) * 255, mode="L").save(
        outdir / f"mask_semantic_{stem}.png"
    )
    Image.fromarray(masked_rgb, mode="RGB").save(outdir / f"overlay_red_{stem}.png")
    Image.fromarray(boundary_rgb, mode="RGB").save(outdir / f"overlay_boundary_red_{stem}.png")
    save_comparison(outdir / f"raw_vs_overlay_red_{stem}.png", raw_rgb, masked_rgb)

    Image.fromarray(r_u8, mode="L").save(outdir / f"r_soma_250um_{stem}.png")
    Image.fromarray(r_overlay, mode="RGB").save(
        outdir / f"r_soma_250um_overlay_on_raw_mask_{stem}.png"
    )
    save_comparison(outdir / f"r_soma_250um_pure_vs_overlay_{stem}.png", r_u8, r_overlay)
    Image.fromarray(f_u8, mode="L").save(outdir / f"f_soma_250um_{stem}.png")
    Image.fromarray(f_overlay, mode="RGB").save(
        outdir / f"f_soma_250um_overlay_on_raw_mask_{stem}.png"
    )
    save_comparison(outdir / f"f_soma_250um_pure_vs_overlay_{stem}.png", f_u8, f_overlay)

    fixed_grid_index = int(
        axis_bins[1][index] if plane == "xz" else axis_bins[2][index]
    )
    return {
        "plane": plane,
        "fixed_axis": "y" if plane == "xz" else "x",
        "fixed_index": index,
        "fixed_250um_grid_index": fixed_grid_index,
        "source_shape": [int(v) for v in raw_section.shape],
        "display_shape": [int(v) for v in raw_display.shape],
        "foreground_pixels_source": int(np.count_nonzero(semantic)),
        "foreground_fraction_source": float(np.mean(semantic)),
        "display_low": display_low,
        "display_high": display_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--raw_key", default="0")
    parser.add_argument("--label_key", default="0")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plane", choices=("xz", "yz", "both"), default="both")
    parser.add_argument("--x_index", type=int, default=None)
    parser.add_argument("--y_index", type=int, default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--low_percentile", type=float, default=0.5)
    parser.add_argument("--high_percentile", type=float, default=99.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.62)
    parser.add_argument("--field_overlay_alpha", type=float, default=0.62)
    parser.add_argument("--color", default="255,20,20")
    parser.add_argument("--physical_x_um", type=float, default=1200.0)
    parser.add_argument("--physical_y_um", type=float, default=1200.0)
    parser.add_argument("--physical_z_um", type=float, default=1200.0)
    parser.add_argument(
        "--voxel_z_um",
        type=float,
        default=None,
        help="Explicit Z spacing in µm; overrides --physical_z_um / number of Z slices",
    )
    parser.add_argument("--spatial_bin_um", type=float, default=250.0)
    parser.add_argument("--scan_block", default="40,256,256")
    parser.add_argument("--histogram_bins", type=int, default=60)
    args = parser.parse_args()

    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if min(
        args.physical_x_um,
        args.physical_y_um,
        args.physical_z_um,
        args.spatial_bin_um,
    ) <= 0:
        raise ValueError("Physical dimensions and spatial bin size must be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_arr = open_array(args.raw, args.raw_key)
    label_arr = open_array(args.labels, args.label_key)
    shape = tuple(int(v) for v in raw_arr.shape)
    if shape != tuple(int(v) for v in label_arr.shape):
        raise ValueError(f"Raw/label volume mismatch: {shape} vs {label_arr.shape}")

    voxel_y_um = args.physical_y_um / shape[1]
    voxel_x_um = args.physical_x_um / shape[2]
    voxel_z_um = (
        float(args.voxel_z_um)
        if args.voxel_z_um is not None
        else args.physical_z_um / shape[0]
    )
    voxel_um_zyx = (voxel_z_um, voxel_y_um, voxel_x_um)
    physical_shape_um_zyx = tuple(shape[i] * voxel_um_zyx[i] for i in range(3))
    progress(f"Volume shape ZYX: {shape}")
    progress(f"Voxel size µm ZYX: {voxel_um_zyx}")
    progress(f"Physical extent µm ZYX: {physical_shape_um_zyx}")

    stats = scan_labels(
        label_arr,
        voxel_um_zyx,
        args.spatial_bin_um,
        parse_positive_zyx(args.scan_block),
    )
    instance_ids, volumes_um3, radii_um, r_soma, f_soma = derive_soma_fields(
        stats, voxel_um_zyx, args.spatial_bin_um
    )
    radius_summary = save_radius_outputs(
        args.outdir, instance_ids, volumes_um3, radii_um, args.histogram_bins
    )
    f_soma_summary = save_f_soma_outputs(
        args.outdir,
        f_soma,
        np.asarray(stats["soma_voxels_grid"]),
        np.asarray(stats["bin_voxel_capacity"]),
        float(np.prod(voxel_um_zyx)),
        args.histogram_bins,
    )
    np.savez_compressed(
        args.outdir / "soma_fields_250um.npz",
        r_soma_um=r_soma,
        f_soma=f_soma,
        soma_voxels=np.asarray(stats["soma_voxels_grid"]),
        bin_voxel_capacity=np.asarray(stats["bin_voxel_capacity"]),
    )

    x_index = shape[2] // 2 if args.x_index is None else args.x_index
    y_index = shape[1] // 2 if args.y_index is None else args.y_index
    if not 0 <= x_index < shape[2] or not 0 <= y_index < shape[1]:
        raise ValueError("Requested center-section index is outside the volume")

    color = parse_color(args.color)
    planes = ("xz", "yz") if args.plane == "both" else (args.plane,)
    rendered = []
    for plane in planes:
        rendered.append(
            render_plane(
                raw_arr,
                label_arr,
                plane,
                y_index if plane == "xz" else x_index,
                args.outdir,
                args.scale,
                args.low_percentile,
                args.high_percentile,
                color,
                args.overlay_alpha,
                args.field_overlay_alpha,
                r_soma,
                f_soma,
                stats["axis_bins"],
                float(r_soma.max()),
                float(f_soma.max()),
            )
        )

    metadata = {
        "raw": str(args.raw.resolve()),
        "labels": str(args.labels.resolve()),
        "volume_shape_zyx": list(shape),
        "voxel_size_um_zyx": list(voxel_um_zyx),
        "physical_extent_um_zyx": list(physical_shape_um_zyx),
        "z_spacing_source": "explicit voxel_z_um" if args.voxel_z_um is not None else "physical_z_um divided by number of Z slices",
        "physical_extent_input_um_zyx": [
            args.physical_z_um,
            args.physical_y_um,
            args.physical_x_um,
        ],
        "spatial_bin_um": args.spatial_bin_um,
        "spatial_grid_shape_zyx": list(stats["grid_shape"]),
        "r_soma_definition": "Unweighted mean equivalent-sphere radius of instance centroids in each physical cube",
        "f_soma_definition": "Foreground voxel volume divided by sampled cube volume",
        "equivalent_radius_formula": "r = (3 * segmented_volume / (4*pi))^(1/3)",
        "r_soma_display_max_um": float(r_soma.max()),
        "f_soma_display_max": float(f_soma.max()),
        "radius_summary": radius_summary,
        "f_soma_summary": f_soma_summary,
        "total_foreground_voxels": stats["total_foreground_voxels"],
        "scale": args.scale,
        "overlay_color_rgb": list(color),
        "mask_overlay_alpha": args.overlay_alpha,
        "field_overlay_alpha": args.field_overlay_alpha,
        "sections": rendered,
        "limitations": [
            "Cellpose labels are unique per inference block, but instances crossing block boundaries were not merged; split boundary objects can bias radii downward.",
            "The converter stored unit scale metadata, so physical dimensions come from command-line acquisition information.",
            "Black-to-white r_soma and f_soma images are independently normalized to their respective full-grid maxima.",
        ],
    }
    metadata_path = args.outdir / "visualization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    progress(f"Wrote visualization outputs to: {args.outdir}")
    progress(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
