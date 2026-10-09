#!/usr/bin/env python3
"""Extract segmentation summaries in locally reconstructed MRI voxel footprints."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


MRI_COLUMNS = ["mri_i", "mri_j", "mri_k"]
TOMOGRAMS = ("0022", "0029")
TOMO_SHAPE_IJK = np.array([2560, 2560, 2156], dtype=np.int64)
TOMO_SHAPE_ZYX = tuple(TOMO_SHAPE_IJK[::-1])
PREDICTORS = {
    "sandi": ["De", "Din", "Rsoma", "fextra", "fneurite", "fsoma"],
    "mpm": ["R1", "R2s", "A"],
    "r2s": ["R2s"],
    "dki": ["FA", "MD", "AD", "AK", "MK", "MSD", "MSK", "RD", "RK", "S0", "b0"],
}


def progress(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_csv_list(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def parse_zyx(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.replace("x", ",").split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("Expected three positive Z,Y,X integers")
    return parts


def open_zarr_array(path: Path, key: str):
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if key not in root:
        raise KeyError(f"Array key {key!r} is missing from {path}")
    array = root[key]
    if len(array.shape) != 3:
        raise ValueError(f"Expected a 3D ZYX array in {path}, got {array.shape}")
    return array


def ensure_capacity(arrays: tuple[np.ndarray, ...], required: int) -> tuple[np.ndarray, ...]:
    if required <= arrays[0].size:
        return arrays
    size = arrays[0].size
    while size < required:
        size *= 2
    grown = []
    for array in arrays:
        expanded = np.zeros(size, dtype=array.dtype)
        expanded[: array.size] = array
        grown.append(expanded)
    return tuple(grown)


def owner_from_label(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value)
    if "0022" in text:
        return "0022"
    if "0029" in text:
        return "0029"
    return None


def tomo_columns(tomo: str, suffix: str) -> list[str]:
    return [f"tomo{tomo}_{axis}_{suffix}" for axis in "ijk"]


def estimate_local_bases(
    frame: pd.DataFrame,
    tomo: str,
    max_neighbor_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local MRI-index-to-tomogram transforms for every table row."""
    indices = frame[MRI_COLUMNS].to_numpy(np.int64)
    centers = frame[tomo_columns(tomo, "650nm")].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    lookup = {tuple(index): row for row, index in enumerate(indices)}
    bases = np.full((len(frame), 3, 3), np.nan, dtype=np.float64)
    inverse = np.full_like(bases, np.nan)

    mapped = np.isfinite(centers).all(axis=1)
    for row, index in enumerate(indices):
        if not mapped[row]:
            continue
        vectors: list[np.ndarray] = []
        for axis in range(3):
            lower_row = upper_row = None
            lower_distance = upper_distance = None
            for distance in range(1, max_neighbor_distance + 1):
                if lower_row is None:
                    candidate = index.copy()
                    candidate[axis] -= distance
                    found = lookup.get(tuple(candidate))
                    if found is not None and mapped[found]:
                        lower_row, lower_distance = found, distance
                if upper_row is None:
                    candidate = index.copy()
                    candidate[axis] += distance
                    found = lookup.get(tuple(candidate))
                    if found is not None and mapped[found]:
                        upper_row, upper_distance = found, distance
                if lower_row is not None and upper_row is not None:
                    break

            if lower_row is not None and upper_row is not None:
                denominator = float(lower_distance + upper_distance)
                vector = (centers[upper_row] - centers[lower_row]) / denominator
            elif upper_row is not None:
                vector = (centers[upper_row] - centers[row]) / float(upper_distance)
            elif lower_row is not None:
                vector = (centers[row] - centers[lower_row]) / float(lower_distance)
            else:
                break
            vectors.append(vector)

        if len(vectors) != 3:
            continue
        basis = np.column_stack(vectors)
        determinant = float(np.linalg.det(basis))
        condition = float(np.linalg.cond(basis))
        if not np.isfinite(determinant) or abs(determinant) < 1e-6 or condition > 1e4:
            continue
        bases[row] = basis
        inverse[row] = np.linalg.inv(basis)
    return centers, bases, inverse


@dataclass
class ModalityGeometry:
    name: str
    table_path: Path
    frame: pd.DataFrame
    owner: np.ndarray
    centers: dict[str, np.ndarray]
    bases: dict[str, np.ndarray]
    inverse: dict[str, np.ndarray]
    valid_owner_basis: np.ndarray
    soma_voxels: np.ndarray
    vessel_voxels: np.ndarray
    cell_count: np.ndarray
    radius_sum: np.ndarray
    radius_median: list[list[float]]

    @classmethod
    def load(cls, name: str, path: Path, max_neighbor_distance: int):
        frame = pd.read_csv(path, dtype={"tomogram": "string"}, low_memory=False)
        required = MRI_COLUMNS + PREDICTORS[name] + ["coverage_percent", "tomogram"]
        for tomo in TOMOGRAMS:
            required.extend(tomo_columns(tomo, "650nm"))
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

        owner = np.array([owner_from_label(value) for value in frame["tomogram"]], dtype=object)
        centers: dict[str, np.ndarray] = {}
        bases: dict[str, np.ndarray] = {}
        inverse: dict[str, np.ndarray] = {}
        for tomo in TOMOGRAMS:
            centers[tomo], bases[tomo], inverse[tomo] = estimate_local_bases(
                frame, tomo, max_neighbor_distance
            )
        valid = np.zeros(len(frame), dtype=bool)
        for tomo in TOMOGRAMS:
            rows = owner == tomo
            valid[rows] = np.isfinite(inverse[tomo][rows]).all(axis=(1, 2))
        return cls(
            name=name,
            table_path=path,
            frame=frame,
            owner=owner,
            centers=centers,
            bases=bases,
            inverse=inverse,
            valid_owner_basis=valid,
            soma_voxels=np.zeros(len(frame), dtype=np.uint64),
            vessel_voxels=np.zeros(len(frame), dtype=np.uint64),
            cell_count=np.zeros(len(frame), dtype=np.uint32),
            # Columns store sum(R^1), sum(R^2), sum(R^3), sum(R^4),
            # sum(R^5), and sum(R^7), respectively.
            radius_sum=np.zeros((len(frame), 6), dtype=np.float64),
            radius_median=[[] for _ in range(len(frame))],
        )


class FootprintMapper:
    def __init__(self, geometry: ModalityGeometry, tomo: str, k_neighbors: int):
        selected = geometry.valid_owner_basis & (geometry.owner == tomo)
        self.rows = np.flatnonzero(selected)
        self.centers = geometry.centers[tomo][self.rows]
        self.inverse = geometry.inverse[tomo][self.rows]
        self.k = min(k_neighbors, len(self.rows))
        self.tree = cKDTree(self.centers) if len(self.rows) else None

    def map_points(self, points_ijk: np.ndarray) -> np.ndarray:
        result = np.full(len(points_ijk), -1, dtype=np.int64)
        if self.tree is None or not len(points_ijk):
            return result
        _, candidates = self.tree.query(points_ijk, k=self.k, workers=-1)
        if self.k == 1:
            candidates = candidates[:, None]
        unresolved = np.ones(len(points_ijk), dtype=bool)
        for rank in range(self.k):
            active = np.flatnonzero(unresolved)
            if not len(active):
                break
            local_candidate = candidates[active, rank]
            delta = points_ijk[active] - self.centers[local_candidate]
            local = np.einsum("nij,nj->ni", self.inverse[local_candidate], delta)
            inside = np.max(np.abs(local), axis=1) <= 0.500001
            accepted = active[inside]
            result[accepted] = self.rows[local_candidate[inside]]
            unresolved[accepted] = False
        return result


def chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                yield (
                    slice(z0, min(shape[0], z0 + chunks[0])),
                    slice(y0, min(shape[1], y0 + chunks[1])),
                    slice(x0, min(shape[2], x0 + chunks[2])),
                )


def map_foreground(
    points_ijk: np.ndarray,
    geometries: list[ModalityGeometry],
    mappers: dict[tuple[str, str], FootprintMapper],
    tomo: str,
    kind: str,
) -> None:
    for geometry in geometries:
        rows = mappers[(geometry.name, tomo)].map_points(points_ijk)
        rows = rows[rows >= 0]
        if not len(rows):
            continue
        counts = np.bincount(rows, minlength=len(geometry.frame)).astype(np.uint64)
        if kind == "soma":
            geometry.soma_voxels += counts
        else:
            geometry.vessel_voxels += counts


def scan_cell_labels(
    array,
    geometries: list[ModalityGeometry],
    mappers: dict[tuple[str, str], FootprintMapper],
    tomo: str,
    chunks: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(1024, dtype=np.uint64)
    sums_z = np.zeros(1024, dtype=np.float64)
    sums_y = np.zeros(1024, dtype=np.float64)
    sums_x = np.zeros(1024, dtype=np.float64)
    selections = list(chunk_slices(tuple(int(v) for v in array.shape), chunks))
    for number, selection in enumerate(selections, start=1):
        block = np.asarray(array[selection])
        foreground = block > 0
        if foreground.any():
            local_z, local_y, local_x = np.nonzero(foreground)
            z = local_z + selection[0].start
            y = local_y + selection[1].start
            x = local_x + selection[2].start
            labels = block[foreground].astype(np.int64, copy=False)
            unique, inverse_labels, local_counts = np.unique(
                labels, return_inverse=True, return_counts=True
            )
            counts, sums_z, sums_y, sums_x = ensure_capacity(
                (counts, sums_z, sums_y, sums_x), int(unique[-1]) + 1
            )
            counts[unique] += local_counts.astype(np.uint64)
            sums_z[unique] += np.bincount(inverse_labels, weights=z, minlength=len(unique))
            sums_y[unique] += np.bincount(inverse_labels, weights=y, minlength=len(unique))
            sums_x[unique] += np.bincount(inverse_labels, weights=x, minlength=len(unique))
            points_ijk = np.column_stack((x, y, z)).astype(np.float64, copy=False)
            map_foreground(points_ijk, geometries, mappers, tomo, "soma")
        if number == 1 or number % 100 == 0 or number == len(selections):
            progress(f"{tomo} cell chunks {number}/{len(selections)}")

    instance_ids = np.flatnonzero(counts)
    instance_counts = counts[instance_ids]
    centroids_zyx = np.column_stack(
        (sums_z[instance_ids], sums_y[instance_ids], sums_x[instance_ids])
    ) / instance_counts[:, None]
    return instance_counts, centroids_zyx


def scan_semantic_vessels(
    array,
    geometries: list[ModalityGeometry],
    mappers: dict[tuple[str, str], FootprintMapper],
    tomo: str,
    chunks: tuple[int, int, int],
) -> None:
    selections = list(chunk_slices(tuple(int(v) for v in array.shape), chunks))
    for number, selection in enumerate(selections, start=1):
        block = np.asarray(array[selection])
        local_z, local_y, local_x = np.nonzero(block > 0)
        if len(local_z):
            points_ijk = np.column_stack(
                (
                    local_x + selection[2].start,
                    local_y + selection[1].start,
                    local_z + selection[0].start,
                )
            ).astype(np.float64, copy=False)
            map_foreground(points_ijk, geometries, mappers, tomo, "vessel")
        if number == 1 or number % 100 == 0 or number == len(selections):
            progress(f"{tomo} vessel chunks {number}/{len(selections)}")


def assign_instances(
    instance_counts: np.ndarray,
    centroids_zyx: np.ndarray,
    geometries: list[ModalityGeometry],
    mappers: dict[tuple[str, str], FootprintMapper],
    tomo: str,
    voxel_size_um: float,
) -> None:
    radii = np.cbrt(
        3.0 * instance_counts.astype(np.float64) * voxel_size_um**3 / (4.0 * math.pi)
    )
    centroids_ijk = centroids_zyx[:, ::-1]
    for geometry in geometries:
        rows = mappers[(geometry.name, tomo)].map_points(centroids_ijk)
        valid = rows >= 0
        for row, radius in zip(rows[valid], radii[valid]):
            geometry.cell_count[row] += 1
            geometry.radius_sum[row, 0] += radius
            geometry.radius_sum[row, 1] += radius**2
            geometry.radius_sum[row, 2] += radius**3
            geometry.radius_sum[row, 3] += radius**4
            geometry.radius_sum[row, 4] += radius**5
            geometry.radius_sum[row, 5] += radius**7
            geometry.radius_median[row].append(float(radius))


def sample_center_foreground(array, geometry: ModalityGeometry, tomo: str) -> np.ndarray:
    output = np.full(len(geometry.frame), np.nan)
    selected = np.flatnonzero((geometry.owner == tomo) & np.isfinite(geometry.centers[tomo]).all(axis=1))
    if not len(selected):
        return output
    nearest = np.floor(geometry.centers[tomo][selected] + 0.5).astype(np.int64)
    in_bounds = np.all(nearest >= 0, axis=1) & np.all(nearest < TOMO_SHAPE_IJK, axis=1)
    selected = selected[in_bounds]
    nearest = nearest[in_bounds]
    chunks = tuple(int(v) for v in array.chunks)
    chunk_keys = nearest[:, ::-1] // np.asarray(chunks)
    order = np.lexsort((chunk_keys[:, 2], chunk_keys[:, 1], chunk_keys[:, 0]))
    selected = selected[order]
    nearest = nearest[order]
    chunk_keys = chunk_keys[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(chunk_keys, axis=0), axis=1)), len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        key = chunk_keys[start]
        starts = key * np.asarray(chunks)
        ends = np.minimum(starts + np.asarray(chunks), np.asarray(array.shape))
        block = np.asarray(array[tuple(slice(int(a), int(b)) for a, b in zip(starts, ends))])
        points_zyx = nearest[start:stop, ::-1] - starts
        values = block[points_zyx[:, 0], points_zyx[:, 1], points_zyx[:, 2]]
        output[selected[start:stop]] = values > 0
    return output


def reconstructed_bounds_coverage(
    geometry: ModalityGeometry,
    voxel_size_um: float,
    sample_step_um: float,
    max_axis_samples: int,
) -> np.ndarray:
    output = np.full(len(geometry.frame), np.nan)
    for row in np.flatnonzero(geometry.valid_owner_basis):
        owner = geometry.owner[row]
        basis = geometry.bases[owner][row]
        axis_lengths_um = np.linalg.norm(basis, axis=0) * voxel_size_um
        counts = np.clip(np.ceil(axis_lengths_um / sample_step_um).astype(int), 1, max_axis_samples)
        axes = [(np.arange(count) + 0.5) / count - 0.5 for count in counts]
        mesh = np.meshgrid(*axes, indexing="ij")
        local = np.column_stack([axis.ravel() for axis in mesh])
        covered = np.zeros(len(local), dtype=bool)
        for tomo in TOMOGRAMS:
            center = geometry.centers[tomo][row]
            tomo_basis = geometry.bases[tomo][row]
            if not np.isfinite(center).all() or not np.isfinite(tomo_basis).all():
                continue
            points = center + local @ tomo_basis.T
            covered |= np.all(points >= -0.5, axis=1) & np.all(
                points < TOMO_SHAPE_IJK - 0.5, axis=1
            )
        output[row] = 100.0 * float(np.mean(covered))
    return output


def finalize_geometry(
    geometry: ModalityGeometry,
    center_foreground: dict[str, np.ndarray],
    voxel_size_um: float,
    coverage_step_um: float,
    coverage_max_axis_samples: int,
    output_dir: Path,
) -> dict[str, object]:
    frame = geometry.frame.copy()
    n = len(frame)
    basis_determinant = np.full(n, np.nan)
    basis_condition = np.full(n, np.nan)
    edge_lengths_tomo_voxels = np.full((n, 3), np.nan)
    max_body_diagonal_tomo_voxels = np.full(n, np.nan)
    owner_basis = np.full((n, 3, 3), np.nan)
    diagonal_signs = np.array(
        [
            [sign_i, sign_j, sign_k]
            for sign_i in (-1.0, 1.0)
            for sign_j in (-1.0, 1.0)
            for sign_k in (-1.0, 1.0)
        ]
    )
    for tomo in TOMOGRAMS:
        rows = np.flatnonzero((geometry.owner == tomo) & geometry.valid_owner_basis)
        if len(rows):
            selected_bases = geometry.bases[tomo][rows]
            owner_basis[rows] = selected_bases
            basis_determinant[rows] = np.linalg.det(selected_bases)
            basis_condition[rows] = np.linalg.cond(selected_bases)
            edge_lengths_tomo_voxels[rows] = np.linalg.norm(selected_bases, axis=1)
            body_diagonals = np.einsum(
                "nij,sj->nsi", selected_bases, diagonal_signs
            )
            max_body_diagonal_tomo_voxels[rows] = np.max(
                np.linalg.norm(body_diagonals, axis=2), axis=1
            )

    footprint_voxels_equivalent = np.abs(basis_determinant)
    footprint_um3 = footprint_voxels_equivalent * voxel_size_um**3
    supplied_coverage = pd.to_numeric(frame["coverage_percent"], errors="coerce").to_numpy(float)
    coverage_fraction = supplied_coverage / 100.0
    covered_footprint_voxels_equivalent = footprint_voxels_equivalent * coverage_fraction
    covered_um3 = footprint_um3 * supplied_coverage / 100.0
    voxel_um3 = voxel_size_um**3
    reconstructed = reconstructed_bounds_coverage(
        geometry, voxel_size_um, coverage_step_um, coverage_max_axis_samples
    )

    count = geometry.cell_count.astype(float)
    radii_mean = np.divide(
        geometry.radius_sum[:, 0], count, out=np.full(n, np.nan), where=count > 0
    )
    radii_median = np.array(
        [np.median(values) if values else np.nan for values in geometry.radius_median]
    )
    r2 = geometry.radius_sum[:, 1]
    r3 = geometry.radius_sum[:, 2]
    volume_weighted = np.divide(
        geometry.radius_sum[:, 3], r3, out=np.full(n, np.nan), where=r3 > 0
    )
    effective_r7_r3 = np.power(
        np.divide(geometry.radius_sum[:, 5], r3, out=np.full(n, np.nan), where=r3 > 0),
        0.25,
    )
    effective_r5_r3 = np.sqrt(
        np.divide(geometry.radius_sum[:, 4], r3, out=np.full(n, np.nan), where=r3 > 0)
    )
    surface_to_volume_radius_inverse = np.divide(
        r2, r3, out=np.full(n, np.nan), where=r3 > 0
    )
    volume_to_surface_radius = np.divide(
        r3, r2, out=np.full(n, np.nan), where=r2 > 0
    )

    center = np.full(n, np.nan)
    for tomo in TOMOGRAMS:
        rows = geometry.owner == tomo
        center[rows] = center_foreground[tomo][rows]

    frame["analysis_tomogram"] = geometry.owner
    frame["local_basis_valid"] = geometry.valid_owner_basis.astype(np.uint8)
    frame["local_basis_determinant"] = basis_determinant
    frame["local_basis_condition"] = basis_condition
    frame["local_footprint_edge_mri_i_tomo_voxels"] = edge_lengths_tomo_voxels[:, 0]
    frame["local_footprint_edge_mri_j_tomo_voxels"] = edge_lengths_tomo_voxels[:, 1]
    frame["local_footprint_edge_mri_k_tomo_voxels"] = edge_lengths_tomo_voxels[:, 2]
    frame["local_footprint_max_body_diagonal_tomo_voxels"] = (
        max_body_diagonal_tomo_voxels
    )
    frame["local_footprint_max_body_diagonal_um"] = (
        max_body_diagonal_tomo_voxels * voxel_size_um
    )
    for tomo_axis, tomo_axis_name in enumerate("ijk"):
        for mri_axis, mri_axis_name in enumerate("ijk"):
            frame[
                f"local_basis_tomo_{tomo_axis_name}_per_mri_{mri_axis_name}"
            ] = owner_basis[:, tomo_axis, mri_axis]
    frame["local_footprint_voxels_equivalent"] = footprint_voxels_equivalent
    frame["coverage_fraction"] = coverage_fraction
    frame["covered_footprint_voxels_equivalent"] = (
        covered_footprint_voxels_equivalent
    )
    frame["local_footprint_volume_um3"] = footprint_um3
    frame["covered_footprint_volume_um3"] = covered_um3
    frame["reconstructed_bounds_coverage_percent"] = reconstructed
    frame["coverage_difference_percentage_points"] = reconstructed - supplied_coverage
    frame["center_cell_foreground"] = center
    frame["soma_foreground_voxels"] = geometry.soma_voxels
    frame["vessel_foreground_voxels"] = geometry.vessel_voxels
    frame["f_soma_seg"] = np.divide(
        geometry.soma_voxels * voxel_um3,
        covered_um3,
        out=np.full(n, np.nan),
        where=covered_um3 > 0,
    )
    frame["f_vessel_seg"] = np.divide(
        geometry.vessel_voxels * voxel_um3,
        covered_um3,
        out=np.full(n, np.nan),
        where=covered_um3 > 0,
    )
    frame["cell_count"] = geometry.cell_count
    frame["cell_radius_sum_r1_um"] = geometry.radius_sum[:, 0]
    frame["cell_radius_sum_r2_um2"] = geometry.radius_sum[:, 1]
    frame["cell_radius_sum_r3_um3"] = geometry.radius_sum[:, 2]
    frame["cell_radius_sum_r4_um4"] = geometry.radius_sum[:, 3]
    frame["cell_radius_sum_r5_um5"] = geometry.radius_sum[:, 4]
    frame["cell_radius_sum_r7_um7"] = geometry.radius_sum[:, 5]
    frame["cell_density_per_mm3"] = np.divide(
        count * 1e9, covered_um3, out=np.full(n, np.nan), where=covered_um3 > 0
    )
    frame["cell_radius_mean_um"] = radii_mean
    frame["cell_radius_median_um"] = radii_median
    frame["cell_radius_volume_weighted_um"] = volume_weighted
    frame["cell_radius_effective_r7_r3_um"] = effective_r7_r3
    frame["cell_radius_effective_r5_r3_um"] = effective_r5_r3
    frame["cell_surface_to_volume_radius_inverse_um_inv"] = (
        surface_to_volume_radius_inverse
    )
    frame["cell_volume_to_surface_radius_um"] = volume_to_surface_radius

    output_path = output_dir / f"hippo002_{geometry.name}_segmentation_features.csv.gz"
    frame.to_csv(output_path, index=False, compression="gzip")
    eligible = geometry.valid_owner_basis & np.isfinite(covered_um3) & (covered_um3 > 0)
    return {
        "modality": geometry.name,
        "input_table": str(geometry.table_path),
        "output_table": str(output_path),
        "rows": n,
        "rows_with_valid_local_basis": int(geometry.valid_owner_basis.sum()),
        "rows_with_positive_covered_volume": int(eligible.sum()),
        "rows_with_cells": int(np.sum(geometry.cell_count > 0)),
        "median_absolute_bounds_coverage_error_percentage_points": float(
            np.nanmedian(np.abs(reconstructed - supplied_coverage))
        ),
        "fraction_f_soma_outside_0_1": float(
            np.mean((frame.loc[eligible, "f_soma_seg"] < 0) | (frame.loc[eligible, "f_soma_seg"] > 1))
        ),
        "fraction_f_vessel_outside_0_1": float(
            np.mean((frame.loc[eligible, "f_vessel_seg"] < 0) | (frame.loc[eligible, "f_vessel_seg"] > 1))
        ),
    }


def table_for_modality(tables_dir: Path, name: str) -> Path:
    matches = sorted(tables_dir.glob(f"Hippo002_table_{name}*_in_tomos.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} table in {tables_dir}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--cell-0022", type=Path, required=True)
    parser.add_argument("--cell-0029", type=Path, required=True)
    parser.add_argument("--vessel-0022", type=Path, required=True)
    parser.add_argument("--vessel-0029", type=Path, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--modalities", default="sandi,mpm,r2s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-um", type=float, default=0.65)
    parser.add_argument("--scan-chunks", type=parse_zyx, default=(32, 256, 256))
    parser.add_argument("--max-neighbor-distance", type=int, default=3)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--coverage-sample-step-um", type=float, default=8.0)
    parser.add_argument("--coverage-max-axis-samples", type=int, default=16)
    args = parser.parse_args()

    modalities = parse_csv_list(args.modalities)
    unknown = sorted(set(modalities) - set(PREDICTORS))
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")
    if "dki" in modalities:
        raise ValueError("DKI is intentionally excluded; investigate its registration separately")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometries = [
        ModalityGeometry.load(
            name, table_for_modality(args.tables_dir, name), args.max_neighbor_distance
        )
        for name in modalities
    ]
    mappers = {
        (geometry.name, tomo): FootprintMapper(geometry, tomo, args.k_neighbors)
        for geometry in geometries
        for tomo in TOMOGRAMS
    }

    cell_paths = {"0022": args.cell_0022, "0029": args.cell_0029}
    vessel_paths = {"0022": args.vessel_0022, "0029": args.vessel_0029}
    center_foreground = {
        geometry.name: {tomo: np.full(len(geometry.frame), np.nan) for tomo in TOMOGRAMS}
        for geometry in geometries
    }

    for tomo in TOMOGRAMS:
        progress(f"Opening tomo{tomo} cell labels: {cell_paths[tomo]}")
        cell_array = open_zarr_array(cell_paths[tomo], args.array_key)
        vessel_array = open_zarr_array(vessel_paths[tomo], args.array_key)
        expected_shape = tuple(int(value) for value in cell_array.shape)
        if expected_shape != TOMO_SHAPE_ZYX:
            raise ValueError(
                f"tomo{tomo} labels have shape {expected_shape}; expected {TOMO_SHAPE_ZYX} from legend"
            )
        if tuple(int(value) for value in vessel_array.shape) != expected_shape:
            raise ValueError(f"tomo{tomo} cell/vessel shape mismatch")
        for geometry in geometries:
            center_foreground[geometry.name][tomo] = sample_center_foreground(
                cell_array, geometry, tomo
            )

        instance_counts, centroids_zyx = scan_cell_labels(
            cell_array, geometries, mappers, tomo, args.scan_chunks
        )
        assign_instances(
            instance_counts,
            centroids_zyx,
            geometries,
            mappers,
            tomo,
            args.voxel_size_um,
        )
        scan_semantic_vessels(
            vessel_array, geometries, mappers, tomo, args.scan_chunks
        )

    summaries = [
        finalize_geometry(
            geometry,
            center_foreground[geometry.name],
            args.voxel_size_um,
            args.coverage_sample_step_um,
            args.coverage_max_axis_samples,
            args.output_dir,
        )
        for geometry in geometries
    ]
    report = {
        "schema_version": 2,
        "method": "local_affine_MRI_footprints_from_neighboring_registered_centers",
        "modalities": modalities,
        "voxel_size_um": args.voxel_size_um,
        "tomogram_shape_zyx": [int(value) for value in TOMO_SHAPE_ZYX],
        "overlap_rule": "rows marked 0022+0029 are assigned to tomo0022, matching the supplied precedence note",
        "radius_definitions": {
            "mean": "mean(R)",
            "median": "median(R)",
            "volume_weighted": "sum(R^4)/sum(R^3)",
            "effective_r7_r3": "(sum(R^7)/sum(R^3))^(1/4)",
            "effective_r5_r3": "(sum(R^5)/sum(R^3))^(1/2)",
            "surface_to_volume_radius_inverse": "sum(R^2)/sum(R^3) [1/um]",
            "volume_to_surface_radius": "sum(R^3)/sum(R^2) [um]",
            "sphere_geometry_note": (
                "Exact area/volume is 3*sum(R^2)/sum(R^3), and exact volume/area "
                "is sum(R^3)/(3*sum(R^2)); constant factors do not affect correlation or R2."
            ),
        },
        "limitations": [
            "MRI footprints are local-affine reconstructions from neighboring centers, not exact transformed corners.",
            "Supplied coverage is used as the valid-volume denominator.",
            "Reconstructed coverage tests rectangular tomogram bounds and cannot reproduce the collaborator's circular stitch mask.",
            "Instance radii require block-boundary-reconciled cell labels.",
            "DKI is excluded because the collaborator flagged its registration as suspicious.",
        ],
        "inputs": {
            "cell": {key: str(value) for key, value in cell_paths.items()},
            "vessel": {key: str(value) for key, value in vessel_paths.items()},
        },
        "summaries": summaries,
    }
    report_path = args.output_dir / "extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    progress(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
