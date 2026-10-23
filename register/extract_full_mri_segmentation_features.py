#!/usr/bin/env python3
"""Extract cell and optional vessel features for every full-run MRI table."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from full_registration_config import (
    SPECIMENS,
    TableSpec,
    cell_mask_path,
    correction_factor,
    discover_tables,
    parse_tomogram_tokens,
    vessel_mask_path,
)


CORNER_NAMES = ("000", "001", "010", "011", "100", "101", "110", "111")
TOMO_SHAPE_IJK = np.array([2560, 2560, 2156], dtype=np.int64)
TOMO_SHAPE_ZYX = tuple(TOMO_SHAPE_IJK[::-1])
TASKS = [
    (specimen, tomo)
    for specimen, config in SPECIMENS.items()
    for tomo in config["tomograms"]
]


def progress(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_zyx(value: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in value.replace("x", ",").split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected positive Z,Y,X values")
    return values


def open_array(path: Path, key: str):
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if key not in root:
        raise KeyError(f"Missing array {key!r} in {path}")
    array = root[key]
    if tuple(int(value) for value in array.shape) != TOMO_SHAPE_ZYX:
        raise ValueError(f"Unexpected shape {array.shape} in {path}; expected {TOMO_SHAPE_ZYX}")
    return array


def geometry_columns(specimen: str) -> list[str]:
    columns = [
        "mri_i",
        "mri_j",
        "mri_k",
        "coverage_percent",
        "coord_usable",
        "corners_usable",
        "tomogram",
        "tomogram_usable",
        "xpct_mean",
        "xpct_samples",
    ]
    for tomo in SPECIMENS[specimen]["tomograms"]:
        columns.extend(f"tomo{tomo}_{axis}_650nm" for axis in "ijk")
        columns.extend(
            f"c{corner}_tomo{tomo}_{axis}_650nm"
            for corner in CORNER_NAMES
            for axis in "ijk"
        )
    return columns


def direct_geometry(frame: pd.DataFrame, tomo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = frame[[f"tomo{tomo}_{axis}_650nm" for axis in "ijk"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    corners = np.stack(
        [
            frame[[f"c{corner}_tomo{tomo}_{axis}_650nm" for axis in "ijk"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
            for corner in CORNER_NAMES
        ],
        axis=1,
    )
    bits = np.array([[int(bit) for bit in corner] for corner in CORNER_NAMES])
    basis = np.full((len(frame), 3, 3), np.nan)
    for axis in range(3):
        high = np.mean(corners[:, bits[:, axis] == 1, :], axis=1)
        low = np.mean(corners[:, bits[:, axis] == 0, :], axis=1)
        basis[:, :, axis] = high - low
    finite = np.isfinite(centers).all(axis=1) & np.isfinite(basis).all(axis=(1, 2))
    inverse = np.full_like(basis, np.nan)
    if finite.any():
        determinants = np.linalg.det(basis[finite])
        conditions = np.linalg.cond(basis[finite])
        good_local = np.isfinite(determinants) & (np.abs(determinants) > 1e-6)
        good_local &= np.isfinite(conditions) & (conditions < 1e4)
        finite_rows = np.flatnonzero(finite)
        good_rows = finite_rows[good_local]
        inverse[good_rows] = np.linalg.inv(basis[good_rows])
    return centers, basis, inverse


def select_owners(frame: pd.DataFrame, spec: TableSpec, inverse: dict[str, np.ndarray]) -> np.ndarray:
    allowed = SPECIMENS[spec.specimen]["tomograms"]
    owners = np.full(len(frame), None, dtype=object)
    for row_index, (usable_value, touched_value) in enumerate(
        zip(frame["tomogram_usable"], frame["tomogram"])
    ):
        candidates = parse_tomogram_tokens(usable_value, allowed)
        candidates.extend(
            tomo
            for tomo in parse_tomogram_tokens(touched_value, allowed)
            if tomo not in candidates
        )
        for tomo in candidates:
            if np.isfinite(inverse[tomo][row_index]).all():
                owners[row_index] = tomo
                break
    return owners


@dataclass
class Geometry:
    spec: TableSpec
    frame: pd.DataFrame
    owner: np.ndarray
    centers: dict[str, np.ndarray]
    bases: dict[str, np.ndarray]
    inverse: dict[str, np.ndarray]
    valid: np.ndarray
    soma_voxels: np.ndarray
    vessel_voxels: np.ndarray
    cell_count: np.ndarray
    radius_sums: np.ndarray
    per_cell_parts: list[pd.DataFrame] = field(default_factory=list)

    @classmethod
    def load(cls, spec: TableSpec):
        header = list(pd.read_csv(spec.path, nrows=0).columns)
        usecols = list(dict.fromkeys([*spec.predictors, *geometry_columns(spec.specimen)]))
        missing = sorted(set(usecols) - set(header))
        if missing:
            raise ValueError(f"{spec.path}: missing columns {missing}")
        frame = pd.read_csv(spec.path, usecols=usecols, low_memory=False)
        frame.insert(0, "mri_table_row", np.arange(len(frame), dtype=np.int64))
        frame.insert(0, "dataset_id", spec.dataset_id)
        frame.insert(1, "acquisition", spec.acquisition)
        frame.insert(2, "specimen", spec.specimen)
        frame.insert(3, "protocol", spec.protocol)
        for predictor in spec.predictors:
            factor = correction_factor(spec.acquisition, predictor)
            frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce")
            if factor != 1.0:
                frame[f"{predictor}_uncorrected"] = frame[predictor]
                frame[predictor] *= factor
        centers, bases, inverse = {}, {}, {}
        for tomo in SPECIMENS[spec.specimen]["tomograms"]:
            centers[tomo], bases[tomo], inverse[tomo] = direct_geometry(frame, tomo)
        owner = select_owners(frame, spec, inverse)
        valid = np.array(
            [
                tomo is not None and np.isfinite(inverse[str(tomo)][row]).all()
                for row, tomo in enumerate(owner)
            ],
            dtype=bool,
        )
        n = len(frame)
        return cls(
            spec=spec,
            frame=frame,
            owner=owner,
            centers=centers,
            bases=bases,
            inverse=inverse,
            valid=valid,
            soma_voxels=np.zeros(n, dtype=np.uint64),
            vessel_voxels=np.zeros(n, dtype=np.uint64),
            cell_count=np.zeros(n, dtype=np.uint32),
            radius_sums=np.zeros((n, 6), dtype=np.float64),
        )


class Mapper:
    def __init__(self, geometry: Geometry, tomo: str, neighbors: int):
        selected = geometry.valid & (geometry.owner == tomo)
        self.rows = np.flatnonzero(selected)
        self.centers = geometry.centers[tomo][self.rows]
        self.inverse = geometry.inverse[tomo][self.rows]
        self.k = min(neighbors, len(self.rows))
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


def chunks(shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]):
    for z in range(0, shape[0], chunk_shape[0]):
        for y in range(0, shape[1], chunk_shape[1]):
            for x in range(0, shape[2], chunk_shape[2]):
                yield (
                    slice(z, min(z + chunk_shape[0], shape[0])),
                    slice(y, min(y + chunk_shape[1], shape[1])),
                    slice(x, min(x + chunk_shape[2], shape[2])),
                )


def ensure_capacity(arrays: tuple[np.ndarray, ...], required: int) -> tuple[np.ndarray, ...]:
    if required <= len(arrays[0]):
        return arrays
    size = len(arrays[0])
    while size < required:
        size *= 2
    output = []
    for array in arrays:
        grown = np.zeros(size, dtype=array.dtype)
        grown[: len(array)] = array
        output.append(grown)
    return tuple(output)


def map_foreground(points: np.ndarray, geometries: list[Geometry], mappers, tomo: str, vessel: bool):
    for geometry in geometries:
        rows = mappers[(geometry.spec.dataset_id, tomo)].map_points(points)
        rows = rows[rows >= 0]
        if not len(rows):
            continue
        counts = np.bincount(rows, minlength=len(geometry.frame)).astype(np.uint64)
        if vessel:
            geometry.vessel_voxels += counts
        else:
            geometry.soma_voxels += counts


def scan_cells(array, geometries, mappers, tomo, chunk_shape):
    counts = np.zeros(1024, dtype=np.uint64)
    sums = [np.zeros(1024, dtype=np.float64) for _ in range(3)]
    selections = list(chunks(tuple(array.shape), chunk_shape))
    for number, selection in enumerate(selections, 1):
        block = np.asarray(array[selection])
        foreground = block > 0
        if foreground.any():
            local = np.column_stack(np.nonzero(foreground))
            global_zyx = local + np.array([part.start for part in selection])
            labels = block[foreground].astype(np.int64, copy=False)
            unique, inverse_labels, local_counts = np.unique(
                labels, return_inverse=True, return_counts=True
            )
            counts, *sums = ensure_capacity((counts, *sums), int(unique[-1]) + 1)
            counts[unique] += local_counts.astype(np.uint64)
            for axis in range(3):
                sums[axis][unique] += np.bincount(
                    inverse_labels, weights=global_zyx[:, axis], minlength=len(unique)
                )
            map_foreground(global_zyx[:, ::-1], geometries, mappers, tomo, vessel=False)
        if number == 1 or number % 100 == 0 or number == len(selections):
            progress(f"{tomo} cell chunks {number}/{len(selections)}")
    instance_ids = np.flatnonzero(counts)
    volumes = counts[instance_ids]
    centroids = np.column_stack([values[instance_ids] for values in sums]) / volumes[:, None]
    return instance_ids, volumes, centroids


def assign_instances(instance_ids, volumes, centroids_zyx, geometries, mappers, specimen, tomo, voxel_um):
    radii = np.cbrt(3.0 * volumes.astype(float) * voxel_um**3 / (4.0 * math.pi))
    for geometry in geometries:
        rows = mappers[(geometry.spec.dataset_id, tomo)].map_points(centroids_zyx[:, ::-1])
        keep = rows >= 0
        assigned_rows = rows[keep]
        assigned_radii = radii[keep]
        np.add.at(geometry.cell_count, assigned_rows, 1)
        for power_index, power in enumerate((1, 2, 3, 4, 5, 7)):
            np.add.at(geometry.radius_sums[:, power_index], assigned_rows, assigned_radii**power)
        if keep.any():
            geometry.per_cell_parts.append(
                pd.DataFrame(
                    {
                        "dataset_id": geometry.spec.dataset_id,
                        "acquisition": geometry.spec.acquisition,
                        "specimen": specimen,
                        "protocol": geometry.spec.protocol,
                        "mri_table_row": assigned_rows,
                        "mri_i": geometry.frame["mri_i"].to_numpy()[assigned_rows],
                        "mri_j": geometry.frame["mri_j"].to_numpy()[assigned_rows],
                        "mri_k": geometry.frame["mri_k"].to_numpy()[assigned_rows],
                        "analysis_tomogram": tomo,
                        "cell_instance_id": instance_ids[keep],
                        "cell_volume_voxels": volumes[keep],
                        "cell_centroid_z": centroids_zyx[keep, 0],
                        "cell_centroid_y": centroids_zyx[keep, 1],
                        "cell_centroid_x": centroids_zyx[keep, 2],
                        "cell_radius_um": assigned_radii,
                    }
                )
            )


def scan_vessels(array, geometries, mappers, tomo, chunk_shape):
    selections = list(chunks(tuple(array.shape), chunk_shape))
    for number, selection in enumerate(selections, 1):
        block = np.asarray(array[selection]) > 0
        if block.any():
            local = np.column_stack(np.nonzero(block))
            global_zyx = local + np.array([part.start for part in selection])
            map_foreground(global_zyx[:, ::-1], geometries, mappers, tomo, vessel=True)
        if number == 1 or number % 100 == 0 or number == len(selections):
            progress(f"{tomo} vessel chunks {number}/{len(selections)}")


def finalize(geometry: Geometry, vessel_available: dict[str, bool], voxel_um: float, output_dir: Path):
    frame = geometry.frame.copy()
    n = len(frame)
    basis = np.full((n, 3, 3), np.nan)
    for tomo in SPECIMENS[geometry.spec.specimen]["tomograms"]:
        selected = np.flatnonzero((geometry.owner == tomo) & geometry.valid)
        basis[selected] = geometry.bases[tomo][selected]
    determinant = np.linalg.det(basis)
    condition = np.full(n, np.nan)
    condition[geometry.valid] = np.linalg.cond(basis[geometry.valid])
    footprint_voxels = np.abs(determinant)
    footprint_um3 = footprint_voxels * voxel_um**3
    coverage = pd.to_numeric(frame["coverage_percent"], errors="coerce").to_numpy(float)
    covered_um3 = footprint_um3 * coverage / 100.0
    count = geometry.cell_count.astype(float)
    r1, r2, r3, r4, r5, r7 = geometry.radius_sums.T
    r23 = np.divide(r2, r3, out=np.full(n, np.nan), where=r3 > 0)
    vessel_rows_available = np.array(
        [bool(vessel_available.get(str(owner), False)) for owner in geometry.owner], dtype=bool
    )

    frame["analysis_tomogram"] = geometry.owner
    frame["local_basis_valid"] = geometry.valid.astype(np.uint8)
    frame["local_basis_determinant"] = determinant
    frame["local_basis_condition"] = condition
    frame["local_footprint_voxels_equivalent"] = footprint_voxels
    frame["local_footprint_volume_um3"] = footprint_um3
    frame["coverage_fraction"] = coverage / 100.0
    frame["covered_footprint_volume_um3"] = covered_um3
    frame["soma_foreground_voxels"] = geometry.soma_voxels
    frame["vessel_mask_available"] = vessel_rows_available.astype(np.uint8)
    frame["vessel_foreground_voxels"] = np.where(
        vessel_rows_available, geometry.vessel_voxels.astype(float), np.nan
    )
    frame["f_soma_seg"] = np.divide(
        geometry.soma_voxels * voxel_um**3,
        covered_um3,
        out=np.full(n, np.nan),
        where=covered_um3 > 0,
    )
    frame["f_vessel_seg"] = np.divide(
        geometry.vessel_voxels * voxel_um**3,
        covered_um3,
        out=np.full(n, np.nan),
        where=(covered_um3 > 0) & vessel_rows_available,
    )
    frame["cell_count"] = geometry.cell_count
    for name, values in zip(("r1_um", "r2_um2", "r3_um3", "r4_um4", "r5_um5", "r7_um7"), geometry.radius_sums.T):
        frame[f"cell_radius_sum_{name}"] = values
    frame["cell_density_per_mm3"] = np.divide(
        count * 1e9, covered_um3, out=np.full(n, np.nan), where=covered_um3 > 0
    )
    frame["cell_radius_mean_um"] = np.divide(r1, count, out=np.full(n, np.nan), where=count > 0)
    frame["cell_radius_volume_weighted_um"] = np.divide(
        r4, r3, out=np.full(n, np.nan), where=r3 > 0
    )
    frame["cell_radius_effective_r5_r3_um"] = np.sqrt(
        np.divide(r5, r3, out=np.full(n, np.nan), where=r3 > 0)
    )
    frame["cell_radius_effective_r7_r3_um"] = np.power(
        np.divide(r7, r3, out=np.full(n, np.nan), where=r3 > 0), 0.25
    )
    frame["cell_surface_to_volume_r23_um_inv"] = r23
    frame["cell_surface_density_um_inv"] = np.divide(
        4.0 * math.pi * r2, covered_um3, out=np.full(n, np.nan), where=covered_um3 > 0
    )

    per_cell = pd.concat(geometry.per_cell_parts, ignore_index=True) if geometry.per_cell_parts else pd.DataFrame()
    median_lookup = per_cell.groupby("mri_table_row")["cell_radius_um"].median() if len(per_cell) else pd.Series(dtype=float)
    frame["cell_radius_median_um"] = frame["mri_table_row"].map(median_lookup)

    table_dir = output_dir / "tables"
    radius_dir = output_dir / "per_cell_radius_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    radius_dir.mkdir(parents=True, exist_ok=True)
    feature_path = table_dir / f"{geometry.spec.dataset_id}_segmentation_features.csv.gz"
    radius_path = radius_dir / f"{geometry.spec.dataset_id}_cell_radii_by_mri_voxel.csv.gz"
    frame.to_csv(feature_path, index=False, compression="gzip")
    per_cell.to_csv(radius_path, index=False, compression="gzip")
    return {
        "dataset_id": geometry.spec.dataset_id,
        "acquisition": geometry.spec.acquisition,
        "specimen": geometry.spec.specimen,
        "protocol": geometry.spec.protocol,
        "predictors": ";".join(geometry.spec.predictors),
        "input_rows": len(frame),
        "valid_geometry_rows": int(geometry.valid.sum()),
        "positive_basis_determinants": int((determinant > 0).sum()),
        "negative_basis_determinants": int((determinant < 0).sum()),
        "median_footprint_volume_um3": float(np.nanmedian(footprint_um3)),
        "coord_usable_rows": int(pd.to_numeric(frame["coord_usable"], errors="coerce").eq(1).sum()),
        "coverage80_usable_rows": int(
            (
                pd.to_numeric(frame["coord_usable"], errors="coerce").eq(1)
                & (pd.to_numeric(frame["coverage_percent"], errors="coerce") >= 80)
            ).sum()
        ),
        "assigned_cells": len(per_cell),
        "vessel_rows_available": int(vessel_rows_available.sum()),
        "feature_table": str(feature_path),
        "per_cell_radius_table": str(radius_path),
    }


def partial_directory(output_dir: Path, specimen: str, tomo: str) -> Path:
    return output_dir / "partials" / f"{specimen}_{tomo}"


def save_partial(geometry: Geometry, specimen: str, tomo: str, vessel_available: bool, output_dir: Path):
    directory = partial_directory(output_dir, specimen, tomo)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{geometry.spec.dataset_id}.npz",
        soma_voxels=geometry.soma_voxels,
        vessel_voxels=geometry.vessel_voxels,
        cell_count=geometry.cell_count,
        radius_sums=geometry.radius_sums,
        vessel_available=np.asarray([int(vessel_available)], dtype=np.uint8),
    )
    per_cell = pd.concat(geometry.per_cell_parts, ignore_index=True) if geometry.per_cell_parts else pd.DataFrame()
    per_cell.to_csv(
        directory / f"{geometry.spec.dataset_id}_cell_radii.csv.gz",
        index=False,
        compression="gzip",
    )


def load_partials(geometry: Geometry, output_dir: Path) -> dict[str, bool]:
    availability = {}
    for tomo in SPECIMENS[geometry.spec.specimen]["tomograms"]:
        directory = partial_directory(output_dir, geometry.spec.specimen, tomo)
        path = directory / f"{geometry.spec.dataset_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing extraction partial: {path}")
        with np.load(path) as partial:
            geometry.soma_voxels += partial["soma_voxels"]
            geometry.vessel_voxels += partial["vessel_voxels"]
            geometry.cell_count += partial["cell_count"]
            geometry.radius_sums += partial["radius_sums"]
            availability[tomo] = bool(partial["vessel_available"][0])
        per_cell_path = directory / f"{geometry.spec.dataset_id}_cell_radii.csv.gz"
        if per_cell_path.stat().st_size > 30:
            part = pd.read_csv(per_cell_path)
            if len(part):
                geometry.per_cell_parts.append(part)
    return availability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--voxel-size-um", type=float, default=0.65)
    parser.add_argument("--scan-chunks", type=parse_zyx, default=(32, 256, 256))
    parser.add_argument("--mapper-neighbors", type=int, default=16)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_tables(args.input_root)
    if args.task_index is not None and args.finalize_only:
        raise ValueError("--task-index and --finalize-only are mutually exclusive")

    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(TASKS):
            raise ValueError(f"--task-index must be between 0 and {len(TASKS) - 1}")
        specimen, tomo = TASKS[args.task_index]
        selected = [Geometry.load(spec) for spec in specs if spec.specimen == specimen]
        config = SPECIMENS[specimen]
        mappers = {
            (geometry.spec.dataset_id, tomo): Mapper(geometry, tomo, args.mapper_neighbors)
            for geometry in selected
        }
        cell_path = cell_mask_path(args.data_root, specimen, tomo, merged=True)
        if not cell_path.is_dir():
            raise FileNotFoundError(f"Missing merged cell mask: {cell_path}")
        vessel_path = vessel_mask_path(args.data_root, specimen, tomo)
        vessel_available = vessel_path.is_dir()
        progress(f"Opening {specimen} tomo {tomo} cells")
        instance_ids, volumes, centroids = scan_cells(
            open_array(cell_path, args.array_key), selected, mappers, tomo, args.scan_chunks
        )
        assign_instances(
            instance_ids, volumes, centroids, selected, mappers, specimen, tomo, args.voxel_size_um
        )
        if vessel_available:
            progress(f"Opening {specimen} tomo {tomo} vessels")
            scan_vessels(
                open_array(vessel_path, args.array_key), selected, mappers, tomo, args.scan_chunks
            )
        else:
            progress(f"No vessel mask for {specimen} tomo {tomo}; vessel features will be NA")
        for geometry in selected:
            save_partial(geometry, specimen, tomo, vessel_available, args.output_dir)
        (partial_directory(args.output_dir, specimen, tomo) / "COMPLETED").touch()
        progress(f"Completed extraction partial {args.task_index}: {specimen} tomo {tomo}")
        return

    if not args.finalize_only:
        raise ValueError("Distributed full extraction requires --task-index or --finalize-only")

    geometries = [Geometry.load(spec) for spec in specs]
    all_summaries = []
    mask_inventory = []
    for geometry in geometries:
        vessel_available = load_partials(geometry, args.output_dir)
        all_summaries.append(
            finalize(geometry, vessel_available, args.voxel_size_um, args.output_dir)
        )
    for specimen, config in SPECIMENS.items():
        for tomo in config["tomograms"]:
            cell_path = cell_mask_path(args.data_root, specimen, tomo, merged=True)
            vessel_path = vessel_mask_path(args.data_root, specimen, tomo)
            partial = partial_directory(args.output_dir, specimen, tomo) / "COMPLETED"
            mask_inventory.append(
                {
                    "specimen": specimen,
                    "sample_id": config["sample_id"],
                    "tomogram": tomo,
                    "cell_mask": str(cell_path),
                    "vessel_mask": str(vessel_path),
                    "vessel_available": int(vessel_path.is_dir()),
                    "partial_completed": int(partial.is_file()),
                }
            )

    pd.DataFrame(all_summaries).to_csv(args.output_dir / "feature_manifest.csv", index=False)
    pd.DataFrame(mask_inventory).to_csv(args.output_dir / "mask_inventory.csv", index=False)
    report = {
        "schema_version": 1,
        "table_count": len(specs),
        "specimen_count": len(SPECIMENS),
        "tomogram_count": sum(len(config["tomograms"]) for config in SPECIMENS.values()),
        "vessel_tomograms_available": sum(row["vessel_available"] for row in mask_inventory),
        "voxel_size_um": args.voxel_size_um,
        "r23_definition": "sum(r^2)/sum(r^3) [1/um]",
        "r32_not_computed": True,
        "hh_exvivo_scaling": "No additional factor: full-run input tables already contain corrected values",
    }
    (args.output_dir / "extraction_report.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    progress(f"Completed full extraction: {args.output_dir}")


if __name__ == "__main__":
    main()
