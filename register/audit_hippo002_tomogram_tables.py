#!/usr/bin/env python3
"""Audit Hippo002 MRI tables mapped into the two original tomograms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MRI_COLUMNS = ["mri_i", "mri_j", "mri_k"]
STITCH_COLUMNS = ["seg_i_650nm", "seg_j_650nm", "seg_k_650nm"]
STITCH_NEAREST = [f"{column}_nearest" for column in STITCH_COLUMNS]
TOMOGRAMS = ("0022", "0029")
TOMO_SHAPE_IJK = np.array([2560, 2560, 2156], dtype=np.int64)


def tomo_columns(tomo: str, suffix: str) -> list[str]:
    return [f"tomo{tomo}_{axis}_{suffix}" for axis in "ijk"]


def compare_original(
    updated: pd.DataFrame, original_path: Path
) -> dict[str, object]:
    if not original_path.exists():
        return {"available": False, "path": str(original_path)}

    original = pd.read_csv(original_path)
    result: dict[str, object] = {
        "available": True,
        "path": str(original_path),
        "same_rows": bool(len(updated) == len(original)),
        "missing_original_columns": [
            column for column in original.columns if column not in updated.columns
        ],
    }
    if not result["same_rows"] or result["missing_original_columns"]:
        result["identical"] = False
        return result

    column_results: dict[str, object] = {}
    all_identical = True
    for column in original.columns:
        left = pd.to_numeric(original[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(updated[column], errors="coerce").to_numpy(float)
        matching_nan = np.isnan(left) & np.isnan(right)
        finite = np.isfinite(left) & np.isfinite(right)
        equal = matching_nan | (finite & np.isclose(left, right, rtol=0.0, atol=0.0))
        mismatches = int((~equal).sum())
        all_identical &= mismatches == 0
        column_results[column] = {"mismatches": mismatches}
    result["columns"] = column_results
    result["identical"] = bool(all_identical)
    return result


def audit_tomogram(frame: pd.DataFrame, tomo: str) -> dict[str, object]:
    continuous_columns = tomo_columns(tomo, "650nm")
    nearest_columns = tomo_columns(tomo, "nearest")
    in_volume_column = f"tomo{tomo}_in_volume"
    in_circle_column = f"tomo{tomo}_in_circle"
    required = continuous_columns + nearest_columns + [in_volume_column, in_circle_column]
    missing = [column for column in required if column not in frame]
    if missing:
        return {"missing_columns": missing}

    continuous = frame[continuous_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nearest = frame[nearest_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mapped = np.isfinite(continuous).all(axis=1)
    partial_mapping = np.isfinite(continuous).any(axis=1) & ~mapped
    expected_nearest = np.floor(continuous + 0.5)
    rounding_mismatch = mapped & np.any(nearest != expected_nearest, axis=1)
    nearest_finite = np.isfinite(nearest).all(axis=1)
    expected_in_volume = (
        mapped
        & nearest_finite
        & np.all(nearest >= 0, axis=1)
        & np.all(nearest < TOMO_SHAPE_IJK, axis=1)
    )
    in_volume = pd.to_numeric(frame[in_volume_column], errors="coerce").to_numpy(float)
    in_circle = pd.to_numeric(frame[in_circle_column], errors="coerce").to_numpy(float)
    comparable_volume = np.isfinite(in_volume)
    volume_mismatch = comparable_volume & (in_volume.astype(bool) != expected_in_volume)

    mri_indices = frame[MRI_COLUMNS].to_numpy(np.int64)
    lookup = {tuple(index): row for row, index in enumerate(mri_indices)}
    central_difference_norms_um: list[list[float]] = [[], [], []]
    complete_neighborhoods = 0
    for row, index in enumerate(mri_indices):
        if not mapped[row]:
            continue
        has_all_neighbors = True
        row_vectors: list[np.ndarray] = []
        for axis in range(3):
            lower = index.copy()
            upper = index.copy()
            lower[axis] -= 1
            upper[axis] += 1
            lower_row = lookup.get(tuple(lower))
            upper_row = lookup.get(tuple(upper))
            if (
                lower_row is None
                or upper_row is None
                or not mapped[lower_row]
                or not mapped[upper_row]
            ):
                has_all_neighbors = False
                break
            row_vectors.append((continuous[upper_row] - continuous[lower_row]) / 2.0)
        if has_all_neighbors:
            complete_neighborhoods += 1
            for axis, vector in enumerate(row_vectors):
                central_difference_norms_um[axis].append(float(np.linalg.norm(vector) * 0.65))

    return {
        "mapped_centers": int(mapped.sum()),
        "partial_coordinate_rows": int(partial_mapping.sum()),
        "nearest_rounding_mismatches": int(rounding_mismatch.sum()),
        "reported_in_volume": int(np.nansum(in_volume == 1)),
        "calculated_in_volume": int(expected_in_volume.sum()),
        "in_volume_flag_mismatches": int(volume_mismatch.sum()),
        "reported_in_circle": int(np.nansum(in_circle == 1)),
        "complete_plus_minus_one_mri_neighborhoods": complete_neighborhoods,
        "complete_neighborhood_fraction_of_mapped": float(
            complete_neighborhoods / max(int(mapped.sum()), 1)
        ),
        "median_mri_axis_step_um_from_center_lattice": [
            float(np.median(values)) if values else None
            for values in central_difference_norms_um
        ],
        "invalid_in_volume_flags": int(np.sum(np.isfinite(in_volume) & ~np.isin(in_volume, [0, 1]))),
        "invalid_in_circle_flags": int(np.sum(np.isfinite(in_circle) & ~np.isin(in_circle, [0, 1]))),
        "coordinate_ranges_ijk": {
            "min": np.nanmin(continuous, axis=0).tolist(),
            "max": np.nanmax(continuous, axis=0).tolist(),
        },
    }


def audit_table(path: Path, original_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(path, dtype={"tomogram": "string"}, low_memory=False)
    stem = path.stem.replace("_in_tomos", "")
    original_path = original_dir / f"{stem}.csv"
    labels = frame["tomogram"].fillna("<blank>").astype(str).value_counts(dropna=False)
    both_mapped = np.ones(len(frame), dtype=bool)
    both_in_volume = np.ones(len(frame), dtype=bool)
    for tomo in TOMOGRAMS:
        coordinates = frame[tomo_columns(tomo, "650nm")].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        both_mapped &= np.isfinite(coordinates).all(axis=1)
        both_in_volume &= (
            pd.to_numeric(frame[f"tomo{tomo}_in_volume"], errors="coerce")
            .fillna(0)
            .to_numpy(int)
            == 1
        )

    return {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "tomogram_labels": {str(key): int(value) for key, value in labels.items()},
        "centers_mapped_to_both_tomograms": int(both_mapped.sum()),
        "centers_inside_both_tomograms": int(both_in_volume.sum()),
        "tomo0022": audit_tomogram(frame, "0022"),
        "tomo0029": audit_tomogram(frame, "0029"),
        "original_table_comparison": compare_original(frame, original_path),
    }


def write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Hippo002 in-tomogram table audit",
        "",
        "Expected individual-tomogram shape: `(i,j,k) = (2560,2560,2156)`; "
        "OME-Zarr array order: `(z,y,x) = (k,j,i)`.",
        "",
        "| Table | Rows | Original unchanged | Both mapped | Both in volume | 0022 volume | 0029 volume | Complete neighbour lattice |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for table in report["tables"]:
        lines.append(
            "| {file} | {rows} | {same} | {mapped} | {inside} | {v22} | {v29} | {neighbors} |".format(
                file=table["file"],
                rows=table["rows"],
                same=table["original_table_comparison"].get("identical", False),
                mapped=table["centers_mapped_to_both_tomograms"],
                inside=table["centers_inside_both_tomograms"],
                v22=table["tomo0022"].get("reported_in_volume", "n/a"),
                v29=table["tomo0029"].get("reported_in_volume", "n/a"),
                neighbors=min(
                    table["tomo0022"].get("complete_plus_minus_one_mri_neighborhoods", 0),
                    table["tomo0029"].get("complete_plus_minus_one_mri_neighborhoods", 0),
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These tables bridge each MRI-voxel centre to the two original tomogram arrays. ",
            "They permit direct centre spot-checks against the OME-Zarr masks after reversing "
            "CSV `(i,j,k)` to array `(k,j,i)`. They still do not encode the full transformed "
            "footprint of an MRI voxel. Exact density and volume-fraction aggregation requires "
            "voxel corners, dense sample coordinates, or the complete spatial transform.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).parent
    parser.add_argument("--tables-dir", type=Path, default=root / "Hippo002_tables_in_tomos")
    parser.add_argument(
        "--original-tables-dir", type=Path, default=root / "Hippo002_table_correlation"
    )
    parser.add_argument("--outdir", type=Path, default=root / "audit_output")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "tomogram_shape_ijk": TOMO_SHAPE_IJK.tolist(),
        "zarr_shape_zyx": TOMO_SHAPE_IJK[::-1].tolist(),
        "tables": [
            audit_table(path, args.original_tables_dir)
            for path in sorted(args.tables_dir.glob("*_in_tomos.csv"))
        ],
    }
    json_path = args.outdir / "hippo002_tomogram_table_audit.json"
    md_path = args.outdir / "hippo002_tomogram_table_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
