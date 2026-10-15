#!/usr/bin/env python3
"""Audit full registration tables and cluster segmentation-mask availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_registration_config import (
    PRIMARY_EXCLUDED_ACQUISITIONS,
    SPECIMENS,
    cell_mask_path,
    discover_tables,
    parse_tomogram_tokens,
    vessel_mask_path,
)


def owner_for_row(row: pd.Series, specimen: str) -> str | None:
    allowed = SPECIMENS[specimen]["tomograms"]
    usable = parse_tomogram_tokens(row.get("tomogram_usable"), allowed)
    touched = parse_tomogram_tokens(row.get("tomogram"), allowed)
    return (usable or touched or [None])[0]


def coordinate_columns(tomos: tuple[str, ...]) -> list[str]:
    columns = ["tomogram", "tomogram_usable", "coord_usable", "coverage_percent"]
    for tomo in tomos:
        columns.extend(f"tomo{tomo}_{axis}_650nm" for axis in "ijk")
        for corner in ("000", "001", "010", "011", "100", "101", "110", "111"):
            columns.extend(f"c{corner}_tomo{tomo}_{axis}_650nm" for axis in "ijk")
    return columns


def direct_geometry_columns(tomo: str) -> list[str]:
    columns = [f"tomo{tomo}_{axis}_650nm" for axis in "ijk"]
    for corner in ("000", "001", "010", "011", "100", "101", "110", "111"):
        columns.extend(f"c{corner}_tomo{tomo}_{axis}_650nm" for axis in "ijk")
    return columns


def audit_table(spec, chunk_size: int) -> dict[str, object]:
    header = list(pd.read_csv(spec.path, nrows=0).columns)
    required = [
        "mri_i",
        "mri_j",
        "mri_k",
        "coverage_percent",
        "coord_usable",
        "tomogram",
        "tomogram_usable",
        *spec.predictors,
        *coordinate_columns(SPECIMENS[spec.specimen]["tomograms"]),
    ]
    required = list(dict.fromkeys(required))
    missing = sorted(set(required) - set(header))
    if missing:
        return {
            "dataset_id": spec.dataset_id,
            "input_table": str(spec.path),
            "status": "missing_columns",
            "missing_columns": ";".join(missing),
        }

    totals = {
        "rows": 0,
        "coord_usable": 0,
        "usable_coverage_60": 0,
        "usable_coverage_70": 0,
        "usable_coverage_80": 0,
        "usable_coverage_100": 0,
        "usable_rows_without_owner": 0,
        "usable_rows_without_direct_geometry": 0,
        "duplicate_mri_coordinates": 0,
    }
    seen_coordinates: set[tuple[int, int, int]] = set()
    predictor_finite_80 = {predictor: 0 for predictor in spec.predictors}
    usecols = list(dict.fromkeys(["mri_i", "mri_j", "mri_k", *required]))
    for frame in pd.read_csv(spec.path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        totals["rows"] += len(frame)
        usable = pd.to_numeric(frame["coord_usable"], errors="coerce").eq(1)
        coverage = pd.to_numeric(frame["coverage_percent"], errors="coerce")
        totals["coord_usable"] += int(usable.sum())
        for threshold in (60, 70, 80, 100):
            totals[f"usable_coverage_{threshold}"] += int((usable & (coverage >= threshold)).sum())

        for values in frame[["mri_i", "mri_j", "mri_k"]].itertuples(index=False, name=None):
            coordinate = tuple(int(value) for value in values)
            if coordinate in seen_coordinates:
                totals["duplicate_mri_coordinates"] += 1
            seen_coordinates.add(coordinate)

        eligible80 = usable & (coverage >= 80)
        for predictor in spec.predictors:
            values = pd.to_numeric(frame[predictor], errors="coerce")
            predictor_finite_80[predictor] += int((eligible80 & np.isfinite(values)).sum())

        allowed = SPECIMENS[spec.specimen]["tomograms"]
        owners = []
        for usable_value, touched_value in zip(frame["tomogram_usable"], frame["tomogram"]):
            candidates = parse_tomogram_tokens(usable_value, allowed)
            candidates.extend(
                tomo
                for tomo in parse_tomogram_tokens(touched_value, allowed)
                if tomo not in candidates
            )
            owners.append(candidates[0] if candidates else None)
        owners = np.asarray(owners, dtype=object)
        usable_array = usable.to_numpy()
        totals["usable_rows_without_owner"] += int((usable_array & pd.isna(owners)).sum())
        for tomo in allowed:
            selected = usable_array & (owners == tomo)
            if not selected.any():
                continue
            values = frame.loc[selected, direct_geometry_columns(tomo)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            totals["usable_rows_without_direct_geometry"] += int(
                (~np.isfinite(values).all(axis=1)).sum()
            )

    return {
        "dataset_id": spec.dataset_id,
        "acquisition": spec.acquisition,
        "specimen": spec.specimen,
        "protocol": spec.protocol,
        "predictors": ";".join(spec.predictors),
        "primary_analysis_enabled": int(spec.acquisition not in PRIMARY_EXCLUDED_ACQUISITIONS),
        "status": "ok" if not any(
            totals[key]
            for key in (
                "duplicate_mri_coordinates",
                "usable_rows_without_owner",
                "usable_rows_without_direct_geometry",
            )
        ) else "coordinate_problem",
        **totals,
        "predictor_finite_usable80": json.dumps(predictor_finite_80, sort_keys=True),
        "input_table": str(spec.path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--allow-missing-merged-cells", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_tables(args.input_root)
    table_rows = [audit_table(spec, args.chunk_size) for spec in specs]
    pd.DataFrame(table_rows).to_csv(args.output_dir / "registration_table_audit.csv", index=False)

    mask_rows = []
    for specimen, config in SPECIMENS.items():
        for tomo in config["tomograms"]:
            merged = cell_mask_path(args.data_root, specimen, tomo, merged=True)
            unmerged = cell_mask_path(args.data_root, specimen, tomo, merged=False)
            vessel = vessel_mask_path(args.data_root, specimen, tomo)
            mask_rows.append(
                {
                    "specimen": specimen,
                    "sample_id": config["sample_id"],
                    "tomogram": tomo,
                    "merged_cell_mask": str(merged),
                    "merged_cell_available": int(merged.is_dir()),
                    "unmerged_cell_mask": str(unmerged),
                    "unmerged_cell_available": int(unmerged.is_dir()),
                    "vessel_mask": str(vessel),
                    "vessel_available": int(vessel.is_dir()),
                }
            )
    masks = pd.DataFrame(mask_rows)
    masks.to_csv(args.output_dir / "segmentation_mask_inventory.csv", index=False)

    bad_tables = [row["dataset_id"] for row in table_rows if row.get("status") != "ok"]
    missing_merged = masks.loc[masks["merged_cell_available"] == 0]
    report = {
        "schema_version": 1,
        "table_count": len(specs),
        "specimen_count": len(SPECIMENS),
        "tomogram_count": int(len(masks)),
        "tables_with_problems": bad_tables,
        "merged_cell_masks_available": int(masks["merged_cell_available"].sum()),
        "merged_cell_masks_required": int(len(masks)),
        "vessel_masks_available": int(masks["vessel_available"].sum()),
        "vessel_masks_optional": True,
        "primary_excluded_acquisitions": sorted(PRIMARY_EXCLUDED_ACQUISITIONS),
    }
    (args.output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if bad_tables:
        raise SystemExit("Registration-table audit failed")
    if len(missing_merged) and not args.allow_missing_merged_cells:
        print(missing_merged[["specimen", "sample_id", "tomogram"]].to_string(index=False))
        raise SystemExit("Merged cell masks are missing; radius extraction is not reproducible")


if __name__ == "__main__":
    main()
