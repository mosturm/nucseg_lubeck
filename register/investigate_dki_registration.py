#!/usr/bin/env python3
"""Keep DKI separate and report geometric/orientation registration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_mri_segmentation_features import MRI_COLUMNS, estimate_local_bases
from validate_table_against_nifti import hypothesis_scores


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        str(q): float(np.quantile(finite, q))
        for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    } if len(finite) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-um", type=float, default=0.65)
    parser.add_argument("--source-nifti", type=Path)
    parser.add_argument("--contrast", default="FA")
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    frame = pd.read_csv(args.table, low_memory=False)
    rows: list[pd.DataFrame] = []
    geometry_report: dict[str, object] = {}
    for tomo in ("0022", "0029"):
        centers, bases, inverse = estimate_local_bases(frame, tomo, 3)
        valid = np.isfinite(inverse).all(axis=(1, 2))
        determinants = np.full(len(frame), np.nan)
        conditions = np.full(len(frame), np.nan)
        axis_lengths = np.full((len(frame), 3), np.nan)
        determinants[valid] = np.linalg.det(bases[valid])
        conditions[valid] = np.linalg.cond(bases[valid])
        axis_lengths[valid] = np.linalg.norm(bases[valid], axis=1) * args.voxel_size_um
        part = frame[MRI_COLUMNS].copy()
        part["tomogram"] = tomo
        part["basis_valid"] = valid.astype(np.uint8)
        part["jacobian_determinant"] = determinants
        part["jacobian_condition"] = conditions
        for axis in range(3):
            part[f"mri_axis_{axis}_step_um"] = axis_lengths[:, axis]
        rows.append(part)
        signs = np.sign(determinants[valid])
        geometry_report[tomo] = {
            "mapped_centers": int(np.isfinite(centers).all(axis=1).sum()),
            "valid_local_bases": int(valid.sum()),
            "negative_jacobian_fraction": float(np.mean(signs < 0)) if len(signs) else None,
            "determinant_quantiles": quantiles(determinants),
            "condition_quantiles": quantiles(conditions),
            "axis_step_um_quantiles": [quantiles(axis_lengths[:, axis]) for axis in range(3)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(
        args.output_dir / "dki_local_geometry_diagnostics.csv", index=False
    )

    orientation = {"status": "not_run", "reason": "No source NIfTI supplied"}
    if args.source_nifti is not None:
        import nibabel as nib

        image = np.asanyarray(nib.load(str(args.source_nifti)).dataobj)
        selected = frame
        if len(selected) > args.max_samples:
            selected = selected.sample(args.max_samples, random_state=46)
        scores = hypothesis_scores(
            selected[MRI_COLUMNS].to_numpy(np.int64),
            selected[args.contrast].to_numpy(float),
            image,
        )
        identity = next(
            (
                index + 1
                for index, score in enumerate(scores)
                if score["permutation"] == [0, 1, 2] and score["flips"] == [False] * 3
            ),
            None,
        )
        orientation = {
            "status": "complete",
            "source_nifti": str(args.source_nifti),
            "contrast": args.contrast,
            "identity_rank": identity,
            "best_hypothesis": scores[0] if scores else None,
            "all_hypotheses": scores,
        }

    report = {
        "schema_version": 1,
        "table": str(args.table),
        "status": "NOT_CLEARED_FOR_CORRELATION",
        "reason": (
            "The collaborator flagged DKI registration as suspicious. Clear it only after "
            "source-NIfTI identity indexing, overlays, and landmark checks pass."
        ),
        "geometry": geometry_report,
        "source_nifti_orientation": orientation,
    }
    (args.output_dir / "dki_registration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "orientation": orientation["status"]}, indent=2))


if __name__ == "__main__":
    main()
