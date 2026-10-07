#!/usr/bin/env python3
"""Validate table MRI indices against source NIfTI values and axis hypotheses."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hypothesis_scores(
    coordinates: np.ndarray,
    expected: np.ndarray,
    image: np.ndarray,
) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    for permutation in itertools.permutations(range(3)):
        base = coordinates[:, permutation]
        for flips in itertools.product((False, True), repeat=3):
            indices = base.copy()
            for axis, flip in enumerate(flips):
                if flip:
                    indices[:, axis] = image.shape[axis] - 1 - indices[:, axis]
            in_bounds = np.ones(len(indices), dtype=bool)
            for axis in range(3):
                in_bounds &= (indices[:, axis] >= 0) & (indices[:, axis] < image.shape[axis])
            if not in_bounds.any():
                continue
            selected = indices[in_bounds].astype(np.int64)
            observed = image[selected[:, 0], selected[:, 1], selected[:, 2]].astype(float)
            target = expected[in_bounds]
            finite = np.isfinite(observed) & np.isfinite(target)
            if finite.sum() < 3:
                continue
            difference = observed[finite] - target[finite]
            correlation = float(np.corrcoef(observed[finite], target[finite])[0, 1])
            scores.append(
                {
                    "permutation": list(permutation),
                    "flips": list(flips),
                    "in_bounds_fraction": float(in_bounds.mean()),
                    "n_compared": int(finite.sum()),
                    "mae": float(np.mean(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "correlation": correlation,
                    "exact_fraction": float(np.mean(np.isclose(difference, 0, atol=1e-6, rtol=1e-5))),
                }
            )
    return sorted(scores, key=lambda item: (-item["exact_fraction"], item["rmse"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--column", required=True, help="Contrast column represented by --nifti")
    parser.add_argument("--nifti", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-identity-best", action="store_true")
    args = parser.parse_args()

    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("Install nibabel in the environment: pip install nibabel") from exc

    frame = pd.read_csv(args.table)
    required = ["mri_i", "mri_j", "mri_k", args.column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) > args.max_samples:
        frame = frame.sample(args.max_samples, random_state=args.seed)

    coordinates = frame[["mri_i", "mri_j", "mri_k"]].to_numpy(np.int64)
    expected = frame[args.column].to_numpy(float)
    nii = nib.load(str(args.nifti))
    image = np.asanyarray(nii.dataobj)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI, got {image.shape}")

    scores = hypothesis_scores(coordinates, expected, image)
    report = {
        "table": str(args.table.resolve()),
        "column": args.column,
        "nifti": str(args.nifti.resolve()),
        "shape": list(image.shape),
        "zooms_mm": list(nii.header.get_zooms()[:3]),
        "affine": np.asarray(nii.affine).tolist(),
        "axis_codes": list(nib.aff2axcodes(nii.affine)),
        "qform_code": int(nii.header["qform_code"]),
        "sform_code": int(nii.header["sform_code"]),
        "affine_determinant": float(np.linalg.det(nii.affine[:3, :3])),
        "hypotheses": scores,
        "identity_is_best": bool(
            scores
            and scores[0]["permutation"] == [0, 1, 2]
            and scores[0]["flips"] == [False, False, False]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores[:5], indent=2))
    print(f"Identity is best: {report['identity_is_best']}")
    print(f"Wrote {args.output}")
    if args.require_identity_best and not report["identity_is_best"]:
        raise SystemExit("Identity MRI index orientation was not the best hypothesis")


if __name__ == "__main__":
    main()
