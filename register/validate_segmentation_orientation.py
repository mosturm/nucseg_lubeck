#!/usr/bin/env python3
"""Rank MRI axis permutation/flip hypotheses against registered cell-center controls."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binary_metrics(expected: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(expected & observed))
    fp = int(np.sum(~expected & observed))
    fn = int(np.sum(expected & ~observed))
    tn = int(np.sum(~expected & ~observed))
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": float(dice),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "mcc": float(mcc),
    }


def math_sqrt(value: int) -> float:
    return float(np.sqrt(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--control-nifti", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-identity-best", action="store_true")
    args = parser.parse_args()

    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("Install nibabel: python -m pip install nibabel") from exc

    frame = pd.read_csv(args.features, low_memory=False)
    required = ["mri_i", "mri_j", "mri_k", "center_cell_foreground"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame.dropna(subset=required)
    coordinates = frame[["mri_i", "mri_j", "mri_k"]].to_numpy(np.int64)
    expected = frame["center_cell_foreground"].to_numpy(float) > 0
    image = np.asanyarray(nib.load(str(args.control_nifti)).dataobj)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D control NIfTI, got {image.shape}")

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
            selected = indices[in_bounds]
            if len(selected) < 10:
                continue
            observed = image[selected[:, 0], selected[:, 1], selected[:, 2]] > 0
            metrics = binary_metrics(expected[in_bounds], observed)
            in_bounds_fraction = float(np.mean(in_bounds))
            scores.append(
                {
                    "permutation": list(permutation),
                    "flips": list(flips),
                    "n": int(len(selected)),
                    "in_bounds_fraction": in_bounds_fraction,
                    "orientation_score": float(metrics["mcc"] * in_bounds_fraction),
                    **metrics,
                }
            )
    scores.sort(
        key=lambda item: (-item["orientation_score"], -item["mcc"], -item["dice"], -item["n"])
    )
    identity_rank = next(
        (
            rank
            for rank, score in enumerate(scores, start=1)
            if score["permutation"] == [0, 1, 2] and score["flips"] == [False, False, False]
        ),
        None,
    )
    report = {
        "schema_version": 1,
        "features": str(args.features),
        "control_nifti": str(args.control_nifti),
        "control_shape": list(image.shape),
        "positive_fraction_from_segmentation_centers": float(np.mean(expected)),
        "identity_rank": identity_rank,
        "identity_is_best": identity_rank == 1,
        "best": scores[0] if scores else None,
        "hypotheses": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identity_rank": identity_rank, "best": scores[0] if scores else None}, indent=2))
    if args.require_identity_best and identity_rank != 1:
        raise SystemExit("Identity MRI orientation was not the best hypothesis")


if __name__ == "__main__":
    main()
