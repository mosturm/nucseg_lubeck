#!/usr/bin/env python3
"""Analyze full-run MRI-histology associations with honest grouped validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from full_registration_config import PRIMARY_EXCLUDED_ACQUISITIONS


TARGETS = [
    "f_soma_seg",
    "cell_density_per_mm3",
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r5_r3_um",
    "cell_radius_effective_r7_r3_um",
    "cell_surface_to_volume_r23_um_inv",
    "cell_surface_density_um_inv",
    "f_vessel_seg",
]
RADIUS_TARGETS = {
    "cell_radius_mean_um",
    "cell_radius_median_um",
    "cell_radius_volume_weighted_um",
    "cell_radius_effective_r5_r3_um",
    "cell_radius_effective_r7_r3_um",
    "cell_surface_to_volume_r23_um_inv",
}


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return output
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return output


def safe_correlation(function, x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    result = function(x, y)
    return float(result.statistic), float(result.pvalue)


def spatial_groups(frame: pd.DataFrame, block_size: int) -> np.ndarray:
    coordinates = frame[["mri_i", "mri_j", "mri_k"]].to_numpy(np.int64)
    blocks = np.floor_divide(coordinates, block_size)
    specimen = frame["specimen"].astype(str).to_numpy()
    return np.array(
        [f"{sample}:{i}:{j}:{k}" for sample, (i, j, k) in zip(specimen, blocks)],
        dtype=object,
    )


def balanced_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < folds:
        raise ValueError(f"Only {len(unique)} groups for {folds} folds")
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(unique)), -counts))
    load = np.zeros(folds, dtype=int)
    assignment = np.zeros(len(unique), dtype=int)
    for group in order:
        fold = int(np.argmin(load))
        assignment[group] = fold
        load[fold] += counts[group]
    return assignment[inverse]


def predictions(x, y, frame, scope: str, block_size: int, folds: int, seed: int):
    if scope == "pooled_protocol":
        specimens = frame["specimen"].astype(str).to_numpy()
        unique = np.unique(specimens)
        if len(unique) < 2:
            raise ValueError("Pooled validation requires at least two specimens")
        fold_ids = np.array([np.flatnonzero(unique == specimen)[0] for specimen in specimens])
        scheme = "leave_one_specimen_out"
        group_count = len(unique)
    else:
        groups = spatial_groups(frame, block_size)
        fold_ids = balanced_folds(groups, min(folds, len(np.unique(groups))), seed)
        scheme = f"spatial_block_{len(np.unique(fold_ids))}fold"
        group_count = len(np.unique(groups))
    predicted = np.full(len(y), np.nan)
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        test = ~train
        if train.sum() < 3 or np.ptp(x[train]) == 0:
            predicted[test] = np.mean(y[train])
        else:
            fit = stats.linregress(x[train], y[train])
            predicted[test] = fit.intercept + fit.slope * x[test]
    return predicted, scheme, group_count


def association(frame, predictor, target, scope, block_size, folds, seed):
    x = frame[predictor].to_numpy(float)
    y = frame[target].to_numpy(float)
    pearson, pearson_p = safe_correlation(stats.pearsonr, x, y)
    spearman, spearman_p = safe_correlation(stats.spearmanr, x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    fit = stats.linregress(x, y)
    predicted, scheme, group_count = predictions(
        x, y, frame, scope, block_size, folds, seed
    )
    denominator = np.sum((y - np.mean(y)) ** 2)
    cv_r2 = 1.0 - np.sum((y - predicted) ** 2) / denominator if denominator > 0 else math.nan
    return {
        "n": len(frame),
        "n_specimens": frame["specimen"].nunique(),
        "n_validation_groups": group_count,
        "validation_scheme": scheme,
        "linear_slope": float(fit.slope),
        "linear_intercept": float(fit.intercept),
        "linear_r2_in_sample": float(fit.rvalue**2),
        "linear_p": float(fit.pvalue),
        "pearson_r": pearson,
        "pearson_p": pearson_p,
        "spearman_rho": spearman,
        "spearman_p": spearman_p,
        "heldout_r2": float(cv_r2),
        "heldout_mae": float(np.mean(np.abs(y - predicted))),
        "heldout_rmse": float(np.sqrt(np.mean((y - predicted) ** 2))),
    }


def eligible_pair(frame, predictor, target, coverage_min, min_cells):
    columns = [
        predictor,
        target,
        "specimen",
        "mri_i",
        "mri_j",
        "mri_k",
        "cell_count",
        "coord_usable",
        "coverage_percent",
        "local_basis_valid",
        "covered_footprint_volume_um3",
    ]
    selected = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    selected = selected[
        (pd.to_numeric(selected["coord_usable"], errors="coerce") == 1)
        & (pd.to_numeric(selected["coverage_percent"], errors="coerce") >= coverage_min)
        & (selected["local_basis_valid"] == 1)
        & (selected["covered_footprint_volume_um3"] > 0)
    ]
    if target in RADIUS_TARGETS:
        selected = selected[selected["cell_count"] >= min_cells]
    return selected


def write_raw_table(frame, predictors, coverage_min, min_cells, path):
    output = frame.copy()
    base = (
        (pd.to_numeric(output["coord_usable"], errors="coerce") == 1)
        & (pd.to_numeric(output["coverage_percent"], errors="coerce") >= coverage_min)
        & (output["local_basis_valid"] == 1)
        & (output["covered_footprint_volume_um3"] > 0)
    )
    output["has_positive_coverage"] = (
        pd.to_numeric(output["coverage_percent"], errors="coerce") > 0
    ).astype(np.uint8)
    output["included_in_base_analysis"] = base.astype(np.uint8)
    output["included_for_radius_regressions"] = (
        base & (output["cell_count"] >= min_cells)
    ).astype(np.uint8)
    output.to_csv(path, index=False, compression="gzip")


def theory_fit(frame, r1_column, scope, block_size, folds, seed):
    selected = eligible_pair(
        frame,
        r1_column,
        "cell_surface_to_volume_r23_um_inv",
        coverage_min=frame.attrs["coverage_min"],
        min_cells=frame.attrs["min_cells"],
    )
    if len(selected) < 15:
        return None
    x = selected["cell_surface_to_volume_r23_um_inv"].to_numpy(float)
    y = selected[r1_column].to_numpy(float)
    fit = stats.linregress(x, y)
    predicted, scheme, groups = predictions(x, y, selected, scope, block_size, folds, seed)
    denominator = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": len(selected),
        "n_specimens": selected["specimen"].nunique(),
        "validation_scheme": scheme,
        "n_validation_groups": groups,
        "r1_column": r1_column,
        "r23_slope": float(fit.slope),
        "r1_free_intercept": float(fit.intercept),
        "in_sample_r2": float(fit.rvalue**2),
        "heldout_r2": float(1.0 - np.sum((y - predicted) ** 2) / denominator),
        "spearman_rho": float(stats.spearmanr(x, y).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-min", type=float, default=80.0)
    parser.add_argument("--min-cells-radius", type=int, default=5)
    parser.add_argument("--spatial-block-size", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--include-portugal-dki-primary", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_voxel_tables"
    raw_dir.mkdir(exist_ok=True)
    manifest = pd.read_csv(args.features_dir / "feature_manifest.csv")
    frames = {}
    for row in manifest.itertuples(index=False):
        frame = pd.read_csv(row.feature_table, low_memory=False)
        frame.attrs["coverage_min"] = args.coverage_min
        frame.attrs["min_cells"] = args.min_cells_radius
        predictors = str(row.predictors).split(";")
        frames[row.dataset_id] = (frame, predictors, row)
        write_raw_table(
            frame,
            predictors,
            args.coverage_min,
            args.min_cells_radius,
            raw_dir / f"{row.dataset_id}_raw_voxel_values.csv.gz",
        )

    scopes = []
    for dataset_id, (frame, predictors, row) in frames.items():
        scopes.append(("dataset", dataset_id, frame, predictors, row.acquisition, row.protocol))
    protocol_groups = manifest.groupby(["acquisition", "protocol"], sort=True)
    for (acquisition, protocol), group in protocol_groups:
        ids = group["dataset_id"].tolist()
        pooled = pd.concat([frames[dataset_id][0] for dataset_id in ids], ignore_index=True)
        predictors = sorted(set.intersection(*(set(frames[dataset_id][1]) for dataset_id in ids)))
        scopes.append(
            ("pooled_protocol", f"{acquisition}__{protocol}", pooled, predictors, acquisition, protocol)
        )

    records, theory_records = [], []
    for scope, scope_id, frame, predictors, acquisition, protocol in scopes:
        primary = acquisition not in PRIMARY_EXCLUDED_ACQUISITIONS or args.include_portugal_dki_primary
        for predictor in predictors:
            for target in TARGETS:
                selected = eligible_pair(
                    frame, predictor, target, args.coverage_min, args.min_cells_radius
                )
                minimum = 15 if scope == "dataset" else 20
                if len(selected) < minimum:
                    continue
                try:
                    result = association(
                        selected,
                        predictor,
                        target,
                        scope,
                        args.spatial_block_size,
                        args.folds,
                        args.seed,
                    )
                except ValueError:
                    continue
                if result is not None:
                    records.append(
                        {
                            "scope": scope,
                            "scope_id": scope_id,
                            "acquisition": acquisition,
                            "protocol": protocol,
                            "primary_analysis": int(primary),
                            "predictor": predictor,
                            "target": target,
                            **result,
                        }
                    )
        for r1_column in [column for column in predictors if column in {"R1", "R1_denoised"}]:
            frame.attrs["coverage_min"] = args.coverage_min
            frame.attrs["min_cells"] = args.min_cells_radius
            try:
                result = theory_fit(
                    frame,
                    r1_column,
                    scope,
                    args.spatial_block_size,
                    args.folds,
                    args.seed,
                )
            except ValueError:
                result = None
            if result:
                theory_records.append(
                    {
                        "scope": scope,
                        "scope_id": scope_id,
                        "acquisition": acquisition,
                        "protocol": protocol,
                        "primary_analysis": int(primary),
                        **result,
                    }
                )

    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No eligible MRI-histology associations")
    results["spearman_p_fdr_bh"] = np.nan
    for key, index in results.groupby(["scope", "primary_analysis"]).groups.items():
        results.loc[index, "spearman_p_fdr_bh"] = bh_adjust(results.loc[index, "spearman_p"])
    results = results.sort_values(["scope", "scope_id", "target", "predictor"])
    results.to_csv(args.output_dir / "all_univariate_correlations_and_grouped_cv.csv", index=False)
    results[(results["scope"] == "pooled_protocol") & (results["primary_analysis"] == 1)].to_csv(
        args.output_dir / "primary_pooled_protocol_correlations_and_loso_cv.csv", index=False
    )
    pd.DataFrame(theory_records).to_csv(args.output_dir / "r1_vs_r23_theory_fits.csv", index=False)

    feasibility = pd.DataFrame(
        [
            ["Core univariate heatmaps", "yes", "All acquisitions except Portugal DKI are primary by default."],
            ["R1 versus r23", "yes", "Includes Leipzig 7T and all available R1 grids; no r3/r2 analysis."],
            ["MTsat surface test", "limited", "Only HH in-vivo MPM; P5/P7 have very few >=80% voxels."],
            ["SV/PD water normalization", "limited", "PD is available only in HH in-vivo; ex-vivo A is not absolute PD."],
            ["Multi-field slope comparison", "partial", "Leipzig 7T is present; no 9.4T MPM and no confirmed 1.5T MPM."],
            ["Portugal DKI", "excluded", "Collaborator flagged registration as suspicious; outputs retained for audit."],
            ["Vessel associations", "conditional", "Computed only for MRI voxels whose selected tomogram has a vessel mask."],
        ],
        columns=["experiment", "status", "reason"],
    )
    feasibility.to_csv(args.output_dir / "experiment_feasibility.csv", index=False)
    report = {
        "schema_version": 1,
        "coverage_min": args.coverage_min,
        "minimum_cells_for_radius_metrics": args.min_cells_radius,
        "dataset_count": int((manifest["dataset_id"]).nunique()),
        "pooled_protocol_count": int(manifest.groupby(["acquisition", "protocol"]).ngroups),
        "specimens": sorted(manifest["specimen"].unique().tolist()),
        "r23_only": True,
        "r32_excluded": True,
        "primary_excluded_acquisitions": [] if args.include_portugal_dki_primary else sorted(PRIMARY_EXCLUDED_ACQUISITIONS),
        "association_count": len(results),
        "primary_pooled_association_count": int(
            ((results["scope"] == "pooled_protocol") & (results["primary_analysis"] == 1)).sum()
        ),
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
