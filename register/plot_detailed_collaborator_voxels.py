#!/usr/bin/env python3
"""Plot collaborator-requested per-MRI-voxel soma-radius distributions."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


R23 = "cell_surface_to_volume_r23_um_inv"
REGIONS = {
    "above_sulcus": ("0001", "0008", "0015"),
    "below_sulcus": ("0022", "0029", "0036"),
}
REGION_LABELS = {
    "above_sulcus": "Above sulcus: tomograms 0001, 0008, 0015",
    "below_sulcus": "Below sulcus: tomograms 0022, 0029, 0036",
}
REGION_COLORS = {"above_sulcus": "#3974D6", "below_sulcus": "#D92F2F"}
CATEGORY_COLORS = {"low": "#2A9D8F", "middle": "#D99B2B", "high": "#D1495B"}
COHORTS = {
    "hamburg": {
        "dataset_id": "HH_exvivo__P2__mpm_0p3mm",
        "title": "P2 | Hamburg ex vivo | mpm_0p3mm",
        "reference": {
            "above_sulcus": {"n": 346, "slope": 2.256676204054247},
            "below_sulcus": {"n": 361, "slope": 1.796008563883977},
        },
    },
    "leipzig": {
        "dataset_id": "Leipzig__P2__mpm_0p3mm",
        "title": "P2 | Leipzig | mpm_0p3mm",
        "reference": {
            "above_sulcus": {"n": 279, "slope": 2.1565575825809398},
            "below_sulcus": {"n": 344, "slope": 2.357996100237812},
        },
    },
}
QUANTILE_BANDS = {
    "low": (0.10, 0.05, 0.15),
    "middle": (0.50, 0.45, 0.55),
    "high": (0.90, 0.85, 0.95),
}
REQUIRED_RAW_COLUMNS = {
    "mri_table_row",
    "mri_i",
    "mri_j",
    "mri_k",
    "R1",
    "analysis_tomogram",
    "coverage_percent",
    "coord_usable",
    "cell_count",
    "included_in_base_analysis",
    "included_for_radius_regressions",
    R23,
}
REQUIRED_RADIUS_COLUMNS = {"mri_table_row", "cell_radius_um"}


def normalize_tomogram(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.map(lambda value: f"{int(value):04d}" if np.isfinite(value) else "")


def raw_path(source_run: Path, dataset_id: str) -> Path:
    return (
        source_run
        / "analysis_cov_80"
        / "raw_voxel_tables"
        / f"{dataset_id}_raw_voxel_values.csv.gz"
    )


def radius_path(source_run: Path, dataset_id: str) -> Path:
    return (
        source_run
        / "features"
        / "per_cell_radius_tables"
        / f"{dataset_id}_cell_radii_by_mri_voxel.csv.gz"
    )


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def summarize_radii(values: pd.Series) -> pd.Series:
    radii = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(radii):
        return pd.Series(dtype=float)
    q25, median, q75, p90, p95 = np.quantile(radii, [0.25, 0.50, 0.75, 0.90, 0.95])
    return pd.Series(
        {
            "distribution_cell_count": len(radii),
            "radius_mean_um": float(np.mean(radii)),
            "radius_std_um": float(np.std(radii, ddof=1)) if len(radii) > 1 else math.nan,
            "radius_median_um_empirical": float(median),
            "radius_iqr_um": float(q75 - q25),
            "radius_p90_um": float(p90),
            "radius_p95_um": float(p95),
            "radius_max_um": float(np.max(radii)),
            "radius_skewness": float(stats.skew(radii, bias=False)) if len(radii) > 2 else math.nan,
            "radius_tail_ratio_p95_median": float(p95 / median) if median > 0 else math.nan,
        }
    )


def fit_gev(radii: np.ndarray) -> dict[str, float]:
    if len(radii) < 10 or np.ptp(radii) == 0:
        return {"gev_shape_scipy": math.nan, "gev_location_um": math.nan, "gev_scale_um": math.nan}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, location, scale = stats.genextreme.fit(radii)
        if not np.isfinite([shape, location, scale]).all() or scale <= 0:
            raise ValueError("invalid GEV fit")
        return {
            "gev_shape_scipy": float(shape),
            "gev_location_um": float(location),
            "gev_scale_um": float(scale),
        }
    except Exception:
        return {"gev_shape_scipy": math.nan, "gev_location_um": math.nan, "gev_scale_um": math.nan}


def distribution_panel(
    axis,
    radii: np.ndarray,
    title: str,
    color: str,
    radius_max: float,
) -> dict[str, float]:
    radii = radii[np.isfinite(radii)]
    if not len(radii):
        axis.text(0.5, 0.5, "No assigned cells", ha="center", va="center")
        axis.set_title(title)
        return {"gev_shape_scipy": math.nan, "gev_location_um": math.nan, "gev_scale_um": math.nan}
    clipped = radii[(radii >= 0) & (radii <= radius_max)]
    bins = np.histogram_bin_edges(clipped, bins="fd") if len(clipped) > 1 else np.linspace(0, radius_max, 8)
    if len(bins) < 7:
        bins = np.linspace(0, radius_max, 12)
    axis.hist(clipped, bins=bins, density=True, color=color, alpha=0.28, edgecolor="white")
    x = np.linspace(0.0, radius_max, 500)
    if len(radii) >= 3 and np.ptp(radii) > 0:
        try:
            kde = stats.gaussian_kde(radii)
            axis.plot(x, kde(x), color=color, linewidth=2.2, label="empirical KDE")
        except np.linalg.LinAlgError:
            pass
    gev = fit_gev(radii)
    if np.isfinite(gev["gev_scale_um"]):
        density = stats.genextreme.pdf(
            x,
            gev["gev_shape_scipy"],
            loc=gev["gev_location_um"],
            scale=gev["gev_scale_um"],
        )
        density[~np.isfinite(density)] = np.nan
        axis.plot(x, density, color="#333333", linestyle="--", linewidth=1.5, label="GEV fit")
    axis.axvline(float(np.mean(radii)), color=color, linestyle=":", linewidth=2)
    axis.set_xlim(0.0, radius_max)
    axis.set_xlabel("Sphere-equivalent soma radius (um)")
    axis.set_ylabel("Density")
    axis.set_title(title, fontsize=9.5)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=7.5)
    return gev


def select_random_quantile_examples(
    frame: pd.DataFrame,
    examples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    ranked = frame.copy().sort_values(R23).reset_index(drop=True)
    ranked["r23_percentile_rank"] = ranked[R23].rank(method="average", pct=True)
    chosen_rows: list[pd.Series] = []
    used: set[int] = set()
    for category, (target, lower, upper) in QUANTILE_BANDS.items():
        candidates = ranked[
            ranked["r23_percentile_rank"].between(lower, upper, inclusive="both")
            & ~ranked.index.isin(used)
        ]
        if len(candidates) < examples:
            available = ranked.loc[~ranked.index.isin(used)].copy()
            available["distance"] = np.abs(available["r23_percentile_rank"] - target)
            candidates = available.nsmallest(max(examples, min(20, len(available))), "distance")
        indices = rng.choice(candidates.index.to_numpy(), size=examples, replace=False)
        selected = ranked.loc[indices].sort_values(R23)
        for rank, (index, row) in enumerate(selected.iterrows(), 1):
            used.add(int(index))
            row = row.copy()
            row["selection_category"] = category
            row["selection_rank"] = rank
            row["selection_label"] = f"{category.capitalize()} {rank}"
            row["target_r23_quantile"] = target
            row["selection_band_low"] = lower
            row["selection_band_high"] = upper
            chosen_rows.append(row)
    return pd.DataFrame(chosen_rows)


def save_figure(figure, base: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def task_one(
    source_run: Path,
    output_dir: Path,
    min_cells: int,
    examples: int,
    seed: int,
    radius_max: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_selected = []
    selected_radii = []
    line_records = []
    rng = np.random.default_rng(seed)

    for cohort_key, config in COHORTS.items():
        dataset_id = config["dataset_id"]
        source_raw = require_file(raw_path(source_run, dataset_id))
        source_radii = require_file(radius_path(source_run, dataset_id))
        raw = pd.read_csv(source_raw, low_memory=False)
        radii = pd.read_csv(source_radii, low_memory=False)
        require_columns(raw, REQUIRED_RAW_COLUMNS, source_raw)
        require_columns(radii, REQUIRED_RADIUS_COLUMNS, source_radii)
        raw["normalized_tomogram"] = normalize_tomogram(raw["analysis_tomogram"])
        region_frames = {}
        region_selected = {}

        for region, tomograms in REGIONS.items():
            eligible = raw[
                (pd.to_numeric(raw["included_for_radius_regressions"], errors="coerce") == 1)
                & raw["normalized_tomogram"].isin(tomograms)
                & (pd.to_numeric(raw["cell_count"], errors="coerce") >= min_cells)
                & pd.to_numeric(raw["R1"], errors="coerce").notna()
                & pd.to_numeric(raw[R23], errors="coerce").notna()
            ].copy()
            if len(eligible) < len(QUANTILE_BANDS) * examples:
                raise RuntimeError(f"Only {len(eligible)} eligible rows for {dataset_id} {region}")
            fit = stats.linregress(eligible[R23], eligible["R1"])
            reference = config["reference"][region]
            line_records.append(
                {
                    "cohort": cohort_key,
                    "dataset_id": dataset_id,
                    "region": region,
                    "tomograms": "+".join(tomograms),
                    "n": len(eligible),
                    "intercept": float(fit.intercept),
                    "slope": float(fit.slope),
                    "linear_r2": float(fit.rvalue**2),
                    "reference_image_n": reference["n"],
                    "reference_image_slope": reference["slope"],
                    "n_matches_reference": int(len(eligible) == reference["n"]),
                    "slope_difference_from_reference": float(fit.slope - reference["slope"]),
                }
            )
            selected = select_random_quantile_examples(eligible, examples, rng)
            selected.insert(0, "region", region)
            selected.insert(0, "cohort", cohort_key)
            region_frames[region] = eligible
            region_selected[region] = selected
            all_selected.append(selected)

            figure, axes = plt.subplots(3, examples, figsize=(18, 13.5), constrained_layout=True)
            axes = np.asarray(axes).reshape(3, examples)
            records = []
            for position, (_, row) in enumerate(selected.iterrows()):
                grid_row, grid_column = divmod(position, examples)
                voxel_radii = radii.loc[
                    radii["mri_table_row"] == row["mri_table_row"], "cell_radius_um"
                ].to_numpy(float)
                title = (
                    f"{row['selection_label']}: r23={row[R23]:.3f} 1/um, R1={row['R1']:.3f}\n"
                    f"MRI ({int(row['mri_i'])},{int(row['mri_j'])},{int(row['mri_k'])}), "
                    f"tomo {row['normalized_tomogram']}, n={len(voxel_radii)}, "
                    f"mean={np.mean(voxel_radii):.2f} um"
                )
                gev = distribution_panel(
                    axes[grid_row, grid_column],
                    voxel_radii,
                    title,
                    CATEGORY_COLORS[str(row["selection_category"])],
                    radius_max,
                )
                request_id = f"{cohort_key}_{region}_{row['selection_category']}_{int(row['selection_rank'])}"
                long = radii[radii["mri_table_row"] == row["mri_table_row"]].copy()
                long.insert(0, "request_id", request_id)
                long.insert(1, "cohort", cohort_key)
                long.insert(2, "region", region)
                long.insert(3, "selection_category", row["selection_category"])
                selected_radii.append(long)
                records.append(
                    {
                        **row.to_dict(),
                        **summarize_radii(pd.Series(voxel_radii)).to_dict(),
                        **gev,
                        "request_id": request_id,
                    }
                )
            figure.suptitle(
                f"{config['title']} | {REGION_LABELS[region]}\n"
                "Reproducible random MRI voxels from the 10th, 50th, and 90th percentile bands",
                fontsize=15,
            )
            save_figure(
                figure,
                output_dir / f"{cohort_key}_P2_{region}_low_middle_high_cell_distributions",
            )
            pd.DataFrame(records).to_csv(
                output_dir / f"{cohort_key}_P2_{region}_selected_voxels.csv", index=False
            )

        figure, axis = plt.subplots(figsize=(11.5, 8.0), constrained_layout=True)
        for region in REGIONS:
            frame = region_frames[region]
            color = REGION_COLORS[region]
            fit = stats.linregress(frame[R23], frame["R1"])
            axis.scatter(frame[R23], frame["R1"], s=24, alpha=0.68, edgecolors="none", color=color)
            xline = np.linspace(frame[R23].min(), frame[R23].max(), 250)
            axis.plot(
                xline,
                fit.intercept + fit.slope * xline,
                color=color,
                linewidth=2.8,
                label=f"{REGION_LABELS[region]} (n={len(frame)}), slope={fit.slope:.3f}",
            )
            for _, row in region_selected[region].iterrows():
                axis.scatter(
                    row[R23],
                    row["R1"],
                    s=105,
                    color=CATEGORY_COLORS[str(row["selection_category"])],
                    edgecolor=color,
                    linewidth=2,
                    zorder=5,
                )
        pooled = pd.concat(region_frames.values(), ignore_index=True)
        pooled_fit = stats.linregress(pooled[R23], pooled["R1"])
        xline = np.linspace(pooled[R23].min(), pooled[R23].max(), 250)
        axis.plot(
            xline,
            pooled_fit.intercept + pooled_fit.slope * xline,
            color="black",
            linestyle="--",
            linewidth=2.1,
            label=f"All P2 (n={len(pooled)}), slope={pooled_fit.slope:.3f}",
        )
        axis.set_xlabel(r"$r_{23}=\sum r^2/\sum r^3$ [$\mu m^{-1}$]")
        axis.set_ylabel("R1")
        axis.set_title(f"{config['title']}: R1 versus r23 by sulcus region")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        save_figure(figure, output_dir / f"{cohort_key}_P2_R1_vs_r23_by_sulcus_region")

    selected_frame = pd.concat(all_selected, ignore_index=True)
    selected_frame.to_csv(output_dir / "task1_all_selected_voxels.csv", index=False)
    pd.concat(selected_radii, ignore_index=True).to_csv(
        output_dir / "task1_all_selected_per_cell_radii.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(line_records).to_csv(output_dir / "task1_reference_line_replication.csv", index=False)
    return selected_frame, line_records


def format_value(value: object, scientific: bool = False) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.3e}" if scientific else f"{numeric:.3f}"


def task_two(
    source_run: Path,
    coordinate_table: Path,
    output_dir: Path,
    radius_max: float,
) -> tuple[pd.DataFrame, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = pd.read_csv(require_file(coordinate_table), sep=";", encoding="utf-8")
    requests = requests[
        requests["kontrast"].astype(str).str.strip().eq("Hamburg ex vivo R1, R2* und MD")
    ].copy()
    if len(requests) != 6:
        raise RuntimeError(f"Expected six Hamburg ex vivo requests, found {len(requests)}")
    raw_cache: dict[str, pd.DataFrame] = {}
    radius_cache: dict[str, pd.DataFrame] = {}
    match_records = []
    long_radii = []
    warnings_out = []
    first_for_voxel: dict[tuple[str, int], str] = {}

    for request in requests.itertuples(index=False):
        dataset_id = re.sub(r"_raw_voxel_values\.csv\.gz$", "", str(request.quelldatei))
        if dataset_id not in raw_cache:
            source_raw = require_file(raw_path(source_run, dataset_id))
            source_radii = require_file(radius_path(source_run, dataset_id))
            raw_cache[dataset_id] = pd.read_csv(source_raw, low_memory=False)
            radius_cache[dataset_id] = pd.read_csv(source_radii, low_memory=False)
            require_columns(raw_cache[dataset_id], REQUIRED_RAW_COLUMNS, source_raw)
            require_columns(radius_cache[dataset_id], REQUIRED_RADIUS_COLUMNS, source_radii)
        raw = raw_cache[dataset_id]
        match = raw[
            (pd.to_numeric(raw["mri_i"], errors="coerce") == int(request.mri_i))
            & (pd.to_numeric(raw["mri_j"], errors="coerce") == int(request.mri_j))
            & (pd.to_numeric(raw["mri_k"], errors="coerce") == int(request.mri_k))
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one match for {dataset_id} ({request.mri_i},{request.mri_j},{request.mri_k}), found {len(match)}"
            )
        row = match.iloc[0]
        request_id = f"{request.gruppe}_{request.probe}_point{int(request.punkt)}"
        key = (dataset_id, int(row["mri_table_row"]))
        duplicate_of = first_for_voxel.get(key, "")
        first_for_voxel.setdefault(key, request_id)
        radii = radius_cache[dataset_id]
        selected_radii = radii[radii["mri_table_row"] == row["mri_table_row"]].copy()
        selected_radii.insert(0, "request_id", request_id)
        selected_radii.insert(1, "duplicate_of", duplicate_of)
        selected_radii.insert(2, "request_group", request.gruppe)
        selected_radii.insert(3, "request_point", int(request.punkt))
        long_radii.append(selected_radii)
        summary = summarize_radii(selected_radii["cell_radius_um"])
        record = {
            "request_id": request_id,
            "group": request.gruppe,
            "sample": request.probe,
            "point": int(request.punkt),
            "dataset_id": dataset_id,
            "mri_i": int(request.mri_i),
            "mri_j": int(request.mri_j),
            "mri_k": int(request.mri_k),
            "mri_table_row": int(row["mri_table_row"]),
            "duplicate_of": duplicate_of,
            "R1": row.get("R1", np.nan),
            "R2s": row.get("R2s", np.nan),
            "MD": row.get("MD", np.nan),
            "r23_um_inv": row.get(R23, np.nan),
            "coverage_percent": row.get("coverage_percent", np.nan),
            "coord_usable": row.get("coord_usable", np.nan),
            "corners_usable": row.get("corners_usable", np.nan),
            "included_in_base_analysis": row.get("included_in_base_analysis", np.nan),
            "analysis_tomogram": row.get("analysis_tomogram", ""),
            **summary.to_dict(),
        }
        if int(pd.to_numeric(pd.Series([record["included_in_base_analysis"]]), errors="coerce").fillna(0).iloc[0]) != 1:
            warnings_out.append(
                f"{request_id} maps to a valid footprint with {len(selected_radii)} cells but was excluded from primary correlations (coord_usable={record['coord_usable']})."
            )
        if duplicate_of:
            warnings_out.append(f"{request_id} duplicates MRI voxel {duplicate_of}.")
        match_records.append(record)

    matches = pd.DataFrame(match_records)
    matches.to_csv(output_dir / "task2_coordinate_matches_and_qc.csv", index=False)
    pd.concat(long_radii, ignore_index=True).to_csv(
        output_dir / "task2_requested_per_cell_radii.csv.gz", index=False, compression="gzip"
    )

    all_requested_radii = pd.concat(long_radii, ignore_index=True)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    group_order = ["Laesion", "Kontrolle"]
    group_colors = {"Laesion": "#D1495B", "Kontrolle": "#277DA1"}
    for grid_row, group in enumerate(group_order):
        subset = matches[matches["group"] == group].sort_values("point")
        if len(subset) != 3:
            raise RuntimeError(f"Expected three requests for {group}, found {len(subset)}")
        for grid_column, (_, row) in enumerate(subset.iterrows()):
            values = all_requested_radii.loc[
                all_requested_radii["request_id"] == row["request_id"], "cell_radius_um"
            ].to_numpy(float)
            duplicate = f"; same MRI voxel as {row['duplicate_of']}" if row["duplicate_of"] else ""
            qc = "primary-QC included" if int(row["included_in_base_analysis"]) == 1 else "exploratory: primary-QC excluded"
            title = (
                f"{group} point {int(row['point'])}: MRI ({int(row['mri_i'])},{int(row['mri_j'])},{int(row['mri_k'])}){duplicate}\n"
                f"R1={format_value(row['R1'])}, R2*={format_value(row['R2s'])}, "
                f"MD={format_value(row['MD'], scientific=True)}, r23={format_value(row['r23_um_inv'])} 1/um\n"
                f"n={len(values)}, coverage={format_value(row['coverage_percent'])}%, {qc}"
            )
            distribution_panel(
                axes[grid_row, grid_column], values, title, group_colors[group], radius_max
            )
    figure.suptitle(
        "Hamburg ex vivo MPM: requested lesion and control soma-radius distributions",
        fontsize=15,
    )
    save_figure(figure, output_dir / "task2_HH_exvivo_requested_six_coordinate_distributions")
    return matches, warnings_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--coordinate-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--examples-per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--radius-max-um", type=float, default=11.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task1_dir = args.output_dir / "task1_sulcus_region_distributions"
    task2_dir = args.output_dir / "task2_requested_coordinates"
    selected, line_records = task_one(
        args.source_run_root,
        task1_dir,
        args.min_cells,
        args.examples_per_category,
        args.seed,
        args.radius_max_um,
    )
    matches, warnings_out = task_two(
        args.source_run_root,
        args.coordinate_table,
        task2_dir,
        args.radius_max_um,
    )

    report = {
        "schema_version": 1,
        "source_full_run": str(args.source_run_root.resolve()),
        "coordinate_table": str(args.coordinate_table.resolve()),
        "random_seed": args.seed,
        "examples_per_low_middle_high_category": args.examples_per_category,
        "selection_percentile_bands": {
            key: {"target": values[0], "lower": values[1], "upper": values[2]}
            for key, values in QUANTILE_BANDS.items()
        },
        "radius_plot_range_um": [0.0, args.radius_max_um],
        "task1_selected_voxels": int(len(selected)),
        "task1_line_replication": line_records,
        "task2_requested_points": int(len(matches)),
        "task2_unique_mri_voxels": int(
            matches[["dataset_id", "mri_table_row"]].drop_duplicates().shape[0]
        ),
        "warnings": warnings_out,
        "cell_assignment_rule": "A reconciled cell instance is assigned to the MRI voxel containing its centroid.",
    }
    (args.output_dir / "detailed_collaborator_request_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README_RESULTS.txt").write_text(
        "Detailed collaborator voxel plots\n\n"
        f"Source full run: {args.source_run_root}\n"
        "task1_sulcus_region_distributions/: four low/middle/high distribution grids and two line-replication plots\n"
        "task2_requested_coordinates/: six requested coordinate panels, matching/QC table, and per-cell radii\n"
        "See detailed_collaborator_request_report.json for reproducibility settings and warnings.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
