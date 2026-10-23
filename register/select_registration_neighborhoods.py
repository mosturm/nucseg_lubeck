#!/usr/bin/env python3
"""Select compact MRI neighborhoods for transparent method illustrations."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


TARGET = "cell_density_per_mm3"
INDEX_COLUMNS = ("mri_i", "mri_j", "mri_k")
PLANES = ((0, 1), (0, 2), (1, 2))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def normalize_tomogram(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{int(number):04d}" if np.isfinite(number) else ""


def cube_faces(origin: np.ndarray) -> list[list[np.ndarray]]:
    x, y, z = origin
    vertices = np.array(
        [
            [x, y, z], [x + 1, y, z], [x, y + 1, z], [x + 1, y + 1, z],
            [x, y, z + 1], [x + 1, y, z + 1], [x, y + 1, z + 1],
            [x + 1, y + 1, z + 1],
        ]
    )
    return [
        vertices[[0, 1, 3, 2]], vertices[[4, 5, 7, 6]],
        vertices[[0, 1, 5, 4]], vertices[[2, 3, 7, 6]],
        vertices[[0, 2, 6, 4]], vertices[[1, 3, 7, 5]],
    ]


def nominal_coordinates(frame: pd.DataFrame) -> np.ndarray:
    coordinates = frame[list(INDEX_COLUMNS)].to_numpy(float)
    return coordinates - coordinates.min(axis=0)


def finish_nominal_axis(axis, coordinates: np.ndarray, title: str) -> None:
    upper = coordinates.max(axis=0) + 1
    lower_limit = np.full(3, -0.3)
    upper_limit = upper + 0.3
    axis.set_xlim(lower_limit[0], upper_limit[0])
    axis.set_ylim(lower_limit[1], upper_limit[1])
    axis.set_zlim(lower_limit[2], upper_limit[2])
    axis.set_box_aspect(upper_limit - lower_limit)
    axis.set_proj_type("ortho")
    axis.view_init(elev=24, azim=-55)
    axis.set_axis_off()
    axis.set_facecolor("black")
    axis.set_title(title, color="white", fontsize=10, pad=2)


def draw_voxel_edges(axis, coordinate: np.ndarray, color="#EF5A5A", alpha=0.62) -> None:
    for face in cube_faces(coordinate):
        ring = np.vstack([face, face[0]])
        axis.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=0.65, alpha=alpha)


def draw_wall(axis, frame: pd.DataFrame, value: str, cmap_name: str, title: str):
    coords = nominal_coordinates(frame)
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max()))
    if math.isclose(low, high):
        high = low + 1.0
    norm = colors.Normalize(low, high)
    cmap = matplotlib.colormaps[cmap_name]
    for row, (coord, number) in enumerate(zip(coords, values), start=1):
        faces = cube_faces(coord)
        collection = Poly3DCollection(
            faces,
            facecolors=[cmap(norm(number))] * 6,
            edgecolors="#BFBFBF" if cmap_name == "gray" else "#161616",
            linewidths=0.8,
            alpha=0.96,
        )
        axis.add_collection3d(collection)
        axis.text(*(coord + 0.5), str(row), color="white", ha="center", va="center", fontsize=7)
    finish_nominal_axis(axis, coords, title)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def draw_simulated_xpct(axis, frame: pd.DataFrame, seed: int) -> None:
    """Preview the intended visual grammar without claiming to show real XPCT."""
    rng = np.random.default_rng(seed)
    coordinates = nominal_coordinates(frame)
    for voxel_index, coordinate in enumerate(coordinates):
        base_shade = 0.18 + 0.09 * ((voxel_index * 7) % 5)
        axis.add_collection3d(
            Poly3DCollection(
                cube_faces(coordinate),
                facecolors=[(base_shade, base_shade, base_shade, 0.18)] * 6,
                edgecolors="none",
            )
        )
        points = coordinate + rng.random((180, 3))
        texture = (
            0.48
            + 0.23 * np.sin(10.0 * points[:, 0] + voxel_index)
            + 0.17 * np.cos(13.0 * points[:, 2])
            + rng.normal(0.0, 0.10, len(points))
        )
        keep = texture > np.quantile(texture, 0.38)
        shade = np.clip(texture[keep], 0.12, 0.96)
        axis.scatter(
            points[keep, 0], points[keep, 1], points[keep, 2],
            c=shade, cmap="gray", vmin=0, vmax=1,
            s=2.6, alpha=0.32, linewidths=0, depthshade=False,
        )
        draw_voxel_edges(axis, coordinate)
    finish_nominal_axis(axis, coordinates, "Simulated XPCT layout")


def draw_simulated_cells(axis, frame: pd.DataFrame, seed: int) -> None:
    """Preview mask clutter using real counts but simulated within-voxel positions."""
    rng = np.random.default_rng(seed)
    coordinates = nominal_coordinates(frame)
    counts = pd.to_numeric(frame.get("cell_count", 0), errors="coerce").fillna(0).to_numpy(float)
    positive = counts[counts > 0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    color_index = 0
    palette = matplotlib.colormaps["turbo"]
    for coordinate, count in zip(coordinates, counts):
        shown = 0 if count <= 0 else int(np.clip(round(10.0 * math.sqrt(count / reference)), 3, 24))
        centers = coordinate + 0.08 + 0.84 * rng.random((shown, 3))
        for center in centers:
            color = palette((color_index * 0.61803398875) % 1.0)
            axis.scatter(
                center[0], center[1], center[2], s=rng.uniform(8, 24),
                color=[color], alpha=0.82, edgecolors="none", depthshade=False,
            )
            color_index += 1
        draw_voxel_edges(axis, coordinate, alpha=0.50)
    finish_nominal_axis(axis, coordinates, "Simulated cell-mask layout")


def eligible_frame(path: Path, predictor: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        *INDEX_COLUMNS,
        "mri_table_row",
        "analysis_tomogram",
        "coverage_percent",
        "included_in_base_analysis",
        TARGET,
        predictor,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    numeric = [*INDEX_COLUMNS, "mri_table_row", "coverage_percent", TARGET, predictor]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["analysis_tomogram"] = frame["analysis_tomogram"].map(normalize_tomogram)
    keep = frame["included_in_base_analysis"].eq(1)
    keep &= frame["coverage_percent"].ge(80)
    keep &= frame[[*INDEX_COLUMNS, TARGET, predictor]].notna().all(axis=1)
    keep &= frame["analysis_tomogram"].ne("")
    return frame.loc[keep].copy()


def rectangles(frame: pd.DataFrame, predictor: str, association: pd.Series):
    values = frame[[predictor, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    predictor_iqr = max(float(values[predictor].quantile(0.75) - values[predictor].quantile(0.25)), 1e-12)
    density_iqr = max(float(values[TARGET].quantile(0.75) - values[TARGET].quantile(0.25)), 1e-12)
    global_rho = float(association["spearman_rho"])
    heldout = float(association["heldout_r2"]) if np.isfinite(association["heldout_r2"]) else -1.0
    association_score = abs(global_rho)
    for tomo, tomo_frame in frame.groupby("analysis_tomogram"):
        tomo_frame = tomo_frame.reset_index(drop=True)
        coordinate_array = tomo_frame[list(INDEX_COLUMNS)].to_numpy(np.int64)
        predictor_values = tomo_frame[predictor].to_numpy(float)
        density_values = tomo_frame[TARGET].to_numpy(float)
        coverage_values = tomo_frame["coverage_percent"].to_numpy(float)
        table_rows = tomo_frame["mri_table_row"].to_numpy(np.int64)
        by_coordinate = {
            tuple(coordinate): position
            for position, coordinate in enumerate(coordinate_array)
        }
        coordinates = sorted(by_coordinate)
        for axis_a, axis_b in PLANES:
            fixed_axis = ({0, 1, 2} - {axis_a, axis_b}).pop()
            for height, width in ((2, 3), (3, 2)):
                for start in coordinates:
                    wanted = []
                    for da in range(height):
                        for db in range(width):
                            coordinate = list(start)
                            coordinate[axis_a] += da
                            coordinate[axis_b] += db
                            wanted.append(tuple(coordinate))
                    if any(coordinate not in by_coordinate for coordinate in wanted):
                        continue
                    if any(coordinate[fixed_axis] != start[fixed_axis] for coordinate in wanted):
                        continue
                    indices = np.asarray([by_coordinate[coordinate] for coordinate in wanted])
                    local_x = predictor_values[indices]
                    local_y = density_values[indices]
                    if np.ptp(local_x) == 0 or np.ptp(local_y) == 0:
                        continue
                    local = np.corrcoef(stats.rankdata(local_x), stats.rankdata(local_y))[0, 1]
                    if not np.isfinite(local) or abs(local) < 0.60 or local * global_rho <= 0:
                        continue
                    predictor_range = float(np.ptp(local_x)) / predictor_iqr
                    density_range = float(np.ptp(local_y)) / density_iqr
                    range_score = min(predictor_range, 3.0) + min(density_range, 3.0)
                    coverage_score = float(coverage_values[indices].mean()) / 100.0
                    score = association_score + abs(float(local)) + 0.12 * range_score + 0.10 * coverage_score
                    yield {
                        "dataset_id": association["scope_id"],
                        "acquisition": association["acquisition"],
                        "specimen": association["scope_id"].split("__")[1],
                        "protocol": association["protocol"],
                        "predictor": predictor,
                        "target": TARGET,
                        "analysis_tomogram": tomo,
                        "plane": "ijk"[axis_a] + "ijk"[axis_b],
                        "wall_shape": f"{height}x{width}",
                        "global_spearman_rho": global_rho,
                        "global_heldout_r2": heldout,
                        "global_fdr_q": float(association["spearman_p_fdr_bh"]),
                        "local_spearman_rho": float(local),
                        "predictor_iqr_ranges": predictor_range,
                        "density_iqr_ranges": density_range,
                        "mean_coverage_percent": float(coverage_values[indices].mean()),
                        "selection_score": score,
                        "mri_table_rows": tuple(int(value) for value in table_rows[indices]),
                        "coordinates": tuple(wanted),
                    }


def compact_planar_neighborhoods(frame: pd.DataFrame, predictor: str, association: pd.Series):
    values = frame[[predictor, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    predictor_iqr = max(float(values[predictor].quantile(0.75) - values[predictor].quantile(0.25)), 1e-12)
    density_iqr = max(float(values[TARGET].quantile(0.75) - values[TARGET].quantile(0.25)), 1e-12)
    global_rho = float(association["spearman_rho"])
    heldout = float(association["heldout_r2"]) if np.isfinite(association["heldout_r2"]) else -1.0
    association_score = abs(global_rho)

    for tomo, tomo_frame in frame.groupby("analysis_tomogram"):
        tomo_frame = tomo_frame.reset_index(drop=True)
        all_coordinates = tomo_frame[list(INDEX_COLUMNS)].to_numpy(np.int64)
        predictor_values = tomo_frame[predictor].to_numpy(float)
        density_values = tomo_frame[TARGET].to_numpy(float)
        coverage_values = tomo_frame["coverage_percent"].to_numpy(float)
        table_rows = tomo_frame["mri_table_row"].to_numpy(np.int64)
        seen: set[tuple[int, ...]] = set()
        for axis_a, axis_b in PLANES:
            fixed_axis = ({0, 1, 2} - {axis_a, axis_b}).pop()
            for _, positions in tomo_frame.groupby(INDEX_COLUMNS[fixed_axis]).indices.items():
                positions = np.asarray(positions, dtype=np.int64)
                if len(positions) < 6:
                    continue
                plane_coordinates = all_coordinates[positions][:, [axis_a, axis_b]]
                tree = cKDTree(plane_coordinates)
                _, neighbors = tree.query(plane_coordinates, k=6)
                for local_positions in neighbors:
                    indices = positions[np.asarray(local_positions, dtype=np.int64)]
                    key = tuple(sorted(int(value) for value in indices))
                    if key in seen:
                        continue
                    seen.add(key)
                    selected_coordinates = all_coordinates[indices]
                    spans = np.ptp(selected_coordinates[:, [axis_a, axis_b]], axis=0)
                    if spans.max() > 5 or np.prod(spans + 1) > 18:
                        continue
                    local_x = predictor_values[indices]
                    local_y = density_values[indices]
                    if np.ptp(local_x) == 0 or np.ptp(local_y) == 0:
                        continue
                    local = np.corrcoef(stats.rankdata(local_x), stats.rankdata(local_y))[0, 1]
                    if not np.isfinite(local) or abs(local) < 0.60 or local * global_rho <= 0:
                        continue
                    predictor_range = float(np.ptp(local_x)) / predictor_iqr
                    density_range = float(np.ptp(local_y)) / density_iqr
                    range_score = min(predictor_range, 3.0) + min(density_range, 3.0)
                    coverage_score = float(coverage_values[indices].mean()) / 100.0
                    compactness = 1.0 / max(float(np.prod(spans + 1)), 1.0)
                    score = (
                        association_score
                        + abs(float(local))
                        + 0.12 * range_score
                        + 0.10 * coverage_score
                        + 0.10 * compactness
                    )
                    yield {
                        "dataset_id": association["scope_id"],
                        "acquisition": association["acquisition"],
                        "specimen": association["scope_id"].split("__")[1],
                        "protocol": association["protocol"],
                        "predictor": predictor,
                        "target": TARGET,
                        "analysis_tomogram": tomo,
                        "plane": "ijk"[axis_a] + "ijk"[axis_b],
                        "wall_shape": "compact6",
                        "global_spearman_rho": global_rho,
                        "global_heldout_r2": heldout,
                        "global_fdr_q": float(association["spearman_p_fdr_bh"]),
                        "local_spearman_rho": float(local),
                        "predictor_iqr_ranges": predictor_range,
                        "density_iqr_ranges": density_range,
                        "mean_coverage_percent": float(coverage_values[indices].mean()),
                        "selection_score": score,
                        "mri_table_rows": tuple(int(value) for value in table_rows[indices]),
                        "coordinates": tuple(tuple(int(v) for v in row) for row in selected_coordinates),
                    }


def compact_3d_neighborhoods(frame: pd.DataFrame, predictor: str, association: pd.Series):
    values = frame[[predictor, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    predictor_iqr = max(float(values[predictor].quantile(0.75) - values[predictor].quantile(0.25)), 1e-12)
    density_iqr = max(float(values[TARGET].quantile(0.75) - values[TARGET].quantile(0.25)), 1e-12)
    global_rho = float(association["spearman_rho"])
    heldout = float(association["heldout_r2"]) if np.isfinite(association["heldout_r2"]) else -1.0
    association_score = abs(global_rho)
    for tomo, tomo_frame in frame.groupby("analysis_tomogram"):
        tomo_frame = tomo_frame.reset_index(drop=True)
        if len(tomo_frame) < 6:
            continue
        coordinates = tomo_frame[list(INDEX_COLUMNS)].to_numpy(np.int64)
        predictor_values = tomo_frame[predictor].to_numpy(float)
        density_values = tomo_frame[TARGET].to_numpy(float)
        coverage_values = tomo_frame["coverage_percent"].to_numpy(float)
        table_rows = tomo_frame["mri_table_row"].to_numpy(np.int64)
        tree = cKDTree(coordinates)
        _, neighborhoods = tree.query(coordinates, k=6)
        seen: set[tuple[int, ...]] = set()
        for indices in neighborhoods:
            indices = np.asarray(indices, dtype=np.int64)
            key = tuple(sorted(int(value) for value in indices))
            if key in seen:
                continue
            seen.add(key)
            selected_coordinates = coordinates[indices]
            spans = np.ptp(selected_coordinates, axis=0)
            box_volume = float(np.prod(spans + 1))
            if spans.max() > 5 or box_volume > 64:
                continue
            local_x = predictor_values[indices]
            local_y = density_values[indices]
            if np.ptp(local_x) == 0 or np.ptp(local_y) == 0:
                continue
            local = np.corrcoef(stats.rankdata(local_x), stats.rankdata(local_y))[0, 1]
            if not np.isfinite(local) or abs(local) < 0.60 or local * global_rho <= 0:
                continue
            predictor_range = float(np.ptp(local_x)) / predictor_iqr
            density_range = float(np.ptp(local_y)) / density_iqr
            range_score = min(predictor_range, 3.0) + min(density_range, 3.0)
            coverage_score = float(coverage_values[indices].mean()) / 100.0
            compactness = 1.0 / max(box_volume, 1.0)
            score = (
                association_score
                + abs(float(local))
                + 0.12 * range_score
                + 0.10 * coverage_score
                + 0.08 * compactness
                - 0.04
            )
            yield {
                "dataset_id": association["scope_id"],
                "acquisition": association["acquisition"],
                "specimen": association["scope_id"].split("__")[1],
                "protocol": association["protocol"],
                "predictor": predictor,
                "target": TARGET,
                "analysis_tomogram": tomo,
                "plane": "3d",
                "wall_shape": "compact6_3d",
                "global_spearman_rho": global_rho,
                "global_heldout_r2": heldout,
                "global_fdr_q": float(association["spearman_p_fdr_bh"]),
                "local_spearman_rho": float(local),
                "predictor_iqr_ranges": predictor_range,
                "density_iqr_ranges": density_range,
                "mean_coverage_percent": float(coverage_values[indices].mean()),
                "selection_score": score,
                "mri_table_rows": tuple(int(value) for value in table_rows[indices]),
                "coordinates": tuple(tuple(int(v) for v in row) for row in selected_coordinates),
            }


def choose_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    selected: list[dict] = []
    per_dataset: dict[str, int] = {}
    per_specimen: dict[str, int] = {}
    occupied: dict[tuple[str, str], set[tuple[int, int, int]]] = {}
    for candidate in candidates:
        dataset = candidate["dataset_id"]
        specimen = candidate["specimen"]
        if per_dataset.get(dataset, 0) >= 3 or per_specimen.get(specimen, 0) >= 5:
            continue
        key = (dataset, candidate["analysis_tomogram"])
        coordinates = set(candidate["coordinates"])
        if coordinates & occupied.get(key, set()):
            continue
        selected.append(candidate)
        per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
        per_specimen[specimen] = per_specimen.get(specimen, 0) + 1
        occupied.setdefault(key, set()).update(coordinates)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        raise RuntimeError(f"Only {len(selected)} non-overlapping neighborhoods passed the fixed criteria")
    return selected


def write_previews(selected: list[dict], raw_dir: Path, output_dir: Path):
    preview_dir = output_dir / "selection_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    contact, contact_axes = plt.subplots(
        len(selected), 5, figsize=(21, 3.6 * len(selected)),
        subplot_kw={"projection": "3d"}, facecolor="black",
    )
    if len(selected) == 1:
        contact_axes = np.asarray([contact_axes])
    for rank, candidate in enumerate(selected, start=1):
        raw_path = raw_dir / f"{candidate['dataset_id']}_raw_voxel_values.csv.gz"
        full = eligible_frame(raw_path, candidate["predictor"])
        rows = set(candidate["mri_table_rows"])
        neighborhood = full[full["mri_table_row"].isin(rows)].sort_values(list(INDEX_COLUMNS))
        candidate["raw_table"] = str(raw_path)
        candidate["mri_table_rows_json"] = json.dumps(list(candidate["mri_table_rows"]))
        candidate["coordinates_json"] = json.dumps(
            [[int(component) for component in value] for value in candidate["coordinates"]]
        )

        for axis in contact_axes[rank - 1]:
            axis.set_facecolor("black")
        draw_wall(contact_axes[rank - 1, 0], neighborhood, candidate["predictor"], "gray", f"{rank}. MRI: {candidate['predictor']}")
        draw_simulated_xpct(contact_axes[rank - 1, 1], neighborhood, seed=4600 + rank)
        draw_simulated_cells(contact_axes[rank - 1, 2], neighborhood, seed=8600 + rank)
        draw_wall(contact_axes[rank - 1, 3], neighborhood, TARGET, "magma", "Cell density")
        scatter_axis = contact_axes[rank - 1, 4]
        scatter_axis.remove()
        scatter_axis = contact.add_subplot(len(selected), 5, 5 * rank)
        x = full[TARGET].to_numpy(float)
        y = full[candidate["predictor"]].to_numpy(float)
        scatter_axis.scatter(x, y, s=8, color="#8D8D8D", alpha=0.35, linewidths=0)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        scatter_axis.plot(line_x, intercept + slope * line_x, color="#E84A5F", linewidth=2)
        scatter_axis.scatter(neighborhood[TARGET], neighborhood[candidate["predictor"]], s=38, color="#FFCB47", edgecolor="black", linewidth=0.4)
        scatter_axis.set_facecolor("black")
        scatter_axis.tick_params(colors="white", labelsize=7)
        for spine in scatter_axis.spines.values():
            spine.set_color("#777777")
        scatter_axis.set_xlabel(r"Cell density (cells mm$^{-3}$)", color="white", fontsize=8)
        scatter_axis.set_ylabel(candidate["predictor"], color="white", fontsize=8)
        scatter_axis.set_title(
            f"{candidate['dataset_id']} | global rho={candidate['global_spearman_rho']:+.2f}\n"
            f"visual subset rho={candidate['local_spearman_rho']:+.2f}",
            color="white", fontsize=8,
        )
    contact.suptitle(
        "Registration layout preview: simulated XPCT/cell placement; real values and correlations",
        color="white", fontsize=15,
    )
    contact.tight_layout(rect=(0, 0, 1, 0.99))
    contact.savefig(output_dir / "selected_neighborhoods_overview.png", dpi=180, facecolor="black", bbox_inches="tight")
    contact.savefig(output_dir / "selected_neighborhoods_overview.pdf", facecolor="black", bbox_inches="tight")
    plt.close(contact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--allow-exploratory", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.analysis_dir / "all_univariate_correlations_and_grouped_cv.csv"
    raw_dir = args.analysis_dir / "raw_voxel_tables"
    results = pd.read_csv(results_path)
    associations = results[(results["scope"] == "dataset") & (results["target"] == TARGET)].copy()
    if not args.allow_exploratory:
        associations = associations[associations["primary_analysis"] == 1]
    associations = associations[
        associations["n"].ge(30)
        & associations["spearman_rho"].abs().ge(0.35)
        & associations["spearman_p_fdr_bh"].lt(0.05)
    ].copy()
    associations["absolute_spearman_rho"] = associations["spearman_rho"].abs()
    associations = associations.sort_values(
        ["absolute_spearman_rho", "spearman_p_fdr_bh"], ascending=[False, True]
    ).head(30)

    all_candidates: list[dict] = []
    errors = []
    for _, association in associations.iterrows():
        path = raw_dir / f"{association['scope_id']}_raw_voxel_values.csv.gz"
        if not path.is_file():
            errors.append(f"Missing {path}")
            continue
        try:
            frame = eligible_frame(path, association["predictor"])
            all_candidates.extend(rectangles(frame, association["predictor"], association))
            all_candidates.extend(compact_planar_neighborhoods(frame, association["predictor"], association))
            all_candidates.extend(compact_3d_neighborhoods(frame, association["predictor"], association))
        except Exception as error:
            errors.append(f"{association['scope_id']} / {association['predictor']}: {error}")

    inventory = (
        pd.DataFrame(all_candidates)
        .groupby(["dataset_id", "predictor"], as_index=False)
        .agg(candidate_count=("selection_score", "size"), best_selection_score=("selection_score", "max"))
        .sort_values("best_selection_score", ascending=False)
    )
    inventory.to_csv(args.output_dir / "candidate_inventory.csv", index=False)
    print(inventory.head(20).to_string(index=False))
    selected = choose_candidates(all_candidates, args.top_k)
    write_previews(selected, raw_dir, args.output_dir)
    summary_rows = []
    voxel_rows = []
    for rank, candidate in enumerate(selected, start=1):
        summary = {key: value for key, value in candidate.items() if key not in {"mri_table_rows", "coordinates"}}
        summary["rank"] = rank
        summary_rows.append(summary)
        raw = eligible_frame(raw_dir / f"{candidate['dataset_id']}_raw_voxel_values.csv.gz", candidate["predictor"])
        selected_rows = raw[raw["mri_table_row"].isin(candidate["mri_table_rows"])].copy()
        selected_rows.insert(0, "rank", rank)
        selected_rows.insert(1, "neighborhood_voxel", np.arange(1, len(selected_rows) + 1))
        selected_rows.insert(2, "selected_predictor", candidate["predictor"])
        voxel_rows.append(selected_rows)
    pd.DataFrame(summary_rows).sort_values("rank").to_csv(args.output_dir / "selected_neighborhoods.csv", index=False)
    pd.concat(voxel_rows, ignore_index=True).to_csv(args.output_dir / "selected_neighborhood_voxels.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"message": errors}).to_csv(args.output_dir / "selection_warnings.csv", index=False)
    print(pd.DataFrame(summary_rows)[["rank", "dataset_id", "predictor", "analysis_tomogram", "plane", "wall_shape", "global_spearman_rho", "global_heldout_r2", "local_spearman_rho", "selection_score"]].to_string(index=False))
    print(f"Wrote selection and previews to {args.output_dir}")


if __name__ == "__main__":
    main()
