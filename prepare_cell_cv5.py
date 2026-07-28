from __future__ import annotations

import csv
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile as tiff

from prepare_cell_only_dataset import read_nrrd


SEED = 46
N_FOLDS = 5
SPLITS = ("Trainingsdaten", "Valdaten", "Testdaten")
EXCLUDED_SAMPLE_IDS = {"tomo_reco_id0004_t0007vn"}

CELL_ROOT = Path("data_cell_only")
SOURCE_SPLITS = tuple(CELL_ROOT / split for split in SPLITS)
OUTPUT_ROOT = CELL_ROOT / "cv5_seed46"


def ensure_zyx(arr: np.ndarray) -> np.ndarray:
    array = np.asarray(arr)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {array.shape}")
    z_axis = int(np.argmin(array.shape))
    return np.moveaxis(array, z_axis, 0) if z_axis != 0 else array


def discover_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for source_split in SOURCE_SPLITS:
        if not source_split.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_split}")
        for image_path in sorted(source_split.glob("*_img.tif")):
            sample_id = image_path.name[: -len("_img.tif")]
            label_path = source_split / f"{sample_id}_label.seg.nrrd"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate cell sample ID: {sample_id}")
            seen_ids.add(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "label_path": label_path,
                    "original_split": source_split.name,
                }
            )

    if len(samples) != 13:
        raise ValueError(f"Expected 13 prepared cell samples, found {len(samples)}")
    missing = EXCLUDED_SAMPLE_IDS - seen_ids
    if missing:
        raise ValueError(f"Excluded samples are absent from cell data: {missing}")
    return [
        sample for sample in samples if sample["sample_id"] not in EXCLUDED_SAMPLE_IDS
    ]


def validate_sources(samples: list[dict[str, object]]) -> None:
    for sample in samples:
        sample_id = str(sample["sample_id"])
        image = ensure_zyx(tiff.imread(Path(sample["image_path"])))
        raw_mask = read_nrrd(Path(sample["label_path"]))[0]
        mask = ensure_zyx(np.transpose(raw_mask, (1, 0, 2)))
        if image.shape != mask.shape:
            raise ValueError(f"{sample_id}: shape mismatch {image.shape} vs {mask.shape}")
        labels = set(np.unique(mask).tolist())
        if not labels.issubset({0, 1}) or 1 not in labels:
            raise ValueError(f"{sample_id}: expected binary 0/1 mask, got {labels}")
        sample["loaded_shape_zyx"] = "x".join(str(size) for size in image.shape)
        sample["foreground_voxels"] = int(np.count_nonzero(mask == 1))


def make_test_groups(samples: list[dict[str, object]]) -> list[list[str]]:
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    random.Random(SEED).shuffle(sample_ids)
    return [sample_ids[index::N_FOLDS] for index in range(N_FOLDS)]


def build_assignments(
    samples: list[dict[str, object]],
) -> tuple[list[dict[str, str]], list[list[str]]]:
    test_groups = make_test_groups(samples)
    assignments_by_fold: list[dict[str, str]] = []

    for fold_index in range(N_FOLDS):
        test_ids = set(test_groups[fold_index])
        val_ids = set(test_groups[(fold_index + 1) % N_FOLDS])
        if test_ids & val_ids:
            raise AssertionError(f"Fold {fold_index + 1}: validation overlaps test")
        assignments_by_fold.append(
            {
                str(sample["sample_id"]): (
                    "Testdaten"
                    if sample["sample_id"] in test_ids
                    else "Valdaten"
                    if sample["sample_id"] in val_ids
                    else "Trainingsdaten"
                )
                for sample in samples
            }
        )
    return assignments_by_fold, test_groups


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_output(
    samples: list[dict[str, object]],
    assignments_by_fold: list[dict[str, str]],
) -> None:
    expected_ids = {str(sample["sample_id"]) for sample in samples}
    test_coverage: Counter[str] = Counter()

    for fold_index, assignments in enumerate(assignments_by_fold, start=1):
        if set(assignments) != expected_ids:
            raise AssertionError(f"Fold {fold_index}: incomplete sample coverage")
        split_sets = {
            split: {sample_id for sample_id, value in assignments.items() if value == split}
            for split in SPLITS
        }
        if any(split_sets[a] & split_sets[b] for a in SPLITS for b in SPLITS if a < b):
            raise AssertionError(f"Fold {fold_index}: split overlap")

        for sample_id, split in assignments.items():
            if split == "Testdaten":
                test_coverage[sample_id] += 1
            split_dir = OUTPUT_ROOT / f"fold_{fold_index:02d}" / "combined" / split
            if not (split_dir / f"{sample_id}_img.tif").is_file():
                raise FileNotFoundError(f"Missing output image for {sample_id}")
            if not (split_dir / f"{sample_id}_label.seg.nrrd").is_file():
                raise FileNotFoundError(f"Missing output label for {sample_id}")

    if any(test_coverage[sample_id] != 1 for sample_id in expected_ids):
        raise AssertionError("Every accepted cell sample must be tested exactly once")


def main() -> None:
    samples = discover_samples()
    validate_sources(samples)
    assignments_by_fold, test_groups = build_assignments(samples)

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}. Remove it explicitly before regenerating."
        )
    OUTPUT_ROOT.mkdir(parents=True)

    all_rows: list[dict[str, object]] = []
    methods: Counter[str] = Counter()
    for fold_index, assignments in enumerate(assignments_by_fold, start=1):
        fold_rows: list[dict[str, object]] = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            split = assignments[sample_id]
            destination = OUTPUT_ROOT / f"fold_{fold_index:02d}" / "combined" / split
            methods.update(
                [
                    link_or_copy(
                        Path(sample["image_path"]), destination / f"{sample_id}_img.tif"
                    ),
                    link_or_copy(
                        Path(sample["label_path"]),
                        destination / f"{sample_id}_label.seg.nrrd",
                    ),
                ]
            )
            row = {
                "fold": fold_index,
                "sample_id": sample_id,
                "split": split,
                "loaded_shape_zyx": sample["loaded_shape_zyx"],
                "foreground_voxels": sample["foreground_voxels"],
                "original_split": sample["original_split"],
            }
            fold_rows.append(row)
            all_rows.append(row)
        fold_root = OUTPUT_ROOT / f"fold_{fold_index:02d}"
        write_csv(fold_root / "metadata.csv", fold_rows, list(fold_rows[0]))

    write_csv(OUTPUT_ROOT / "metadata.csv", all_rows, list(all_rows[0]))
    test_rows = [
        {"fold": fold_index, "sample_id": sample_id}
        for fold_index, group in enumerate(test_groups, start=1)
        for sample_id in group
    ]
    write_csv(OUTPUT_ROOT / "test_coverage.csv", test_rows, ["fold", "sample_id"])
    validate_output(samples, assignments_by_fold)

    counts = Counter(len(group) for group in test_groups)
    readme = [
        "Cell 5-fold outer cross-validation dataset",
        f"seed: {SEED}",
        f"samples: {len(samples)}",
        f"excluded samples: {', '.join(sorted(EXCLUDED_SAMPLE_IDS))}",
        f"test group sizes: {', '.join(str(len(group)) for group in test_groups)}",
        "validation rule: the next outer fold is used for validation",
        "test rule: every accepted sample appears in Testdaten exactly once",
        "remaining samples: Trainingsdaten",
        "prepared labels: binary 0/1 with cell foreground=1",
        "orientation check: current loaders yield matching ZYX image/mask shapes",
        f"test-size frequencies: {dict(counts)}",
        f"materialization methods: {dict(methods)}",
    ]
    (OUTPUT_ROOT / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"Created {OUTPUT_ROOT.resolve()}")
    print(f"Materialization methods: {dict(methods)}")
    for fold_index, assignments in enumerate(assignments_by_fold, start=1):
        split_counts = Counter(assignments.values())
        print(
            f"fold_{fold_index:02d}: train={split_counts['Trainingsdaten']} "
            f"val={split_counts['Valdaten']} test={split_counts['Testdaten']} | "
            f"test={', '.join(test_groups[fold_index - 1])}"
        )


if __name__ == "__main__":
    main()
