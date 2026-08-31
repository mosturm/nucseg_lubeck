from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


SCALES = ("unscaled", "mid", "coarse")
BEST_EPOCH_PATTERN = re.compile(
    r"Best test-loss checkpoint:\s*epoch=(\d+),\s*test_loss=([0-9.eE+-]+)"
)


def read_best_epoch(log_path: Path) -> tuple[int, float]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing fine-tuning log: {log_path}")
    matches = BEST_EPOCH_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError(f"Expected one best-epoch record in {log_path}, found {len(matches)}")
    epoch, loss = matches[0]
    return int(epoch), float(loss)


def read_thresholds(path: Path, n_folds: int) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold summary: {path}")
    values = {scale: [] for scale in SCALES}
    seen: set[tuple[int, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            fold = int(row["fold"])
            scale = row["scale"]
            if scale not in values:
                raise ValueError(f"Unknown scale {scale!r} in {path}")
            key = (fold, scale)
            if key in seen:
                raise ValueError(f"Duplicate threshold row for fold={fold}, scale={scale}")
            seen.add(key)
            values[scale].append(float(row["cellprob_threshold"]))
    expected = {(fold, scale) for fold in range(1, n_folds + 1) for scale in SCALES}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"Incomplete threshold table; missing={missing}, extra={extra}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive one fixed vessel checkpoint epoch and threshold per scale"
    )
    parser.add_argument("--cv_run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    cv_root = args.cv_run_dir.resolve()
    if not cv_root.is_dir():
        raise FileNotFoundError(f"CV run directory not found: {cv_root}")
    thresholds = read_thresholds(cv_root / "selected_thresholds.tsv", args.n_folds)

    recipe: dict[str, object] = {
        "schema_version": 1,
        "source_cv_run": str(cv_root),
        "selection_rule": (
            "Per scale: median validation-best completed epoch and median "
            "validation-selected cellprob threshold across five outer folds"
        ),
        "combined_pretraining": {
            "training_epochs": 120,
            "checkpoint": "final",
            "learning_rate": 3e-5,
            "weight_decay": 0.05,
            "batch_size": 1,
        },
        "scale_finetuning": {
            "training_epochs": 80,
            "learning_rate": 1e-5,
            "weight_decay": 0.05,
            "batch_size": 1,
        },
        "inference": {"min_size": 15, "flow3D_smooth": 0.0, "infer_3d": True},
        "scales": {},
    }

    scale_recipes: dict[str, object] = {}
    for scale in SCALES:
        epochs: list[int] = []
        losses: list[float] = []
        for fold in range(1, args.n_folds + 1):
            epoch, loss = read_best_epoch(cv_root / f"fold_{fold:02d}" / scale / "train.log")
            epochs.append(epoch)
            losses.append(loss)

        selected_epoch = int(statistics.median(epochs))
        if selected_epoch not in epochs:
            raise AssertionError(f"Median epoch for {scale} is not an observed fold epoch")
        checkpoint_index = selected_epoch - 1
        for fold in range(1, args.n_folds + 1):
            checkpoint = (
                cv_root
                / f"fold_{fold:02d}"
                / scale
                / "models"
                / f"vessel_{scale}_finetune_final_epoch_{checkpoint_index:04d}"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Selected epoch {selected_epoch} checkpoint missing for {scale}: {checkpoint}"
                )

        selected_threshold = float(statistics.median(thresholds[scale]))
        scale_recipes[scale] = {
            "fold_best_completed_epochs": epochs,
            "fold_best_validation_losses": losses,
            "selected_completed_epoch": selected_epoch,
            "checkpoint_index": checkpoint_index,
            "fold_validation_cellprob_thresholds": thresholds[scale],
            "reporting_cellprob_threshold": selected_threshold,
        }

    recipe["scales"] = scale_recipes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote fixed vessel recipe: {args.output}")
    for scale in SCALES:
        row = scale_recipes[scale]
        print(
            f"{scale}: epoch={row['selected_completed_epoch']}, "
            f"checkpoint_index={row['checkpoint_index']:04d}, "
            f"cellprob={row['reporting_cellprob_threshold']:g}"
        )


if __name__ == "__main__":
    main()
