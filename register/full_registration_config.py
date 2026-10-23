#!/usr/bin/env python3
"""Shared manifest and naming helpers for the full MRI-XPCT analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SPECIMENS = {
    "P2": {
        "sample_id": "0014",
        "tomograms": ("0001", "0008", "0015", "0022", "0029", "0036"),
    },
    "P5": {"sample_id": "0004", "tomograms": ("0022", "0029")},
    "P7": {
        "sample_id": "0011",
        "tomograms": ("0001", "0002", "0003", "0004"),
    },
}

PRIMARY_EXCLUDED_ACQUISITIONS = {"Portugal_DKI"}
MRI_INDEX_COLUMNS = ("mri_i", "mri_j", "mri_k")


@dataclass(frozen=True)
class TableSpec:
    path: Path
    acquisition: str
    specimen: str
    protocol: str
    dataset_id: str
    predictors: tuple[str, ...]

    @property
    def pooled_protocol_id(self) -> str:
        return f"{self.acquisition}__{self.protocol}"


def predictor_columns(path: Path) -> tuple[str, ...]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in (*MRI_INDEX_COLUMNS, "xpct_mean") if column not in columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    stop = columns.index("xpct_mean")
    predictors = tuple(columns[len(MRI_INDEX_COLUMNS) : stop])
    if not predictors:
        raise ValueError(f"{path}: no MRI predictor columns found before xpct_mean")
    return predictors


def discover_tables(input_root: Path) -> list[TableSpec]:
    root = input_root / "fuer_Mo" if (input_root / "fuer_Mo").is_dir() else input_root
    specs: list[TableSpec] = []
    pattern = re.compile(r"^(P[257])_(.+)_in_tomos\.csv$")
    for path in sorted(root.rglob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            raise ValueError(f"Unexpected registration-table filename: {path}")
        specimen, protocol = match.groups()
        acquisition = path.parent.name
        predictors = predictor_columns(path)
        specs.append(
            TableSpec(
                path=path,
                acquisition=acquisition,
                specimen=specimen,
                protocol=protocol,
                dataset_id=f"{acquisition}__{specimen}__{protocol}",
                predictors=predictors,
            )
        )
    if not specs:
        raise ValueError(f"No *_in_tomos.csv tables found below {root}")
    return specs


def cell_mask_path(data_root: Path, specimen: str, tomo: str, merged: bool = True) -> Path:
    sample_id = SPECIMENS[specimen]["sample_id"]
    suffix = "_instances_merged" if merged else ""
    return data_root / (
        f"tomo_reco_id{sample_id}_t{tomo}_cells_cpneg2p25{suffix}.ome.zarr"
    )


def vessel_mask_path(data_root: Path, specimen: str, tomo: str) -> Path:
    sample_id = SPECIMENS[specimen]["sample_id"]
    return data_root / f"tomo_reco_id{sample_id}_t{tomo}_vessels_multiscale_union.ome.zarr"


def parse_tomogram_tokens(value: object, allowed: tuple[str, ...]) -> list[str]:
    if pd.isna(value):
        return []
    tokens = re.findall(r"\d+(?:\.0)?", str(value))
    normalized = []
    for token in tokens:
        number = int(float(token))
        tomo = f"{number:04d}"
        if tomo in allowed and tomo not in normalized:
            normalized.append(tomo)
    return normalized


def correction_factor(acquisition: str, predictor: str) -> float:
    """Return the input-table scale factor used during extraction.

    The full-run HH ex-vivo tables already contain the corrected values. An
    earlier workflow applied another factor of 0.5; saved feature tables from
    that run retain the desired values in ``*_uncorrected`` columns and the
    analysis stage restores them automatically.
    """
    return 1.0
