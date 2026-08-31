#!/usr/bin/env python3
"""Conservatively reconcile Cellpose instance IDs across inference core faces."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr first: python -m pip install zarr numcodecs") from exc

    try:
        from zarr.codecs import BloscCodec

        def compressor(level: int):
            return BloscCodec(cname="zstd", clevel=int(level))

    except ImportError:
        from numcodecs import Blosc

        def compressor(level: int):
            return Blosc(cname="zstd", clevel=int(level), shuffle=Blosc.BITSHUFFLE)

    return zarr, compressor


def parse_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("Expected three positive integers in Z,Y,X order")
    return values


def open_array(path: Path, key: str):
    zarr, _ = require_zarr()
    root = zarr.open_group(str(path), mode="r")
    if key not in root:
        raise KeyError(f"Array {key!r} not found in {path}; available: {list(root.keys())}")
    array = root[key]
    if len(array.shape) != 3:
        raise ValueError(f"Expected a 3D ZYX label array, got {array.shape}")
    return root, array


def create_output(path: Path, shape, chunks, level: int):
    zarr, make_compressor = require_zarr()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    codec = make_compressor(level)
    if hasattr(root, "create_array"):
        try:
            array = root.create_array(
                "0", shape=shape, chunks=chunks, dtype="uint32", compressors=codec, overwrite=True
            )
        except TypeError:
            array = root.create_array(
                "0", shape=shape, chunks=chunks, dtype="uint32", compressor=codec, overwrite=True
            )
    else:
        array = root.create_dataset(
            "0", shape=shape, chunks=chunks, dtype="uint32", compressor=codec, overwrite=True
        )
    return root, array


def face_pair(axis: int, boundary: int, shape: tuple[int, int, int], array):
    left_slice = [slice(None), slice(None), slice(None)]
    right_slice = [slice(None), slice(None), slice(None)]
    left_slice[axis] = boundary - 1
    right_slice[axis] = boundary
    left = np.asarray(array[tuple(left_slice)], dtype=np.uint32)
    right = np.asarray(array[tuple(right_slice)], dtype=np.uint32)
    if left.shape != right.shape:
        raise AssertionError(f"Face shape mismatch on axis {axis}, boundary {boundary}")
    return left, right


def count_positive(values: np.ndarray) -> dict[int, int]:
    positive = values[values > 0]
    if positive.size == 0:
        return {}
    labels, counts = np.unique(positive, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def candidate_rows(
    axis: int,
    boundary: int,
    left: np.ndarray,
    right: np.ndarray,
    min_contacts: int,
    min_reciprocal_overlap: float,
) -> list[dict[str, object]]:
    touching = (left > 0) & (right > 0) & (left != right)
    if not touching.any():
        return []

    left_values = left[touching].astype(np.uint64, copy=False)
    right_values = right[touching].astype(np.uint64, copy=False)
    packed = (left_values << np.uint64(32)) | right_values
    packed_pairs, contacts = np.unique(packed, return_counts=True)
    left_footprints = count_positive(left)
    right_footprints = count_positive(right)

    raw = []
    for packed_pair, contact in zip(packed_pairs, contacts):
        left_id = int(packed_pair >> np.uint64(32))
        right_id = int(packed_pair & np.uint64(0xFFFFFFFF))
        contact = int(contact)
        raw.append(
            {
                "axis": "zyx"[axis],
                "axis_index": axis,
                "boundary": boundary,
                "left_label": left_id,
                "right_label": right_id,
                "contact_pixels": contact,
                "left_face_pixels": left_footprints[left_id],
                "right_face_pixels": right_footprints[right_id],
                "left_overlap": contact / left_footprints[left_id],
                "right_overlap": contact / right_footprints[right_id],
            }
        )

    best_for_left: dict[int, tuple[int, int, bool]] = {}
    best_for_right: dict[int, tuple[int, int, bool]] = {}
    for row in raw:
        left_id, right_id, contact = (
            int(row["left_label"]),
            int(row["right_label"]),
            int(row["contact_pixels"]),
        )
        current = best_for_left.get(left_id)
        if current is None or contact > current[1]:
            best_for_left[left_id] = (right_id, contact, False)
        elif contact == current[1] and right_id != current[0]:
            best_for_left[left_id] = (current[0], current[1], True)
        current = best_for_right.get(right_id)
        if current is None or contact > current[1]:
            best_for_right[right_id] = (left_id, contact, False)
        elif contact == current[1] and left_id != current[0]:
            best_for_right[right_id] = (current[0], current[1], True)

    for row in raw:
        left_id, right_id = int(row["left_label"]), int(row["right_label"])
        mutual_best = (
            not best_for_left[left_id][2]
            and not best_for_right[right_id][2]
            and best_for_left[left_id][0] == right_id
            and best_for_right[right_id][0] == left_id
        )
        enough_contact = int(row["contact_pixels"]) >= min_contacts
        enough_overlap = min(float(row["left_overlap"]), float(row["right_overlap"])) >= min_reciprocal_overlap
        row["mutual_best"] = int(mutual_best)
        row["accepted"] = int(mutual_best and enough_contact and enough_overlap)
        if not mutual_best:
            row["decision"] = "reject_not_mutual_best"
        elif not enough_contact:
            row["decision"] = "reject_too_few_contacts"
        elif not enough_overlap:
            row["decision"] = "reject_low_reciprocal_overlap"
        else:
            row["decision"] = "accept"
    return raw


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        low, high = sorted((root_first, root_second))
        self.parent[high] = low


def remap_block(block: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if keys.size == 0:
        return block.astype(np.uint32, copy=False)
    flat = block.ravel()
    positions = np.searchsorted(keys, flat)
    valid = positions < keys.size
    matched = np.zeros(flat.shape, dtype=bool)
    matched[valid] = keys[positions[valid]] == flat[valid]
    if not matched.any():
        return block.astype(np.uint32, copy=False)
    output = flat.copy()
    output[matched] = values[positions[matched]]
    return output.reshape(block.shape).astype(np.uint32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array_key", default="0")
    parser.add_argument("--block_shape", default=None)
    parser.add_argument("--min_contacts", type=int, default=3)
    parser.add_argument("--min_reciprocal_overlap", type=float, default=0.25)
    parser.add_argument("--compressor_level", type=int, default=5)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.min_contacts <= 0:
        raise ValueError("--min_contacts must be positive")
    if not 0 <= args.min_reciprocal_overlap <= 1:
        raise ValueError("--min_reciprocal_overlap must be between 0 and 1")

    input_root, input_array = open_array(args.input, args.array_key)
    shape = tuple(int(v) for v in input_array.shape)
    inference = dict(input_root.attrs.get("cellpose_inference", {}))
    if args.block_shape:
        block_shape = parse_zyx(args.block_shape)
        block_shape_source = "command_line"
    elif "block_shape_zyx" in inference:
        block_shape = tuple(int(v) for v in inference["block_shape_zyx"])
        block_shape_source = "cellpose_inference metadata"
    else:
        raise ValueError("Block shape is absent from metadata; pass --block_shape")

    progress(f"Input: {args.input} / {args.array_key}")
    progress(f"Shape ZYX: {shape}; core block ZYX: {block_shape}")
    all_rows: list[dict[str, object]] = []
    union_find = UnionFind()
    boundaries_scanned = 0
    for axis in range(3):
        for boundary in range(block_shape[axis], shape[axis], block_shape[axis]):
            boundaries_scanned += 1
            progress(f"Scanning {'ZYX'[axis]} face at index {boundary}")
            left, right = face_pair(axis, boundary, shape, input_array)
            rows = candidate_rows(
                axis,
                boundary,
                left,
                right,
                args.min_contacts,
                args.min_reciprocal_overlap,
            )
            for row in rows:
                if int(row["accepted"]):
                    union_find.union(int(row["left_label"]), int(row["right_label"]))
            all_rows.extend(rows)

    fieldnames = [
        "axis",
        "axis_index",
        "boundary",
        "left_label",
        "right_label",
        "contact_pixels",
        "left_face_pixels",
        "right_face_pixels",
        "left_overlap",
        "right_overlap",
        "mutual_best",
        "accepted",
        "decision",
    ]
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    mapping = {
        label: union_find.find(label)
        for label in union_find.parent
        if union_find.find(label) != label
    }
    keys = np.asarray(sorted(mapping), dtype=np.uint32)
    values = np.asarray([mapping[int(key)] for key in keys], dtype=np.uint32)
    merge_groups: dict[int, set[int]] = defaultdict(set)
    for label in union_find.parent:
        root = union_find.find(label)
        merge_groups[root].add(label)
    groups = [members for members in merge_groups.values() if len(members) > 1]

    chunks = tuple(int(v) for v in input_array.chunks)
    output_root, output_array = create_output(
        args.output, shape, chunks, args.compressor_level
    )
    foreground_input = 0
    foreground_output = 0
    changed_voxels = 0
    total_chunks = np.prod([int(np.ceil(shape[i] / chunks[i])) for i in range(3)])
    chunk_index = 0
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                chunk_index += 1
                slices = (
                    slice(z0, min(shape[0], z0 + chunks[0])),
                    slice(y0, min(shape[1], y0 + chunks[1])),
                    slice(x0, min(shape[2], x0 + chunks[2])),
                )
                original = np.asarray(input_array[slices], dtype=np.uint32)
                relabeled = remap_block(original, keys, values)
                input_fg = original > 0
                output_fg = relabeled > 0
                if not np.array_equal(input_fg, output_fg):
                    raise AssertionError(f"Semantic foreground changed in chunk {slices}")
                foreground_input += int(np.count_nonzero(input_fg))
                foreground_output += int(np.count_nonzero(output_fg))
                changed_voxels += int(np.count_nonzero(original != relabeled))
                output_array[slices] = relabeled
                if chunk_index == 1 or chunk_index % 250 == 0 or chunk_index == total_chunks:
                    progress(f"Relabel/write chunk {chunk_index}/{total_chunks}")

    for key, value in dict(input_root.attrs).items():
        output_root.attrs[key] = value
    output_root.attrs["instance_boundary_reconciliation"] = {
        "source": str(args.input.resolve()),
        "block_shape_zyx": list(block_shape),
        "block_shape_source": block_shape_source,
        "method": "mutual-best direct face contact",
        "min_contacts": args.min_contacts,
        "min_reciprocal_overlap": args.min_reciprocal_overlap,
        "boundaries_scanned": boundaries_scanned,
        "candidate_pairs": len(all_rows),
        "accepted_face_pairs": int(sum(int(row["accepted"]) for row in all_rows)),
        "merged_equivalence_groups": len(groups),
        "labels_remapped": len(mapping),
        "semantic_foreground_unchanged": foreground_input == foreground_output,
        "audit_csv": str(args.audit_csv.resolve()),
        "limitation": "Post-hoc face matching cannot reproduce overlap predictions that were not saved. Review threshold sensitivity before publication.",
    }

    summary = {
        **dict(output_root.attrs["instance_boundary_reconciliation"]),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "shape_zyx": list(shape),
        "chunks_zyx": list(chunks),
        "foreground_voxels_input": foreground_input,
        "foreground_voxels_output": foreground_output,
        "changed_label_voxels": changed_voxels,
        "largest_merge_group": max((len(group) for group in groups), default=1),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(f"Corrected labels: {args.output}")
    progress(f"Audit: {args.audit_csv}")
    progress(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
