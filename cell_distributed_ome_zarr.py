#!/usr/bin/env python3
"""Chunk-safe distributed Cellpose instance inference for one OME-Zarr volume."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


UINT32_MAX = int(np.iinfo(np.uint32).max)


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_zyx(value: str, *, allow_zero: bool = False) -> tuple[int, int, int]:
    normalized = str(value).replace("x", ",")
    parts = tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
    minimum = 0 if allow_zero else 1
    if len(parts) != 3 or any(part < minimum for part in parts):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Expected three {qualifier} ZYX integers, got {value!r}")
    return parts


def zarr_tools():
    import zarr

    try:
        from zarr.codecs import BloscCodec

        def compressor(level: int):
            return BloscCodec(cname="zstd", clevel=int(level))

    except ImportError:
        from numcodecs import Blosc

        def compressor(level: int):
            return Blosc(cname="zstd", clevel=int(level), shuffle=Blosc.BITSHUFFLE)

    return zarr, compressor


def open_array(path: Path, mode: str = "r", key: str = "0"):
    zarr, _ = zarr_tools()
    root = zarr.open_group(str(path), mode=mode)
    if key not in root:
        raise KeyError(f"Array key {key!r} not found in {path}")
    array = root[key]
    if len(array.shape) != 3:
        raise ValueError(f"Expected a ZYX array in {path}, got {array.shape}")
    return root, array


def create_array(
    path: Path,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    dtype: str,
    level: int,
):
    zarr, make_compressor = zarr_tools()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    codec = make_compressor(level)
    kwargs = dict(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=True,
        fill_value=0,
    )
    if hasattr(root, "create_array"):
        try:
            array = root.create_array("0", compressors=codec, **kwargs)
        except TypeError:
            array = root.create_array("0", compressor=codec, **kwargs)
    else:
        array = root.create_dataset("0", compressor=codec, **kwargs)
    return root, array


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    for candidate in (path / "models" / "best_model", path / "best_model"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve Cellpose model from {path}")


def block_starts(
    shape: tuple[int, int, int], block: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    return [
        (z, y, x)
        for z in range(0, shape[0], block[0])
        for y in range(0, shape[1], block[1])
        for x in range(0, shape[2], block[2])
    ]


def select_starts(
    starts: list[tuple[int, int, int]],
    shape: tuple[int, int, int],
    block: tuple[int, int, int],
    process_fraction: float,
    max_blocks: int,
    preview_mode: str,
    preview_seed: int,
) -> list[tuple[int, int, int]]:
    if not 0.0 < process_fraction <= 1.0:
        raise ValueError("process_fraction must be > 0 and <= 1")
    if max_blocks < 0:
        raise ValueError("max_blocks must be non-negative")
    if process_fraction >= 1.0 and max_blocks == 0:
        return starts
    count = (
        max(1, int(round(len(starts) * process_fraction)))
        if process_fraction < 1.0
        else len(starts)
    )
    if max_blocks > 0:
        count = min(count, max_blocks)
    count = min(count, len(starts))
    if preview_mode == "first":
        return starts[:count]
    if preview_mode == "random":
        return sorted(random.Random(preview_seed).sample(starts, count))
    if preview_mode == "central":
        center = tuple((shape[i] - block[i]) / 2.0 for i in range(3))
        return sorted(
            starts,
            key=lambda start: sum(
                ((float(start[i]) - center[i]) / max(1.0, float(shape[i]))) ** 2
                for i in range(3)
            ),
        )[:count]
    raise ValueError("preview_mode must be first, random, or central")


def block_geometry(
    shape: tuple[int, int, int],
    start: tuple[int, int, int],
    block: tuple[int, int, int],
    halo: tuple[int, int, int],
):
    stop = tuple(min(shape[i], start[i] + block[i]) for i in range(3))
    read_start = tuple(max(0, start[i] - halo[i]) for i in range(3))
    read_stop = tuple(min(shape[i], stop[i] + halo[i]) for i in range(3))
    read_slices = tuple(slice(read_start[i], read_stop[i]) for i in range(3))
    write_slices = tuple(slice(start[i], stop[i]) for i in range(3))
    local_core = tuple(
        slice(start[i] - read_start[i], stop[i] - read_start[i]) for i in range(3)
    )
    return read_slices, write_slices, local_core


def copy_multiscale_metadata(input_root, output_root) -> None:
    input_multiscales = dict(getattr(input_root, "attrs", {})).get("multiscales")
    if input_multiscales:
        multiscales = json.loads(json.dumps(input_multiscales))
        multiscales[0]["name"] = "cellpose_labels"
        multiscales[0]["datasets"][0]["path"] = "0"
    else:
        multiscales = [
            {
                "version": "0.4",
                "name": "cellpose_labels",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]
    output_root.attrs["multiscales"] = multiscales


def prepare(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    input_path = Path(args.input).resolve()
    model_path = resolve_model_path(args.model)
    block = parse_zyx(args.block_shape)
    halo = parse_zyx(args.halo, allow_zero=True)
    chunks = parse_zyx(args.output_chunks)
    workers = int(args.num_workers)
    if workers <= 0:
        raise ValueError("num_workers must be positive")
    for axis in range(3):
        if block[axis] % chunks[axis] != 0:
            raise ValueError(
                f"Block {block} must be divisible by chunks {chunks} for safe writes"
            )

    input_root, image = open_array(input_path, key=args.array_key)
    shape = tuple(int(value) for value in image.shape)
    all_starts = block_starts(shape, block)
    selected = select_starts(
        all_starts,
        shape,
        block,
        float(args.process_fraction),
        int(args.max_blocks),
        args.preview_mode,
        int(args.preview_seed),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    work_path = run_root / "working_labels.ome.zarr"
    work_root, _ = create_array(
        work_path, shape, chunks, "uint32", int(args.compressor_level)
    )
    copy_multiscale_metadata(input_root, work_root)

    interval = UINT32_MAX // workers
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "input": str(input_path),
        "input_array_key": args.array_key,
        "model": str(model_path),
        "working_labels": str(work_path),
        "shape_zyx": list(shape),
        "block_shape_zyx": list(block),
        "halo_zyx": list(halo),
        "output_chunks_zyx": list(chunks),
        "compressor_level": int(args.compressor_level),
        "anisotropy": float(args.anisotropy),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "flow3D_smooth": float(args.flow3d_smooth),
        "pass_channels": bool(args.pass_channels),
        "skip_empty": bool(args.skip_empty),
        "min_core_nonzero_voxels": int(args.min_core_nonzero_voxels),
        "min_core_nonzero_fraction": float(args.min_core_nonzero_fraction),
        "process_fraction": float(args.process_fraction),
        "max_blocks": int(args.max_blocks),
        "preview_mode": args.preview_mode,
        "preview_seed": int(args.preview_seed),
        "planned_blocks": len(all_starts),
        "selected_blocks": [list(start) for start in selected],
        "num_workers": workers,
        "worker_label_interval": interval,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_root / "inference_config.txt").write_text(
        "\n".join(
            [
                f"input={input_path}",
                f"array_key={args.array_key}",
                f"model={model_path}",
                f"block_shape={','.join(map(str, block))}",
                f"halo={','.join(map(str, halo))}",
                f"output_chunks={','.join(map(str, chunks))}",
                f"anisotropy={args.anisotropy}",
                f"cellprob_threshold={args.cellprob_threshold}",
                f"min_size={args.min_size}",
                f"flow3D_smooth={args.flow3d_smooth}",
                "skip_policy=exactly_zero_core_only",
                f"process_fraction={args.process_fraction}",
                f"max_blocks={args.max_blocks}",
                f"preview_mode={args.preview_mode}",
                f"preview_seed={args.preview_seed}",
                f"num_workers={workers}",
            ]
        )
        + "\n"
    )
    (run_root / "PREPARED").touch()
    progress(
        f"Prepared {len(selected)}/{len(all_starts)} blocks on shape={shape}; "
        f"workers={workers}"
    )


def infer_worker(args: argparse.Namespace) -> None:
    from cellpose import core, models

    run_root = Path(args.run_root).resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    worker = int(args.worker_index)
    workers = int(manifest["num_workers"])
    if not 0 <= worker < workers:
        raise ValueError(f"worker_index must be in [0, {workers})")
    if not core.use_gpu():
        raise RuntimeError("Cellpose did not detect the GPU assigned to this worker")

    _, image = open_array(
        Path(manifest["input"]), key=manifest["input_array_key"]
    )
    _, output = open_array(Path(manifest["working_labels"]), mode="r+")
    model = models.CellposeModel(
        gpu=True, pretrained_model=str(Path(manifest["model"]))
    )
    shape = tuple(int(value) for value in manifest["shape_zyx"])
    block = tuple(int(value) for value in manifest["block_shape_zyx"])
    halo = tuple(int(value) for value in manifest["halo_zyx"])
    selected = [tuple(int(value) for value in start) for start in manifest["selected_blocks"]]
    assigned = selected[worker::workers]
    interval = int(manifest["worker_label_interval"])
    next_offset = worker * interval
    interval_end = (worker + 1) * interval
    processed = 0
    skipped = 0
    foreground = 0

    progress(
        f"worker={worker}/{workers} blocks={len(assigned)} model={manifest['model']}"
    )
    for index, start in enumerate(assigned, start=1):
        read_slices, write_slices, local_core = block_geometry(
            shape, start, block, halo
        )
        start_time = time.perf_counter()
        raw = np.asarray(image[read_slices])
        read_time = time.perf_counter()
        core_block = raw[local_core]
        core_nonzero = int(np.count_nonzero(core_block))
        core_fraction = core_nonzero / core_block.size if core_block.size else 0.0
        if manifest["skip_empty"] and (
            core_nonzero <= int(manifest["min_core_nonzero_voxels"])
            or core_fraction <= float(manifest["min_core_nonzero_fraction"])
        ):
            skipped += 1
            progress(
                f"worker={worker} [{index}/{len(assigned)}] skipped empty "
                f"start={start} frac={core_fraction:.5f}"
            )
            continue

        eval_kwargs: dict[str, Any] = {
            "x": raw,
            "do_3D": True,
            "z_axis": 0,
            "channel_axis": None,
            "anisotropy": float(manifest["anisotropy"]),
            "cellprob_threshold": float(manifest["cellprob_threshold"]),
            "min_size": int(manifest["min_size"]),
            "flow3D_smooth": float(manifest["flow3D_smooth"]),
        }
        if manifest["pass_channels"]:
            eval_kwargs["channels"] = [0, 0]
        prediction, _, _ = model.eval(**eval_kwargs)
        eval_time = time.perf_counter()
        core_prediction = np.asarray(prediction, dtype=np.uint32)[local_core]
        max_local = int(core_prediction.max())
        if max_local > 0:
            if next_offset + max_local > interval_end:
                raise OverflowError(
                    f"Worker {worker} exhausted its uint32 label interval"
                )
            core_prediction = core_prediction.copy()
            positive = core_prediction > 0
            core_prediction[positive] += np.uint32(next_offset)
            next_offset += max_local
            foreground += int(np.count_nonzero(positive))
        output[write_slices] = core_prediction
        write_time = time.perf_counter()
        processed += 1
        progress(
            f"worker={worker} [{index}/{len(assigned)}] wrote={start} "
            f"labels={max_local} read={read_time-start_time:.2f}s "
            f"eval={eval_time-read_time:.2f}s write={write_time-eval_time:.2f}s"
        )

    report = {
        "worker": worker,
        "num_workers": workers,
        "assigned_blocks": len(assigned),
        "processed_blocks": processed,
        "skipped_blocks": skipped,
        "foreground_voxels": foreground,
        "temporary_label_min": worker * interval + 1,
        "temporary_label_max_written": next_offset,
    }
    workers_dir = run_root / "workers"
    workers_dir.mkdir(exist_ok=True)
    (workers_dir / f"worker_{worker:02d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (workers_dir / f"worker_{worker:02d}.DONE").touch()
    progress(json.dumps(report, indent=2))


def compact_labels(block: np.ndarray, keys: np.ndarray) -> np.ndarray:
    output = np.zeros(block.shape, dtype=np.uint32)
    positive = block > 0
    if not positive.any():
        return output
    values = block[positive]
    positions = np.searchsorted(keys, values)
    if np.any(positions >= keys.size) or not np.array_equal(keys[positions], values):
        raise AssertionError("Temporary label was absent from the global label table")
    output[positive] = (positions + 1).astype(np.uint32)
    return output


def chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                yield (
                    slice(z0, min(shape[0], z0 + chunks[0])),
                    slice(y0, min(shape[1], y0 + chunks[1])),
                    slice(x0, min(shape[2], x0 + chunks[2])),
                )


def finalize(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    workers = int(manifest["num_workers"])
    missing = [
        worker
        for worker in range(workers)
        if not (run_root / "workers" / f"worker_{worker:02d}.DONE").is_file()
    ]
    if missing:
        raise RuntimeError(f"Cannot finalize; missing successful workers: {missing}")

    input_root, _ = open_array(
        Path(manifest["input"]), key=manifest["input_array_key"]
    )
    _, working = open_array(Path(manifest["working_labels"]))
    shape = tuple(int(value) for value in manifest["shape_zyx"])
    chunks = tuple(int(value) for value in manifest["output_chunks_zyx"])
    slices = list(chunk_slices(shape, chunks))

    unique_labels: set[int] = set()
    for index, selection in enumerate(slices, start=1):
        labels = np.unique(np.asarray(working[selection], dtype=np.uint32))
        unique_labels.update(int(value) for value in labels if value > 0)
        if index == 1 or index % 250 == 0 or index == len(slices):
            progress(f"Label inventory chunk {index}/{len(slices)}")
    keys = np.asarray(sorted(unique_labels), dtype=np.uint32)
    if keys.size > UINT32_MAX:
        raise OverflowError("More instances than uint32 can represent")

    output_path = Path(args.output).resolve()
    output_root, output = create_array(
        output_path,
        shape,
        chunks,
        "uint32",
        int(manifest["compressor_level"]),
    )
    copy_multiscale_metadata(input_root, output_root)
    foreground = 0
    for index, selection in enumerate(slices, start=1):
        source_block = np.asarray(working[selection], dtype=np.uint32)
        compact = compact_labels(source_block, keys)
        foreground += int(np.count_nonzero(compact))
        output[selection] = compact
        if index == 1 or index % 250 == 0 or index == len(slices):
            progress(f"Compact/write chunk {index}/{len(slices)}")

    inference_metadata = {
        "input": manifest["input"],
        "input_array": manifest["input_array_key"],
        "model": manifest["model"],
        "shape_zyx": list(shape),
        "block_shape_zyx": manifest["block_shape_zyx"],
        "halo_zyx": manifest["halo_zyx"],
        "output_chunks_zyx": list(chunks),
        "anisotropy": manifest["anisotropy"],
        "cellprob_threshold": manifest["cellprob_threshold"],
        "min_size": manifest["min_size"],
        "flow3D_smooth": manifest["flow3D_smooth"],
        "skip_empty": manifest["skip_empty"],
        "min_core_nonzero_voxels": manifest["min_core_nonzero_voxels"],
        "min_core_nonzero_fraction": manifest["min_core_nonzero_fraction"],
        "process_fraction": manifest["process_fraction"],
        "max_blocks": manifest["max_blocks"],
        "preview_mode": manifest["preview_mode"],
        "preview_seed": manifest["preview_seed"],
        "distributed_workers": workers,
        "distribution": "round-robin disjoint core blocks",
        "label_compaction": "temporary worker ranges remapped to consecutive global uint32 IDs",
        "note": "Labels are unique per processed block. Adjacent objects split at block boundaries are not merged.",
    }
    output_root.attrs["image-label"] = {
        "version": "0.5",
        "source": {"image": manifest["input"]},
    }
    output_root.attrs["cellpose_inference"] = inference_metadata
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(inference_metadata, indent=2) + "\n")

    total = int(math.prod(shape))
    worker_reports = [
        json.loads((run_root / "workers" / f"worker_{worker:02d}.json").read_text())
        for worker in range(workers)
    ]
    summary = {
        "output": str(output_path),
        "shape_zyx": list(shape),
        "chunks_zyx": list(chunks),
        "instances_before_boundary_reconciliation": int(keys.size),
        "foreground_voxels": foreground,
        "total_voxels": total,
        "foreground_fraction": foreground / total if total else 0.0,
        "workers": worker_reports,
        "note": "Foreground volume counts labels > 0. Cross-block instance reconciliation remains a separate optional step.",
    }
    (run_root / "foreground_volume_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not args.keep_working_labels:
        work_path = Path(manifest["working_labels"]).resolve()
        if work_path.parent != run_root or work_path.name != "working_labels.ome.zarr":
            raise RuntimeError(f"Refusing to remove unexpected working path: {work_path}")
        shutil.rmtree(work_path)
    (run_root / "COMPLETED").touch()
    progress(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and finalize distributed Cellpose OME-Zarr inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--array-key", default="0")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--run-root", required=True)
    prepare_parser.add_argument("--num-workers", type=int, default=4)
    prepare_parser.add_argument("--block-shape", default="40x256x256")
    prepare_parser.add_argument("--halo", default="5x32x32")
    prepare_parser.add_argument("--output-chunks", default="8x256x256")
    prepare_parser.add_argument("--compressor-level", type=int, default=5)
    prepare_parser.add_argument("--anisotropy", type=float, default=1.0)
    prepare_parser.add_argument("--cellprob-threshold", type=float, default=-2.25)
    prepare_parser.add_argument("--min-size", type=int, default=15)
    prepare_parser.add_argument("--flow3d-smooth", type=float, default=0.0)
    prepare_parser.add_argument("--skip-empty", action="store_true")
    prepare_parser.add_argument("--min-core-nonzero-voxels", type=int, default=0)
    prepare_parser.add_argument("--min-core-nonzero-fraction", type=float, default=0.0)
    prepare_parser.add_argument("--process-fraction", type=float, default=1.0)
    prepare_parser.add_argument("--max-blocks", type=int, default=0)
    prepare_parser.add_argument(
        "--preview-mode", choices=("central", "first", "random"), default="random"
    )
    prepare_parser.add_argument("--preview-seed", type=int, default=46)
    prepare_parser.add_argument("--pass-channels", action="store_true")

    worker_parser = subparsers.add_parser("infer-worker")
    worker_parser.add_argument("--run-root", required=True)
    worker_parser.add_argument("--worker-index", type=int, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-root", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--keep-working-labels", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "infer-worker":
        infer_worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
