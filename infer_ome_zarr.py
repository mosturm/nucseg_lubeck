#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Block-wise Cellpose inference on a ZYX OME-Zarr volume.

This is intended for the current organoid deployment volume:
    organoid_masked_uint16.ome.zarr/0

It writes an OME-Zarr label volume with the same ZYX shape. Blocks are processed
with a halo and only the core of each prediction is written, so the full volume
does not need to fit in GPU memory.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from cellpose import core, models


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr first: python -m pip install zarr") from exc

    try:
        from zarr.codecs import BloscCodec

        def make_compressor(level: int):
            return BloscCodec(cname="zstd", clevel=int(level))

    except ImportError:
        try:
            from numcodecs import Blosc
        except ImportError as exc:
            raise SystemExit("Install numcodecs first: python -m pip install numcodecs") from exc

        def make_compressor(level: int):
            return Blosc(cname="zstd", clevel=int(level), shuffle=Blosc.BITSHUFFLE)

    return zarr, make_compressor


def parse_zyx(text: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if len(values) != 3 or any(v < 0 for v in values):
        raise ValueError(f"Expected three non-negative integers as z,y,x; got {text!r}")
    return values


def resolve_model_path(model_arg: str | Path) -> Path:
    p = Path(model_arg)
    if p.is_file():
        return p
    for candidate in (p / "models" / "best_model", p / "best_model"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve model from {p}. Expected checkpoint file, "
        f"{p}/models/best_model, or {p}/best_model."
    )


def open_input_array(zarr_path: Path, array_key: str):
    zarr, _ = require_zarr()
    root = zarr.open_group(str(zarr_path), mode="r")
    if array_key not in root:
        raise KeyError(f"Array key {array_key!r} not found in {zarr_path}. Available keys: {list(root.keys())}")
    arr = root[array_key]
    if len(arr.shape) != 3:
        raise ValueError(f"Expected a 3D ZYX array at {array_key!r}; got shape {arr.shape}")
    return root, arr


def create_output_array(out_path: Path, shape: tuple[int, int, int], chunks: tuple[int, int, int], compressor_level: int):
    zarr, make_compressor = require_zarr()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_path), mode="w")
    compressor = make_compressor(compressor_level)

    if hasattr(root, "create_array"):
        try:
            arr = root.create_array("0", shape=shape, chunks=chunks, dtype="uint32", compressors=compressor, overwrite=True)
        except TypeError:
            arr = root.create_array("0", shape=shape, chunks=chunks, dtype="uint32", compressor=compressor, overwrite=True)
    else:
        arr = root.create_dataset("0", shape=shape, chunks=chunks, dtype="uint32", compressor=compressor, overwrite=True)
    return root, arr


def starts_for_axis(size: int, block: int) -> list[int]:
    if block <= 0:
        raise ValueError("Block sizes must be positive.")
    starts = list(range(0, size, block))
    if starts and starts[-1] >= size:
        starts.pop()
    return starts


def limited_block_starts(
    starts: list[tuple[int, int, int]],
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[tuple[int, int, int]]:
    fraction = float(args.process_fraction)
    max_blocks = int(args.max_blocks)
    if fraction >= 1.0 and max_blocks <= 0:
        return starts

    n_by_fraction = max(1, int(round(len(starts) * fraction))) if fraction < 1.0 else len(starts)
    n = min(len(starts), n_by_fraction)
    if max_blocks > 0:
        n = min(n, max_blocks)

    mode = str(args.preview_mode).lower()
    if mode == "first":
        return starts[:n]
    if mode == "random":
        rng = random.Random(int(args.preview_seed))
        return sorted(rng.sample(starts, n))
    if mode == "central":
        center = tuple((shape[i] - block_shape[i]) / 2.0 for i in range(3))
        return sorted(
            starts,
            key=lambda s: sum(((float(s[i]) - center[i]) / max(1.0, float(shape[i]))) ** 2 for i in range(3)),
        )[:n]
    raise ValueError("--preview_mode must be first, random, or central")


def block_slices(
    shape: tuple[int, int, int],
    core_start: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    core_stop = tuple(min(shape[i], core_start[i] + block_shape[i]) for i in range(3))
    read_start = tuple(max(0, core_start[i] - halo[i]) for i in range(3))
    read_stop = tuple(min(shape[i], core_stop[i] + halo[i]) for i in range(3))

    read_slices = tuple(slice(read_start[i], read_stop[i]) for i in range(3))
    write_slices = tuple(slice(core_start[i], core_stop[i]) for i in range(3))
    local_core = tuple(slice(core_start[i] - read_start[i], core_stop[i] - read_start[i]) for i in range(3))
    return read_slices, write_slices, local_core


def z_quarter_bounds(size_z: int, z_quarter: str) -> tuple[int, int] | None:
    if not z_quarter:
        return None
    value = str(z_quarter).strip().lower()
    if value in ("all", "0", "0/4", "none"):
        return None
    mapping = {"1/4": 0, "2/4": 1, "3/4": 2, "4/4": 3}
    if value not in mapping:
        raise ValueError("--z_quarter must be one of: all, 1/4, 2/4, 3/4, 4/4")
    edges = np.linspace(0, int(size_z), 5).round().astype(int)
    idx = mapping[value]
    return int(edges[idx]), int(edges[idx + 1])


def clip_write_to_z_bounds(
    read_slices: tuple[slice, slice, slice],
    write_slices: tuple[slice, slice, slice],
    local_core: tuple[slice, slice, slice],
    z_bounds: tuple[int, int] | None,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    if z_bounds is None:
        return write_slices, local_core
    z0 = max(int(write_slices[0].start), int(z_bounds[0]))
    z1 = min(int(write_slices[0].stop), int(z_bounds[1]))
    if z1 <= z0:
        return None
    clipped_write = (slice(z0, z1), write_slices[1], write_slices[2])
    clipped_local = (slice(z0 - int(read_slices[0].start), z1 - int(read_slices[0].start)), local_core[1], local_core[2])
    return clipped_write, clipped_local


def eval_block(
    model: models.CellposeModel,
    block: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    eval_kwargs: dict[str, Any] = {
        "x": block,
        "do_3D": True,
        "z_axis": 0,
        "channel_axis": None,
        "anisotropy": args.anisotropy,
        "cellprob_threshold": args.cellprob_threshold,
        "min_size": args.min_size,
        "flow3D_smooth": args.flow3D_smooth,
    }
    if args.pass_channels:
        eval_kwargs["channels"] = [0, 0]

    pred, flows, styles = model.eval(**eval_kwargs)
    return np.asarray(pred, dtype=np.uint32)


def write_ome_zarr_metadata(root, input_root, args: argparse.Namespace, shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> None:
    input_attrs = dict(getattr(input_root, "attrs", {}))
    input_multiscales = input_attrs.get("multiscales")
    if input_multiscales:
        multiscales = json.loads(json.dumps(input_multiscales))
        multiscales[0]["name"] = "cellpose_organoid_labels"
        multiscales[0]["datasets"][0]["path"] = "0"
    else:
        multiscales = [
            {
                "version": "0.4",
                "name": "cellpose_organoid_labels",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [{"path": "0"}],
            }
        ]

    root.attrs["multiscales"] = multiscales
    root.attrs["image-label"] = {
        "version": "0.5",
        "source": {"image": str(Path(args.input).resolve())},
    }
    root.attrs["cellpose_inference"] = {
        "input": str(Path(args.input).resolve()),
        "input_array": args.array_key,
        "model": str(resolve_model_path(args.model).resolve()),
        "shape_zyx": [int(v) for v in shape],
        "block_shape_zyx": [int(v) for v in parse_zyx(args.block_shape)],
        "halo_zyx": [int(v) for v in parse_zyx(args.halo)],
        "output_chunks_zyx": [int(v) for v in chunks],
        "anisotropy": float(args.anisotropy),
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "flow3D_smooth": float(args.flow3D_smooth),
        "skip_empty": bool(args.skip_empty),
        "min_core_nonzero_voxels": int(args.min_core_nonzero_voxels),
        "min_core_nonzero_fraction": float(args.min_core_nonzero_fraction),
        "process_fraction": float(args.process_fraction),
        "max_blocks": int(args.max_blocks),
        "preview_mode": str(args.preview_mode),
        "preview_seed": int(args.preview_seed),
        "z_quarter": str(args.z_quarter),
        "z_quarter_bounds": None
        if z_quarter_bounds(shape[0], args.z_quarter) is None
        else [int(v) for v in z_quarter_bounds(shape[0], args.z_quarter)],
        "note": "Labels are unique per processed block. Adjacent objects split at block boundaries are not merged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run block-wise Cellpose inference on a ZYX OME-Zarr volume.")
    parser.add_argument("--input", type=str, default="/user/unigoe.uprp-moham/u26421/.project/dir.project/nucseg/data_organoids/t0001/fullzarr/organoid_masked_uint16.ome.zarr", help="Input OME-Zarr group.")
    parser.add_argument("--array_key", type=str, default="0", help="Array key inside the input OME-Zarr.")
    parser.add_argument("--model", type=str, default="models/best_model", help="Checkpoint file or folder containing models/best_model.")
    parser.add_argument("--output", type=str, default="/user/unigoe.uprp-moham/u26421/.project/dir.project/nucseg/data_organoids/t0001/fullzarr/cellpose_organoid_labels.ome.zarr", help="Output label OME-Zarr group.")
    parser.add_argument("--block_shape", type=str, default="64,512,512", help="Core block shape as z,y,x.")
    parser.add_argument("--halo", type=str, default="8,64,64", help="Halo around each block as z,y,x. Only the core is written.")
    parser.add_argument("--output_chunks", type=str, default="8,256,256", help="Output Zarr chunks as z,y,x.")
    parser.add_argument("--compressor_level", type=int, default=5, help="Blosc zstd compression level.")
    parser.add_argument("--anisotropy", type=float, default=1.0, help="Z spacing / XY spacing for 3D inference.")
    parser.add_argument("--cellprob_threshold", type=float, default=-1.75, help="Best threshold from validation sweep.")
    parser.add_argument("--min_size", type=int, default=15, help="Minimum object size passed to Cellpose.")
    parser.add_argument("--flow3D_smooth", type=float, default=0.0, help="3D flow smoothing passed to Cellpose.")
    parser.add_argument("--skip_empty", action="store_true", help="Skip blocks that are entirely zero in the input OME-Zarr.")
    parser.add_argument("--min_core_nonzero_voxels", type=int, default=0, help="With --skip_empty, skip core blocks with fewer/equal nonzero voxels than this.")
    parser.add_argument("--min_core_nonzero_fraction", type=float, default=0.0, help="With --skip_empty, skip core blocks with nonzero fraction <= this value.")
    parser.add_argument("--process_fraction", type=float, default=1.0, help="Process only this fraction of planned blocks for quick QC, e.g. 0.05.")
    parser.add_argument("--max_blocks", type=int, default=0, help="Optional hard cap on number of planned blocks to process. 0 means no cap.")
    parser.add_argument("--preview_mode", type=str, default="central", choices=["central", "first", "random"], help="Which blocks to process when --process_fraction or --max_blocks limits the run.")
    parser.add_argument("--preview_seed", type=int, default=2026, help="Random seed for --preview_mode random.")
    parser.add_argument("--z_quarter", type=str, default="all", choices=["all", "1/4", "2/4", "3/4", "4/4"], help="Process only one quarter of z-slices.")
    parser.add_argument("--pass_channels", action="store_true", help="Pass channels=[0,0] to model.eval for older Cellpose builds.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = resolve_model_path(args.model)
    block_shape = parse_zyx(args.block_shape)
    halo = parse_zyx(args.halo)
    output_chunks = parse_zyx(args.output_chunks)
    if not (0.0 < float(args.process_fraction) <= 1.0):
        raise ValueError("--process_fraction must be >0 and <=1.")

    input_root, input_arr = open_input_array(input_path, args.array_key)
    shape = tuple(int(v) for v in input_arr.shape)
    z_bounds = z_quarter_bounds(shape[0], args.z_quarter)
    progress(f"Input: {input_path} / {args.array_key}")
    progress(f"Input shape z,y,x: {shape}")
    if z_bounds is not None:
        progress(f"Processing z quarter {args.z_quarter}: z={z_bounds[0]}:{z_bounds[1]}")
    progress(f"Output: {output_path}")

    output_root, output_arr = create_output_array(output_path, shape, output_chunks, args.compressor_level)

    use_gpu = core.use_gpu()
    progress(f"GPU available: {use_gpu}")
    progress(f"Using model: {model_path}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

    z_starts = starts_for_axis(shape[0], block_shape[0])
    y_starts = starts_for_axis(shape[1], block_shape[1])
    x_starts = starts_for_axis(shape[2], block_shape[2])
    all_starts = [(z0, y0, x0) for z0 in z_starts for y0 in y_starts for x0 in x_starts]
    if z_bounds is not None:
        all_starts = [
            (z0, y0, x0)
            for z0, y0, x0 in all_starts
            if min(shape[0], z0 + block_shape[0]) > z_bounds[0] and z0 < z_bounds[1]
        ]
    selected_starts = limited_block_starts(all_starts, shape, block_shape, args)
    total_blocks = len(selected_starts)
    progress(
        f"Processing {total_blocks}/{len(all_starts)} planned block(s) "
        f"with core={block_shape}, halo={halo}, preview_mode={args.preview_mode}"
    )

    next_label_offset = np.uint64(0)
    processed = 0
    skipped = 0

    for z0, y0, x0 in selected_starts:
        processed += 1
        read_slices, write_slices, local_core = block_slices(shape, (z0, y0, x0), block_shape, halo)
        clipped = clip_write_to_z_bounds(read_slices, write_slices, local_core, z_bounds)
        if clipped is None:
            skipped += 1
            progress(f"[{processed}/{total_blocks}] skipped outside z_quarter core zyx=({z0},{y0},{x0})")
            continue
        write_slices, local_core = clipped
        t0 = time.perf_counter()
        block = np.asarray(input_arr[read_slices])
        t_read = time.perf_counter()
        core_block = block[local_core]
        core_nonzero = int(np.count_nonzero(core_block))
        core_fraction = float(core_nonzero / core_block.size) if core_block.size else 0.0

        if args.skip_empty and (
            core_nonzero <= int(args.min_core_nonzero_voxels)
            or core_fraction <= float(args.min_core_nonzero_fraction)
        ):
            skipped += 1
            progress(
                f"[{processed}/{total_blocks}] skipped empty core zyx=({z0},{y0},{x0}) "
                f"nonzero={core_nonzero} frac={core_fraction:.5f} read={t_read - t0:.2f}s"
            )
            continue

        pred = eval_block(model, block, args)
        t_eval = time.perf_counter()
        pred_core = pred[local_core]

        max_label = int(pred_core.max())
        if max_label > 0:
            pred_core = pred_core.copy()
            fg = pred_core > 0
            pred_core[fg] = pred_core[fg] + np.uint32(next_label_offset)
            next_label_offset += np.uint64(max_label)

        output_arr[write_slices] = pred_core.astype(np.uint32, copy=False)
        t_write = time.perf_counter()
        progress(
            f"[{processed}/{total_blocks}] wrote z={write_slices[0].start}:{write_slices[0].stop} "
            f"y={write_slices[1].start}:{write_slices[1].stop} "
            f"x={write_slices[2].start}:{write_slices[2].stop} "
            f"labels={max_label} "
            f"core_nonzero_frac={core_fraction:.5f} "
            f"read={t_read - t0:.2f}s eval={t_eval - t_read:.2f}s write={t_write - t_eval:.2f}s"
        )

    write_ome_zarr_metadata(output_root, input_root, args, shape, output_chunks)
    sidecar = output_path.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(dict(output_root.attrs["cellpose_inference"]), f, indent=2)

    progress(f"Done. Processed={processed}, skipped_empty={skipped}, max_written_label={int(next_label_offset)}")
    progress(f"Wrote label OME-Zarr: {output_path}")
    progress(f"Wrote metadata sidecar: {sidecar}")


if __name__ == "__main__":
    main()
