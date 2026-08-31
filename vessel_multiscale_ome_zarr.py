from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


SCALES = ("unscaled", "mid", "coarse")


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_zyx(value: str, cast=float) -> tuple[Any, Any, Any]:
    normalized = str(value).replace("x", ",")
    parts = tuple(cast(part.strip()) for part in normalized.split(",") if part.strip())
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"Expected three positive ZYX values, got {value!r}")
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
    kwargs = dict(shape=shape, chunks=chunks, dtype=dtype, overwrite=True, fill_value=0)
    if hasattr(root, "create_array"):
        try:
            array = root.create_array("0", compressors=codec, **kwargs)
        except TypeError:
            array = root.create_array("0", compressor=codec, **kwargs)
    else:
        array = root.create_dataset("0", compressor=codec, **kwargs)
    return root, array


def open_array(path: Path, mode: str = "r", key: str = "0"):
    zarr, _ = zarr_tools()
    root = zarr.open_group(str(path), mode=mode)
    if key not in root:
        raise KeyError(f"Array key {key!r} not found in {path}")
    array = root[key]
    if len(array.shape) != 3:
        raise ValueError(f"Expected a ZYX array in {path}, got {array.shape}")
    return root, array


def destination_shape(
    source_shape: tuple[int, int, int], factor: tuple[float, float, float]
) -> tuple[int, int, int]:
    return tuple(max(1, int(round(source_shape[i] / factor[i]))) for i in range(3))


def downsample_chunked(
    source,
    destination,
    factor: tuple[float, float, float],
) -> None:
    from scipy.ndimage import map_coordinates

    # The recorded Fiji protocol used intensity interpolation in Image > Scale,
    # not an explicit averaging or Gaussian anti-aliasing operation.
    support = (2, 2, 2)
    chunks = tuple(int(value) for value in destination.chunks)
    shape = tuple(int(value) for value in destination.shape)
    source_shape = tuple(int(value) for value in source.shape)

    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                starts = (z0, y0, x0)
                stops = tuple(min(shape[i], starts[i] + chunks[i]) for i in range(3))
                coordinates = [
                    (np.arange(starts[i], stops[i], dtype=np.float64) + 0.5) * factor[i] - 0.5
                    for i in range(3)
                ]
                read_starts = tuple(
                    max(0, int(math.floor(coordinates[i][0])) - support[i]) for i in range(3)
                )
                read_stops = tuple(
                    min(
                        source_shape[i],
                        int(math.ceil(coordinates[i][-1])) + support[i] + 1,
                    )
                    for i in range(3)
                )
                source_block = np.asarray(
                    source[
                        read_starts[0] : read_stops[0],
                        read_starts[1] : read_stops[1],
                        read_starts[2] : read_stops[2],
                    ],
                    dtype=np.float32,
                )
                local_coordinates = np.meshgrid(
                    coordinates[0] - read_starts[0],
                    coordinates[1] - read_starts[1],
                    coordinates[2] - read_starts[2],
                    indexing="ij",
                )
                resized = map_coordinates(
                    source_block,
                    local_coordinates,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                ).astype(np.float32, copy=False)
                destination[
                    starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
                ] = resized
                progress(
                    f"downsample wrote z={starts[0]}:{stops[0]} "
                    f"y={starts[1]}:{stops[1]} x={starts[2]}:{stops[2]}"
                )


def write_basic_metadata(root, name: str, source: Path, factor: tuple[float, float, float]) -> None:
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{"path": "0"}],
        }
    ]
    root.attrs["vessel_multiscale"] = {
        "source": str(source.resolve()),
        "source_voxels_per_output_voxel_zyx": list(factor),
        "coordinate_rule": "output center maps to (index + 0.5) * factor - 0.5",
    }


def prepare(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    input_path = Path(args.input).resolve()
    chunks = tuple(int(value) for value in parse_zyx(args.chunks, int))
    mid_factor = tuple(float(value) for value in parse_zyx(args.mid_factor))
    coarse_factor = tuple(float(value) for value in parse_zyx(args.coarse_factor))
    input_root, source = open_array(input_path, key=args.array_key)
    source_shape = tuple(int(value) for value in source.shape)
    run_root.mkdir(parents=True, exist_ok=True)

    inputs = {"unscaled": str(input_path)}
    factors = {"unscaled": (1.0, 1.0, 1.0), "mid": mid_factor, "coarse": coarse_factor}
    shapes = {"unscaled": source_shape}
    for scale in ("mid", "coarse"):
        factor = factors[scale]
        output_path = run_root / "scaled_inputs" / f"{scale}.ome.zarr"
        shape = destination_shape(source_shape, factor)
        root, destination = create_array(output_path, shape, chunks, "float32", args.compressor_level)
        write_basic_metadata(root, f"{scale}_vessel_input", input_path, factor)
        progress(f"Creating {scale} input: shape={shape}, factor={factor}")
        downsample_chunked(source, destination, factor)
        inputs[scale] = str(output_path)
        shapes[scale] = shape

    outputs: dict[str, str] = {}
    for scale in SCALES:
        path = run_root / "scale_masks" / f"{scale}_semantic.ome.zarr"
        root, _ = create_array(path, shapes[scale], chunks, "uint8", args.compressor_level)
        write_basic_metadata(root, f"{scale}_vessel_semantic_mask", input_path, factors[scale])
        outputs[scale] = str(path)

    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "input": str(input_path),
        "input_array_key": args.array_key,
        "source_shape_zyx": source_shape,
        "chunks_zyx": chunks,
        "resampling": {
            "method": "trilinear center-coordinate sampling matching the recorded Fiji intensity-interpolation protocol",
            "upsampling_masks": "nearest-neighbor center-coordinate sampling",
            "factors_are_source_voxels_per_output_voxel": True,
            "protocol": "preprocessing_roi_scaling_protocol.txt",
        },
        "factors_zyx": {key: list(value) for key, value in factors.items()},
        "scale_inputs": inputs,
        "scale_shapes_zyx": shapes,
        "scale_masks": outputs,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_root / "PREPARED").touch()
    progress(f"Preparation complete: {run_root / 'manifest.json'}")


def block_starts(shape: tuple[int, int, int], block: tuple[int, int, int]):
    return [
        (z, y, x)
        for z in range(0, shape[0], block[0])
        for y in range(0, shape[1], block[1])
        for x in range(0, shape[2], block[2])
    ]


def infer_worker(args: argparse.Namespace) -> None:
    from cellpose import core, models

    run_root = Path(args.run_root).resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    deployment = json.loads(Path(args.deployment_config).read_text())
    block = tuple(int(value) for value in parse_zyx(args.block_shape, int))
    halo = tuple(int(value) for value in parse_zyx(args.halo, int))
    chunks = tuple(int(value) for value in manifest["chunks_zyx"])
    for axis in range(3):
        if block[axis] % chunks[axis] != 0:
            raise ValueError(
                f"Block shape {block} must be divisible by output chunks {chunks} for safe writes"
            )
    if not core.use_gpu():
        raise RuntimeError("Cellpose did not detect the GPU assigned to this array task")

    worker = int(args.worker_index)
    workers = int(args.num_workers)
    for scale in SCALES:
        _, image = open_array(Path(manifest["scale_inputs"][scale]))
        _, output = open_array(Path(manifest["scale_masks"][scale]), mode="r+")
        settings = deployment["scales"][scale]
        model_path = Path(settings["model"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing {scale} deployment model: {model_path}")
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        shape = tuple(int(value) for value in image.shape)
        starts = block_starts(shape, block)[worker::workers]
        progress(f"worker={worker}/{workers} scale={scale} blocks={len(starts)} model={model_path}")
        for index, start in enumerate(starts, start=1):
            stop = tuple(min(shape[i], start[i] + block[i]) for i in range(3))
            read_start = tuple(max(0, start[i] - halo[i]) for i in range(3))
            read_stop = tuple(min(shape[i], stop[i] + halo[i]) for i in range(3))
            raw = np.asarray(
                image[
                    read_start[0] : read_stop[0],
                    read_start[1] : read_stop[1],
                    read_start[2] : read_stop[2],
                ]
            )
            core_slices = tuple(
                slice(start[i] - read_start[i], stop[i] - read_start[i]) for i in range(3)
            )
            if not np.any(raw[core_slices]):
                progress(f"worker={worker} scale={scale} [{index}/{len(starts)}] empty {start}")
                continue
            prediction, _, _ = model.eval(
                x=raw,
                do_3D=True,
                z_axis=0,
                channel_axis=None,
                anisotropy=float(args.anisotropy),
                cellprob_threshold=float(settings["cellprob_threshold"]),
                min_size=int(settings["min_size"]),
                flow3D_smooth=float(settings["flow3D_smooth"]),
            )
            semantic = (np.asarray(prediction)[core_slices] > 0).astype(np.uint8)
            output[
                start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]
            ] = semantic
            progress(
                f"worker={worker} scale={scale} [{index}/{len(starts)}] "
                f"wrote={start}:{stop} foreground={int(semantic.sum())}"
            )
        del model

    workers_dir = run_root / "workers"
    workers_dir.mkdir(exist_ok=True)
    (workers_dir / f"worker_{worker:02d}.DONE").touch()
    progress(f"worker={worker} completed all scales")


def nearest_mask_block(mask, starts, stops, factor):
    indices = [
        np.clip(
            np.floor((np.arange(starts[i], stops[i]) + 0.5) / factor[i]).astype(np.int64),
            0,
            mask.shape[i] - 1,
        )
        for i in range(3)
    ]
    read_starts = tuple(int(values.min()) for values in indices)
    read_stops = tuple(int(values.max()) + 1 for values in indices)
    block = np.asarray(
        mask[
            read_starts[0] : read_stops[0],
            read_starts[1] : read_stops[1],
            read_starts[2] : read_stops[2],
        ]
    )
    local = [indices[i] - read_starts[i] for i in range(3)]
    return block[np.ix_(local[0], local[1], local[2])] > 0


def finalize(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    missing = [
        worker
        for worker in range(int(args.num_workers))
        if not (run_root / "workers" / f"worker_{worker:02d}.DONE").is_file()
    ]
    if missing:
        raise RuntimeError(f"Cannot finalize; missing successful GPU workers: {missing}")

    masks = {scale: open_array(Path(manifest["scale_masks"][scale]))[1] for scale in SCALES}
    factors = {
        scale: tuple(float(value) for value in manifest["factors_zyx"][scale])
        for scale in SCALES
    }
    shape = tuple(int(value) for value in manifest["source_shape_zyx"])
    chunks = tuple(int(value) for value in manifest["chunks_zyx"])
    output_path = Path(args.output).resolve()
    root, output = create_array(output_path, shape, chunks, "uint8", args.compressor_level)
    write_basic_metadata(root, "multiscale_vessel_union", Path(manifest["input"]), (1.0, 1.0, 1.0))
    root.attrs["vessel_multiscale_union"] = {
        "operation": "unscaled OR nearest_upsampled(mid) OR nearest_upsampled(coarse)",
        "manifest": str((run_root / "manifest.json").resolve()),
        "factors_zyx": manifest["factors_zyx"],
    }

    counts = {scale: 0 for scale in SCALES}
    union_count = 0
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                starts = (z0, y0, x0)
                stops = tuple(min(shape[i], starts[i] + chunks[i]) for i in range(3))
                combined = np.zeros(tuple(stops[i] - starts[i] for i in range(3)), dtype=bool)
                for scale in SCALES:
                    mapped = nearest_mask_block(masks[scale], starts, stops, factors[scale])
                    counts[scale] += int(mapped.sum())
                    combined |= mapped
                output[z0 : stops[0], y0 : stops[1], x0 : stops[2]] = combined.astype(np.uint8)
                union_count += int(combined.sum())
    summary = {
        "output": str(output_path),
        "shape_zyx": shape,
        "foreground_voxels_after_mapping_by_scale": counts,
        "union_foreground_voxels": union_count,
        "union_foreground_fraction": union_count / int(np.prod(shape)),
    }
    (run_root / "union_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_root / "COMPLETED").touch()
    progress(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, infer, and union multiscale vessel OME-Zarr masks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--array-key", default="0")
    prepare_parser.add_argument("--run-root", required=True)
    prepare_parser.add_argument("--mid-factor", required=True)
    prepare_parser.add_argument("--coarse-factor", required=True)
    prepare_parser.add_argument("--chunks", default="8x256x256")
    prepare_parser.add_argument("--compressor-level", type=int, default=5)

    worker_parser = subparsers.add_parser("infer-worker")
    worker_parser.add_argument("--run-root", required=True)
    worker_parser.add_argument("--deployment-config", required=True)
    worker_parser.add_argument("--worker-index", type=int, required=True)
    worker_parser.add_argument("--num-workers", type=int, default=8)
    worker_parser.add_argument("--block-shape", default="40x256x256")
    worker_parser.add_argument("--halo", default="5x32x32")
    worker_parser.add_argument("--anisotropy", type=float, default=1.0)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-root", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--num-workers", type=int, default=8)
    finalize_parser.add_argument("--compressor-level", type=int, default=5)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args)
    elif args.command == "infer-worker":
        infer_worker(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
