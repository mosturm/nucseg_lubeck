r"""
Convert the original high-resolution TIFF into a cropped, masked uint16 OME-Zarr.

The downsampled organoid mask is upscaled by coordinate mapping into the original
TIFF coordinate system. Only the organoid bounding box plus margin is written.
Outside the upscaled mask is set to 0. Raw intensities are scaled to uint16 using
the same global display scaling used for the 3D training slabs.

The full 50 GB TIFF is never loaded into memory; the script streams z-chunks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TIF_PATH = Path(
    r"C:\Users\Moritz_Sturm\Desktop\Code\NinaOrganoids\organoid0009"
    r"\tomo_reco_id0009_t0001.tif"
)
DEFAULT_MASK_PATH = ""
DEFAULT_SCALING_METADATA = PROJECT_DIR / "3D_cubes" / "output_id0007_t0001" / "label_slab_metadata.json"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "ome_zarr_output_id0009_t0001"


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def require_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise SystemExit("Install tifffile first: python -m pip install tifffile") from exc
    return tifffile


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("This script needs zarr. Install it with: python -m pip install zarr") from exc

    try:
        from zarr.codecs import BloscCodec

        def make_compressor(level: int):
            return BloscCodec(cname="zstd", clevel=int(level))

    except ImportError:
        try:
            from numcodecs import Blosc
        except ImportError as exc:
            raise SystemExit(
                "This script needs numcodecs with this Zarr version. Install it with: python -m pip install numcodecs"
            ) from exc

        def make_compressor(level: int):
            return Blosc(cname="zstd", clevel=int(level), shuffle=Blosc.BITSHUFFLE)

    return zarr, make_compressor


def output_path_for_mask(mask_path: Path, output_root: Path) -> Path:
    return output_root / mask_path.parent.name / "organoid_masked_uint16.ome.zarr"


def output_path_for_tif(tif_path: Path, output_root: Path) -> Path:
    return output_root / tif_path.stem / f"{tif_path.stem}.ome.zarr"


def tiff_shape_zyx(tif) -> tuple[int, int, int]:
    series = tif.series[0]
    shape = tuple(int(value) for value in series.shape)
    axes = series.axes

    if len(shape) == 3 and set("ZYX").issubset(set(axes)):
        axis_to_size = {axis: size for axis, size in zip(axes, shape)}
        return int(axis_to_size["Z"]), int(axis_to_size["Y"]), int(axis_to_size["X"])

    if len(tif.pages) > 1:
        first = tif.pages[0]
        return int(len(tif.pages)), int(first.imagelength), int(first.imagewidth)

    if len(shape) == 3:
        return shape

    raise ValueError(f"Could not infer 3D z,y,x shape from TIFF series shape={shape}, axes={axes!r}")


def load_scaling(path: Path, vmin_arg: float | None, vmax_arg: float | None) -> tuple[float, float, dict]:
    if vmin_arg is not None and vmax_arg is not None:
        if np.isclose(vmin_arg, vmax_arg):
            raise ValueError("--vmin and --vmax must differ.")
        return float(vmin_arg), float(vmax_arg), {"source": "command_line", "display_vmin": float(vmin_arg), "display_vmax": float(vmax_arg)}

    if not path.exists():
        raise FileNotFoundError(
            f"Missing scaling metadata: {path}. Pass --vmin and --vmax, or generate slabs first."
        )

    with path.open("r", encoding="utf-8-sig") as f:
        metadata = json.load(f)
    scaling = metadata.get("display_scaling")
    if not scaling:
        raise ValueError(f"No display_scaling block found in {path}")
    return float(scaling["display_vmin"]), float(scaling["display_vmax"]), {
        "source": str(path),
        **scaling,
    }


def compute_crop_from_mask(
    mask: np.ndarray,
    tif_shape: tuple[int, int, int],
    margin_original: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_shape = np.asarray(mask.shape, dtype=np.int64)
    tif_shape_arr = np.asarray(tif_shape, dtype=np.int64)
    scale = tif_shape_arr.astype(np.float64) / mask_shape.astype(np.float64)

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask contains no positive organoid voxels.")

    mask_start = coords.min(axis=0)
    mask_stop = coords.max(axis=0) + 1
    crop_start = np.floor(mask_start.astype(np.float64) * scale).astype(np.int64) - int(margin_original)
    crop_stop = np.ceil(mask_stop.astype(np.float64) * scale).astype(np.int64) + int(margin_original)
    crop_start = np.maximum(crop_start, 0)
    crop_stop = np.minimum(crop_stop, tif_shape_arr)
    return crop_start, crop_stop, scale


def mask_for_original_chunk(
    mask: np.ndarray,
    start_zyx: np.ndarray,
    chunk_shape: tuple[int, int, int],
    scale_zyx: np.ndarray,
) -> np.ndarray:
    z0, y0, x0 = (int(value) for value in start_zyx)
    dz, dy, dx = (int(value) for value in chunk_shape)
    z_indices = np.clip(np.floor((np.arange(z0, z0 + dz, dtype=np.float64) + 0.5) / scale_zyx[0]).astype(int), 0, mask.shape[0] - 1)
    y_indices = np.clip(np.floor((np.arange(y0, y0 + dy, dtype=np.float64) + 0.5) / scale_zyx[1]).astype(int), 0, mask.shape[1] - 1)
    x_indices = np.clip(np.floor((np.arange(x0, x0 + dx, dtype=np.float64) + 0.5) / scale_zyx[2]).astype(int), 0, mask.shape[2] - 1)
    return np.asarray(mask[np.ix_(z_indices, y_indices, x_indices)] > 0)


def read_tiff_z_chunk(
    tif,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    as_float32: bool = True,
) -> np.ndarray:
    planes = []
    for z_idx in range(int(z0), int(z1)):
        plane = tif.pages[z_idx].asarray()
        cropped = plane[y0:y1, x0:x1]
        planes.append(np.asarray(cropped, dtype=np.float32) if as_float32 else np.asarray(cropped))
    return np.stack(planes, axis=0)


def scale_chunk_to_uint16(raw: np.ndarray, mask: np.ndarray | None, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmin, vmax):
        vmax = float(vmin) + 1.0
    scaled = (raw - float(vmin)) / (float(vmax) - float(vmin))
    scaled = np.clip(scaled, 0.0, 1.0)
    out = np.rint(scaled * 65535.0).astype(np.uint16)
    if mask is not None:
        out[~mask] = 0
    return out


def make_dataset(root, shape: tuple[int, int, int], chunks: tuple[int, int, int], compressor, dtype="uint16"):
    if hasattr(root, "create_dataset"):
        return root.create_dataset("0", shape=shape, chunks=chunks, dtype=dtype, compressor=compressor, overwrite=True)
    if hasattr(root, "create_array"):
        try:
            return root.create_array("0", shape=shape, chunks=chunks, dtype=dtype, compressors=compressor, overwrite=True)
        except TypeError:
            return root.create_array("0", shape=shape, chunks=chunks, dtype=dtype, compressor=compressor, overwrite=True)
    raise AttributeError("Zarr group has neither create_dataset nor create_array.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-convert original TIFF plus upscaled .npy mask to cropped uint16 OME-Zarr.")
    parser.add_argument("--tif", type=str, default=str(DEFAULT_TIF_PATH), help="Original high-resolution multi-page TIFF.")
    parser.add_argument("--mask", type=str, default=str(DEFAULT_MASK_PATH), help="Downsampled organoid mask .npy.")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root folder for OME-Zarr outputs.")
    parser.add_argument("--out", type=str, default=None, help="Exact .ome.zarr output path. Overrides --out-root.")
    parser.add_argument("--scaling-metadata", type=str, default=str(DEFAULT_SCALING_METADATA), help="Slab metadata containing global display scaling.")
    parser.add_argument("--vmin", type=float, default=None, help="Manual raw intensity vmin for uint16 scaling.")
    parser.add_argument("--vmax", type=float, default=None, help="Manual raw intensity vmax for uint16 scaling.")
    parser.add_argument("--margin-original", type=int, default=100, help="Margin around upscaled mask bbox in original voxels.")
    parser.add_argument("--z-chunk", type=int, default=8, help="Number of original z-planes read and written per stream chunk.")
    parser.add_argument("--zarr-chunks", type=str, default="8,256,256", help="Zarr chunk shape as z,y,x.")
    parser.add_argument("--compressor-level", type=int, default=5, help="Blosc zstd compression level.")
    args = parser.parse_args()

    tif_path = Path(args.tif)
    mask_arg = str(args.mask).strip()
    scaling_arg = str(args.scaling_metadata).strip()
    passthrough_raw = (mask_arg == "") and (scaling_arg == "")
    unmasked_scaled = (mask_arg == "") and (scaling_arg != "")
    mask_path = Path(mask_arg) if mask_arg else None
    if not tif_path.exists():
        raise FileNotFoundError(f"Original TIFF does not exist: {tif_path}")
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"Mask does not exist: {mask_path}")
    if args.z_chunk <= 0:
        raise ValueError("--z-chunk must be positive.")

    zarr_chunks = tuple(int(part.strip()) for part in args.zarr_chunks.split(",") if part.strip())
    if len(zarr_chunks) != 3 or any(value <= 0 for value in zarr_chunks):
        raise ValueError("--zarr-chunks must be three positive integers as z,y,x.")

    tifffile = require_tifffile()
    zarr, make_compressor = require_zarr()
    compressor = make_compressor(args.compressor_level)

    if passthrough_raw:
        vmin = vmax = None
        scaling_metadata = {"source": None, "policy": "none; raw TIFF values copied without scaling"}
        mask = None
        progress("No --mask and no --scaling-metadata provided; copying full TIFF to OME-Zarr without masking or scaling.")
    else:
        if not scaling_arg:
            raise ValueError("Provide --scaling-metadata, or pass both --mask \"\" and --scaling-metadata \"\" for raw passthrough.")
        vmin, vmax, scaling_metadata = load_scaling(Path(scaling_arg), args.vmin, args.vmax)
        progress(f"Using uint16 scaling vmin={vmin:.6g}, vmax={vmax:.6g}")

        if unmasked_scaled:
            mask = None
            progress("No --mask provided; converting full TIFF to scaled uint16 OME-Zarr without masking.")
        else:
            progress(f"Opening mask: {mask_path}")
            mask = np.load(mask_path, mmap_mode="r")
            progress(f"Mask shape z,y,x: {tuple(int(value) for value in mask.shape)}")

    out_path = Path(args.out) if args.out else (
        output_path_for_tif(tif_path, Path(args.out_root)) if mask_path is None else output_path_for_mask(mask_path, Path(args.out_root))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(tif_path) as tif:
        tif_shape = tiff_shape_zyx(tif)
        progress(f"Original TIFF shape z,y,x: {tif_shape}")
        if passthrough_raw or unmasked_scaled:
            crop_start = np.array([0, 0, 0], dtype=np.int64)
            crop_stop = np.asarray(tif_shape, dtype=np.int64)
            scale_zyx = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            output_dtype = tif.pages[0].dtype if passthrough_raw else "uint16"
        else:
            crop_start, crop_stop, scale_zyx = compute_crop_from_mask(mask, tif_shape, args.margin_original)
            output_dtype = "uint16"
        crop_shape = tuple(int(value) for value in (crop_stop - crop_start))
        progress(f"Crop start z,y,x: {[int(value) for value in crop_start]}")
        progress(f"Crop stop  z,y,x: {[int(value) for value in crop_stop]}")
        progress(f"Crop shape z,y,x: {crop_shape}")
        progress(f"Scale factors original/downsampled z,y,x: {[float(value) for value in scale_zyx]}")

        progress(f"Creating OME-Zarr store: {out_path}")
        root = zarr.open_group(str(out_path), mode="w")
        dataset = make_dataset(root, crop_shape, zarr_chunks, compressor, dtype=output_dtype)

        z_abs_start = int(crop_start[0])
        z_abs_stop = int(crop_stop[0])
        total_chunks = int(np.ceil(crop_shape[0] / args.z_chunk))
        for chunk_index, z0_abs in enumerate(range(z_abs_start, z_abs_stop, args.z_chunk), start=1):
            z1_abs = min(z0_abs + args.z_chunk, z_abs_stop)
            chunk_start_abs = np.array([z0_abs, int(crop_start[1]), int(crop_start[2])], dtype=np.int64)
            raw_chunk = read_tiff_z_chunk(
                tif,
                z0_abs,
                z1_abs,
                int(crop_start[1]),
                int(crop_stop[1]),
                int(crop_start[2]),
                int(crop_stop[2]),
                as_float32=not passthrough_raw,
            )
            if passthrough_raw:
                up_mask = None
                out_chunk = raw_chunk
            elif unmasked_scaled:
                up_mask = None
                out_chunk = scale_chunk_to_uint16(raw_chunk, None, vmin, vmax)
            else:
                up_mask = mask_for_original_chunk(mask, chunk_start_abs, raw_chunk.shape, scale_zyx)
                out_chunk = scale_chunk_to_uint16(raw_chunk, up_mask, vmin, vmax)
            z0_rel = z0_abs - z_abs_start
            z1_rel = z0_rel + out_chunk.shape[0]
            dataset[z0_rel:z1_rel, :, :] = out_chunk
            mask_fraction = 1.0 if up_mask is None else float(np.mean(up_mask))
            progress(
                f"Wrote z chunk {chunk_index}/{total_chunks}: abs_z={z0_abs}:{z1_abs}, "
                f"mask_fraction={mask_fraction:.4f}"
            )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "raw_tiff" if passthrough_raw else ("full_scaled_uint16" if unmasked_scaled else "organoid_masked_uint16"),
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [int(crop_start[0]), int(crop_start[1]), int(crop_start[2])]},
                    ],
                }
            ],
        }
    ]
    root.attrs["conversion"] = {
        "original_tif_path": str(tif_path),
        "downsampled_mask_path": None if mask_path is None else str(mask_path),
        "tif_shape_zyx": [int(value) for value in tif_shape],
        "mask_shape_zyx": None if mask is None else [int(value) for value in mask.shape],
        "crop_start_zyx_original": [int(value) for value in crop_start],
        "crop_stop_zyx_original": [int(value) for value in crop_stop],
        "crop_shape_zyx": [int(value) for value in crop_shape],
        "scale_factors_original_per_mask_voxel_zyx": [float(value) for value in scale_zyx],
        "mask_mapping": None if mask_path is None else "floor((original_index + 0.5) / scale_factor) per z,y,x axis",
        "outside_mask_value": None if mask_path is None else 0,
        "masking_applied": bool(mask_path is not None),
        "dtype": str(output_dtype),
        "scaling": scaling_metadata,
        "passthrough_raw_tiff": bool(passthrough_raw),
        "unmasked_scaled_uint16": bool(unmasked_scaled),
        "z_stream_chunk": int(args.z_chunk),
        "zarr_chunks": [int(value) for value in zarr_chunks],
        "note": (
            "Full original TIFF copied to OME-Zarr without masking or scaling."
            if passthrough_raw
            else (
                "Full original TIFF scaled to uint16 OME-Zarr without masking."
                if unmasked_scaled
                else "Cropped high-resolution original TIFF, masked by upscaled .npy organoid mask and scaled to uint16."
            )
        ),
    }

    sidecar_path = out_path.with_suffix(".json")
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(dict(root.attrs["conversion"]), f, indent=2)

    progress(f"Wrote OME-Zarr: {out_path}")
    progress(f"Wrote sidecar metadata: {sidecar_path}")


if __name__ == "__main__":
    main()
