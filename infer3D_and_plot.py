#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D inference + mask-only interactive visualization for Cellpose / Cellpose-SAM.

Features
- Loads trained model, runs true 3D segmentation (e.g. on 800x800x60).
- Uses anisotropy spacing (Z scale) so geometry is correct in 3D.
- Post-process: unify touching cells (face-adjacent) into one (keep ID of the bigger one) *for plotting only*.
- Exports:
  * pred_mask3d.tif (int32 labels from model)
  * pred_mask3d_merged_for_plot.tif (labels after "touching merge")
  * segmentation_3d.html (interactive, rotatable 3D mesh of mask-only)  [needs plotly]
  * mesh_triangles.parquet (or .csv) — triangles to rebuild the 3D view
  * eval_png/ (per-slice PNGs for raw + merged labels; binary + instance-colored)

Dependencies
    pip install cellpose tifffile numpy scikit-image pandas plotly pyarrow pillow
(If plotly/pyarrow are missing, the script still runs and falls back to HTML skip and CSV.)

Notes
- In CPSAM v4+, channels are deprecated; do not pass channels.
- In 3D CPSAM, flow_threshold is ignored; tune cellprob/min_size/flow3D_smooth instead.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import tifffile as tiff

from skimage.measure import marching_cubes
from cellpose import models, core

from PIL import Image
import colorsys

# ---------------------- Orientation helper ----------------------
Z_AXIS: int | None = None
CHANNEL_AXIS: int | None = None

def ensure_zyx(arr: np.ndarray, name: str = "array") -> np.ndarray:
    a = np.asarray(arr)
    if CHANNEL_AXIS is not None and CHANNEL_AXIS < a.ndim:
        a = np.take(a, indices=0, axis=CHANNEL_AXIS)
        a = np.array(a)
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D after channel handling; got shape {a.shape}")
    if Z_AXIS is not None:
        if not (0 <= Z_AXIS <= 2):
            raise ValueError("Z_AXIS must be 0/1/2 or None")
        if Z_AXIS != 0:
            a = np.moveaxis(a, Z_AXIS, 0)
        return a
    sizes = a.shape
    z_guess = int(np.argmin(sizes))
    if sizes[z_guess] * 2 <= min(sizes[(z_guess + 1) % 3], sizes[(z_guess + 2) % 3]):
        if z_guess != 0:
            a = np.moveaxis(a, z_guess, 0)
    return a

# ---------------------- Unify touching labels -------------------
def merge_touching_labels(labels: np.ndarray) -> np.ndarray:
    assert labels.ndim == 3
    lab = labels.copy()
    maxlab = int(lab.max())
    if maxlab == 0:
        return lab

    parent = np.arange(maxlab + 1, dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Z
    a = lab[:-1, :, :]
    b = lab[1:, :, :]
    m = (a > 0) & (b > 0) & (a != b)
    if m.any():
        for x, y in zip(a[m], b[m]): union(int(x), int(y))
    # Y
    a = lab[:, :-1, :]
    b = lab[:, 1:, :]
    m = (a > 0) & (b > 0) & (a != b)
    if m.any():
        for x, y in zip(a[m], b[m]): union(int(x), int(y))
    # X
    a = lab[:, :, :-1]
    b = lab[:, :, 1:]
    m = (a > 0) & (b > 0) & (a != b)
    if m.any():
        for x, y in zip(a[m], b[m]): union(int(x), int(y))

    sizes = Counter(lab[lab > 0].tolist())
    members = defaultdict(list)
    for k in range(1, maxlab + 1):
        if sizes.get(k, 0) > 0:
            members[find(k)].append(k)
    rep = {}
    for root, lst in members.items():
        best = max(lst, key=lambda k: sizes.get(k, 0))
        for k in lst: rep[k] = best

    lut = np.arange(maxlab + 1, dtype=np.int32)
    for k, v in rep.items(): lut[k] = v
    return lut[lab]

# ---------------------- Mesh & Visualization --------------------
def mesh_from_binary(bin_vol: np.ndarray, z_spacing: float = 1.0, step_size: int = 2):
    vol = (bin_vol > 0).astype(np.uint8)
    verts, faces, normals, values = marching_cubes(
        vol, level=0.5, spacing=(z_spacing, 1.0, 1.0),
        step_size=step_size, allow_degenerate=False
    )
    return verts, faces

def save_plotly_mesh(verts: np.ndarray, faces: np.ndarray, out_html: Path):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("[WARN] Plotly not available, skipping HTML export:", e)
        return
    x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
    i, j, k = faces[:, 2], faces[:, 1], faces[:, 0]
    fig = go.Figure(data=[
        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=1.0, name="mask")
    ])
    fig.update_layout(
        title="3D Segmentation (mask-only)",
        scene=dict(xaxis_title="X (px)", yaxis_title="Y (px)", zaxis_title="Z (px)", aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[OK] Interactive HTML written: {out_html}")

def save_triangles_dataframe(verts: np.ndarray, faces: np.ndarray, out_base: Path):
    tri = verts[faces]  # (M,3,3) in (Z,Y,X)
    tri_xyz = np.stack([tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]], axis=2)
    df = pd.DataFrame({
        "x0": tri_xyz[:, 0, 0], "y0": tri_xyz[:, 0, 1], "z0": tri_xyz[:, 0, 2],
        "x1": tri_xyz[:, 1, 0], "y1": tri_xyz[:, 1, 1], "z1": tri_xyz[:, 1, 2],
        "x2": tri_xyz[:, 2, 0], "y2": tri_xyz[:, 2, 1], "z2": tri_xyz[:, 2, 2],
    })
    try:
        df.to_parquet(str(out_base.with_suffix(".parquet")), index=False)
        print(f"[OK] Triangles DataFrame (parquet): {out_base.with_suffix('.parquet')}")
    except Exception as e:
        print("[WARN] Parquet save failed (install pyarrow). Falling back to CSV:", e)
        df.to_csv(str(out_base.with_suffix(".csv")), index=False)
        print(f"[OK] Triangles DataFrame (csv): {out_base.with_suffix('.csv')}")

# ---------------------- Eval PNG helpers -----------------------
def _label_colormap(n: int) -> np.ndarray:
    cmap = np.zeros((n, 3), dtype=np.uint8)
    phi = (1 + 5**0.5) / 2
    for k in range(1, n):
        h = (k / phi) % 1.0
        s, v = 0.65, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cmap[k] = (int(r * 255), int(g * 255), int(b * 255))
    return cmap

def _instances_rgb(slice_labels: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    sl = np.clip(slice_labels.astype(np.int64), 0, cmap.shape[0] - 1)
    return cmap[sl]  # (H,W,3) uint8

def save_eval_pngs(outdir: Path, labels_pred: np.ndarray, labels_merged: np.ndarray, limit_z: int | None = None):
    pred_dir   = outdir / "eval_png" / "pred"
    merged_dir = outdir / "eval_png" / "merged"
    pred_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    Z = labels_pred.shape[0] if limit_z is None else min(labels_pred.shape[0], limit_z)

    max_id = int(max(labels_pred.max(), labels_merged.max()))
    cmap = _label_colormap(max_id + 1)

    for z in range(Z):
        # raw prediction
        sl = labels_pred[z]
        Image.fromarray((sl > 0).astype(np.uint8) * 255, mode="L").save(pred_dir / f"binary_z{z:04d}.png")
        Image.fromarray(_instances_rgb(sl, cmap), mode="RGB").save(pred_dir / f"instances_z{z:04d}.png")
        # merged-for-plotting
        slm = labels_merged[z]
        Image.fromarray((slm > 0).astype(np.uint8) * 255, mode="L").save(merged_dir / f"binary_z{z:04d}.png")
        Image.fromarray(_instances_rgb(slm, cmap), mode="RGB").save(merged_dir / f"instances_z{z:04d}.png")

    print(f"[OK] Eval PNGs written under: {outdir/'eval_png'}")

# ---------------------- Inference ------------------------------
def run_inference_3d(img_path: Path,
                     model_path: Path,
                     anisotropy: float = 5.0,
                     cellprob_threshold: float = -1.0,
                     flow_threshold: float = 0.2,   # ignored in CPSAM 3D; kept for logging
                     min_size: int = 0,
                     flow3D_smooth: float = 0.0):
    use_gpu = core.use_gpu()
    print("GPU available:", use_gpu)

    print(f"Reading image: {img_path}")
    vol = tiff.imread(str(img_path))
    vol = ensure_zyx(vol, "image")
    print("Volume shape (Z,Y,X):", vol.shape)

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    print("Running 3D inference ...")
    # NOTE: No channels, no tile/tile_overlap here (not supported by your build).
    masks_pred, flows, styles = model.eval(
        x=vol,
        do_3D=True,
        z_axis=0,
        channel_axis=None,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        flow3D_smooth=flow3D_smooth
    )
    masks_pred = masks_pred.astype(np.int32)
    print("Pred done. Labels range:", int(masks_pred.min()), "to", int(masks_pred.max()))
    return vol, masks_pred

# ---------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  type=Path, default=Path("./test_data_tomo_reco_Rohrle18_Probe47_segm1.tif"), help="Path to 3D test TIF (single-channel).")
    ap.add_argument("--model", type=Path, default=Path("./models/my_3d_finetune_120epochs"), help="Path to trained model file or folder.")
    ap.add_argument("--outdir", type=Path, default=Path("inference3d"))
    ap.add_argument("--anisotropy", type=float, default=5.0, help="Z spacing in pixels.")
    ap.add_argument("--cellprob", type=float, default=-4.0)
    ap.add_argument("--flow", type=float, default=0.2)  # kept for meta only
    ap.add_argument("--min_size", type=int, default=0, help="Filter tiny objects (voxel count) in 3D.")
    ap.add_argument("--flow3D_smooth", type=float, default=0.1, help="Smoothing on 3D flow (CPSAM).")
    ap.add_argument("--mesh_step", type=int, default=1, help="Marching cubes step size (2=coarser/faster, 1=finer/slower).")
    ap.add_argument("--eval_png_maxz", type=int, default=None, help="Optionally cap number of slices exported as PNGs.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Inference
    vol, labels = run_inference_3d(
        img_path=args.img, model_path=args.model,
        anisotropy=args.anisotropy,
        cellprob_threshold=args.cellprob,
        flow_threshold=args.flow,
        min_size=args.min_size,
        flow3D_smooth=args.flow3D_smooth
    )

    # Save raw predicted labels
    pred_tif = args.outdir / "pred_mask3d.tif"
    tiff.imwrite(str(pred_tif), labels.astype(np.int32))
    print(f"[OK] Saved predicted 3D labels: {pred_tif}")

    # Merge touching labels only for plotting
    labels_merged = merge_touching_labels(labels)
    merged_tif = args.outdir / "pred_mask3d_merged_for_plot.tif"
    tiff.imwrite(str(merged_tif), labels_merged.astype(np.int32))
    print(f"[OK] Saved merged-for-plot labels: {merged_tif}")

    # Per-slice eval PNGs
    save_eval_pngs(args.outdir, labels, labels_merged, limit_z=args.eval_png_maxz)

    # Build mesh from binary (mask-only)
    bin_vol = (labels_merged > 0)
    if not bin_vol.any():
        raise RuntimeError("No positive voxels in prediction — check thresholds or model.")

    verts, faces = mesh_from_binary(bin_vol, z_spacing=args.anisotropy, step_size=args.mesh_step)

    # Save interactive HTML (movable 3D)
    html_path = args.outdir / "segmentation_3d.html"
    save_plotly_mesh(verts, faces, html_path)

    # Save triangles DataFrame for reproducible 3D view
    df_base = args.outdir / "mesh_triangles"
    save_triangles_dataframe(verts, faces, df_base)

    # Save meta for reproducibility
    meta = {
        "img": str(args.img),
        "model": str(args.model),
        "anisotropy": args.anisotropy,
        "cellprob": args.cellprob,
        "flow": args.flow,  # ignored in CPSAM 3D; kept for record
        "min_size": args.min_size,
        "flow3D_smooth": args.flow3D_smooth,
        "mesh_step": args.mesh_step,
        "shape_zyx": list(bin_vol.shape),
    }
    with open(args.outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Wrote meta: {args.outdir / 'meta.json'}")

    print("\nDone. PNGs are under eval_png/. Open the HTML in a browser to rotate/zoom.")

if __name__ == "__main__":
    main()
