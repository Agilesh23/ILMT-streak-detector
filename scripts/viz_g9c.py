# TASK G9c: save clean reference images for 20221026-0059
"""
Clean reference images for frame 20221026-0059 (candidates 270 and 10323) with NO overlays:
- 20221026-0059_clean_overview.png: whole frame downsampled
- 20221026-0059_clean_zoom_600.png: 600x600 px crop centered at row=1980, col=415, full resolution
- 20221026-0059_clean_zoom_1200.png: 1200x1200 px crop at the same center, downsampled by 2

Copies FITS loading, background estimation, crop extraction and stretch code directly from
scripts/viz_g9b.py (same stretch for visual consistency).
Outputs to data/d0b/viz_g9b/.
"""

import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import AsinhNorm
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.io_utils import load_fits


def get_crop(img, cy, cx, size):
    """
    Extract a square crop of given size centered at (cy, cx) with constant 0 padding.
    Directly reused from scripts/viz_g9b.py (and scripts/g1_run.py).
    cy: row (y), cx: col (x).
    Returns (crop, (c0, c1, r0, r1), (c0_cl, c1_cl, r0_cl, r1_cl)).
    """
    half = size // 2
    r0 = int(round(cy)) - half
    r1 = r0 + size
    c0 = int(round(cx)) - half
    c1 = c0 + size

    h, w = img.shape
    r0_cl = max(0, r0)
    r1_cl = min(h, r1)
    c0_cl = max(0, c0)
    c1_cl = min(w, c1)

    sub_crop = img[r0_cl:r1_cl, c0_cl:c1_cl]
    pad_top = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left = max(0, -c0)
    pad_right = max(0, c1 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(sub_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    else:
        crop = sub_crop
    return crop, (c0, c1, r0, r1), (c0_cl, c1_cl, r0_cl, r1_cl)


def find_fits_file(stem):
    """Search for FITS file by stem in standard project search directories."""
    search_dirs = [
        Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials"),
        base_dir / "data",
    ]
    target_name = f"{stem.lower()}.fits"
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.rglob("*.fits"):
            if p.name.lower() == target_name:
                return p
    return None


def make_asinh_norm(data, vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0):
    """Compute AsinhNorm matching astronomical image stretch standards from scripts/viz_g9b.py."""
    vmin = float(np.percentile(data, vmin_pct))
    vmax = float(np.percentile(data, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    try:
        return AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax), vmin, vmax
    except Exception:
        return None, vmin, vmax


def save_clean_image(out_path, img_data, norm=None, vmin=None, vmax=None):
    """Save raw image with NO overlay, NO markers, NO text, and NO axes."""
    if norm is not None:
        scaled = np.clip(norm(img_data), 0.0, 1.0)
        plt.imsave(out_path, scaled, cmap="gray", origin="lower")
    else:
        plt.imsave(out_path, img_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)


def main():
    print("=" * 80)
    print("TASK G9c: SAVE CLEAN REFERENCE IMAGES FOR FRAME 20221026-0059")
    print("=" * 80)

    stem = "20221026-0059"
    fits_path = find_fits_file(stem)
    if not fits_path or not fits_path.exists():
        print(f"Error: FITS file for {stem} not found.")
        sys.exit(1)

    print(f"Loaded FITS: {fits_path}")

    # Output directory
    out_dir = base_dir / "data/d0b/viz_g9b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # 1. Load FITS and background subtraction
    data, header = load_fits(fits_path)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    h_img, w_img = data.shape
    print(f"Image dimensions: {h_img} x {w_img} px")

    saved_records = []

    # Target center for candidates 270 and 10323
    center_r = 1980
    center_c = 415

    # -------------------------------------------------------------------------
    # Target 1: Whole frame downsampled overview
    # -------------------------------------------------------------------------
    scale_ds = max(1, int(math.ceil(max(h_img, w_img) / 1000.0)))
    h_ds = h_img // scale_ds
    w_ds = w_img // scale_ds
    frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
    vmin_ov = float(np.percentile(frame_ds, 1.0))
    vmax_ov = float(np.percentile(frame_ds, 99.5))

    ov_path = out_dir / "20221026-0059_clean_overview.png"
    save_clean_image(ov_path, frame_ds, norm=None, vmin=vmin_ov, vmax=vmax_ov)
    saved_records.append({
        "name": "Overview (downsampled)",
        "path": ov_path,
        "row_bounds_requested": (0, h_img),
        "col_bounds_requested": (0, w_img),
        "row_bounds_clipped": (0, h_img),
        "col_bounds_clipped": (0, w_img),
        "output_shape": frame_ds.shape,
    })

    # -------------------------------------------------------------------------
    # Target 2: 600x600 px crop centered at row=1980, col=415 (full resolution)
    # -------------------------------------------------------------------------
    crop_600, req_600, clip_600 = get_crop(sub, center_r, center_c, 600)
    norm_600, vmin_600, vmax_600 = make_asinh_norm(crop_600, vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0)

    zoom_600_path = out_dir / "20221026-0059_clean_zoom_600.png"
    save_clean_image(zoom_600_path, crop_600, norm=norm_600, vmin=vmin_600, vmax=vmax_600)
    saved_records.append({
        "name": "Zoom 600x600 (full-res)",
        "path": zoom_600_path,
        "row_bounds_requested": (req_600[2], req_600[3]),
        "col_bounds_requested": (req_600[0], req_600[1]),
        "row_bounds_clipped": (clip_600[2], clip_600[3]),
        "col_bounds_clipped": (clip_600[0], clip_600[1]),
        "output_shape": crop_600.shape,
    })

    # -------------------------------------------------------------------------
    # Target 3: 1200x1200 px crop centered at row=1980, col=415 (downsampled by 2)
    # -------------------------------------------------------------------------
    crop_1200, req_1200, clip_1200 = get_crop(sub, center_r, center_c, 1200)
    # Block-mean downsampling by factor of 2
    crop_1200_ds = crop_1200.reshape(600, 2, 600, 2).mean(axis=(1, 3))
    norm_1200, vmin_1200, vmax_1200 = make_asinh_norm(crop_1200_ds, vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0)

    zoom_1200_path = out_dir / "20221026-0059_clean_zoom_1200.png"
    save_clean_image(zoom_1200_path, crop_1200_ds, norm=norm_1200, vmin=vmin_1200, vmax=vmax_1200)
    saved_records.append({
        "name": "Zoom 1200x1200 (downsampled x2)",
        "path": zoom_1200_path,
        "row_bounds_requested": (req_1200[2], req_1200[3]),
        "col_bounds_requested": (req_1200[0], req_1200[1]),
        "row_bounds_clipped": (clip_1200[2], clip_1200[3]),
        "col_bounds_clipped": (clip_1200[0], clip_1200[1]),
        "output_shape": crop_1200_ds.shape,
    })

    # -------------------------------------------------------------------------
    # Print Summary & Bounds Report
    # -------------------------------------------------------------------------
    print("SAVED CLEAN REFERENCE IMAGES & CROP BOUNDS:")
    print("-" * 80)
    for rec in saved_records:
        r_req0, r_req1 = rec["row_bounds_requested"]
        c_req0, c_req1 = rec["col_bounds_requested"]
        r_cl0, r_cl1 = rec["row_bounds_clipped"]
        c_cl0, c_cl1 = rec["col_bounds_clipped"]

        print(f"* {rec['name']}:")
        print(f"  Path:               {rec['path'].relative_to(base_dir)}")
        print(f"  Output Pixel Shape: {rec['output_shape'][0]} x {rec['output_shape'][1]} px")
        print(f"  Requested Bounds:   row [{r_req0}, {r_req1}], col [{c_req0}, {c_req1}]")
        print(f"  Actual In-Frame:    row [{r_cl0}, {r_cl1}], col [{c_cl0}, {c_cl1}] (span: {r_cl1 - r_cl0} x {c_cl1 - c_cl0} px)")
        if (c_cl0 != c_req0) or (c_cl1 != c_req1) or (r_cl0 != r_req0) or (r_cl1 != r_req1):
            pad_left = max(0, -c_req0)
            print(f"  Note on Clipping:   Clipped at image left border (col 0); padded with {pad_left} px constant 0 to maintain center")
        print("-" * 80)

    print("Completed successfully.")


if __name__ == "__main__":
    main()
