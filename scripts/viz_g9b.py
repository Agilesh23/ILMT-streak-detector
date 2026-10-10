# TASK G9b: save check images for 2 lines and 1 duplicate pair
"""
Diagnostic visualizer for TASK G9b:
- Line 20230615-0006_L1: overview, 3 zooms (25%, 50%, 75% length, with overlays and _clean), strip (0 px & +-25 px offsets)
- Line 20240502-0026_L1: overview, 3 zooms (25%, 50%, 75% length, with overlays and _clean), strip (0 px & +-25 px offsets)
- Frame 20221026-0059.fits (candidates 270 and 10323): overview with both candidates in different colors,
  and 300x300 zoom centered at duplicate overlap with overlays and _clean.

Reuses FITS loading (src.io_utils.load_fits), background subtraction (src.background.estimate_background),
crop extraction (get_crop matching scripts/g1_run.py), and percentile / AsinhNorm stretches.
Reads candidates_v2.csv only. Output directory: data/d0b/viz_g9b/.
"""

import math
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import AsinhNorm, LogNorm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates
from src.io_utils import load_fits


def parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' into ((r0, c0), (r1, c1))."""
    if not ep_str or pd.isna(ep_str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def get_crop(img, cy, cx, size):
    """
    Extract a square crop of given size centered at (cy, cx) with constant 0 padding.
    Directly reused from scripts/g1_run.py and scripts/check_panels.py.
    cy: row (y), cx: col (x).
    Returns (crop, (c0, c1, r0, r1)).
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
    return crop, (c0, c1, r0, r1)


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
    """Compute AsinhNorm matching astronomical image stretch standards."""
    vmin = float(np.percentile(data, vmin_pct))
    vmax = float(np.percentile(data, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    try:
        return AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax), vmin, vmax
    except Exception:
        return None, vmin, vmax


def process_line_target(row, fits_path, out_dir, cfg):
    """Generate overview, 3 zooms (overlay + clean), and strip for a line candidate."""
    cand_id = row["cand_v2_id"]
    stem = row["frame"]
    frame_name = row["frame_name"]
    saved_files = []

    # 1. Load FITS and background
    data, header = load_fits(fits_path)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    h_img, w_img = data.shape

    # Endpoints & Geometry: (row0, col0), (row1, col1)
    endpoints = parse_endpoints(row["endpoints"])
    if endpoints is None:
        print(f"Error: Could not parse endpoints for {cand_id}: {row['endpoints']}")
        return saved_files

    (ep0_r, ep0_c), (ep1_r, ep1_c) = endpoints
    dr = ep1_r - ep0_r
    dc = ep1_c - ep0_c
    length_px = float(row["length_px"]) if pd.notna(row.get("length_px")) else math.hypot(dr, dc)
    if length_px > 1e-4:
        ur = dr / length_px
        uc = dc / length_px
    else:
        ur, uc = 1.0, 0.0

    # Retrieve member fragments from RAW blobs to mark centroids
    frag_ids = set()
    if pd.notna(row.get("fragment_ids")):
        for x in str(row["fragment_ids"]).split(";"):
            if x.strip():
                try:
                    frag_ids.add(int(x))
                except ValueError:
                    pass

    member_centroids = []
    if frag_ids:
        det_cfg = cfg.get("detection", {})
        edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
        border_margin_px = det_cfg.get("border_margin_px", 15)
        border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
        left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
        left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
        bypass_cfg = det_cfg.get("long_thin_bypass", None)
        bleed_veto_cfg = det_cfg.get("bleed_veto", None)

        raw_cands, _, _ = find_blob_candidates(
            data, bg, std,
            edge_mask_max_x=edge_mask_max_x,
            border_margin_px=border_margin_px,
            border_max_length_px=border_max_length_px,
            left_edge_max_col=left_edge_max_col,
            left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
            long_thin_bypass=bypass_cfg,
            bleed_veto=bleed_veto_cfg,
        )
        for c in raw_cands:
            lbl = int(c.get("label", -1))
            if lbl in frag_ids:
                member_centroids.append(c["centroid"])

    # A) Overview PNG
    scale_ds = max(1, int(math.ceil(max(h_img, w_img) / 1000.0)))
    h_ds = h_img // scale_ds
    w_ds = w_img // scale_ds
    frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
    vmin_ov = float(np.percentile(frame_ds, 1.0))
    vmax_ov = float(np.percentile(frame_ds, 99.5))

    fig_ov, ax_ov = plt.subplots(figsize=(9, 9), dpi=140)
    ax_ov.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin_ov, vmax=vmax_ov)

    # Thin, semi-transparent line from endpoints
    ax_ov.plot([ep0_c, ep1_c], [ep0_r, ep1_r], color="cyan", linewidth=1.2, alpha=0.6, label="Fitted Line")

    # Member centroids marked
    if member_centroids:
        m_rows = [m[0] for m in member_centroids]
        m_cols = [m[1] for m in member_centroids]
        ax_ov.scatter(m_cols, m_rows, s=36, facecolors="none", edgecolors="lime", linewidth=1.2, alpha=0.8, label=f"Members ({len(member_centroids)})")

    ax_ov.set_xlim(0, w_img)
    ax_ov.set_ylim(0, h_img)
    ax_ov.set_title(f"Overview: {cand_id} ({frame_name})\nEndpoints: ({ep0_r:.1f}, {ep0_c:.1f}) -> ({ep1_r:.1f}, {ep1_c:.1f}) | Length={length_px:.1f}px", fontsize=10)
    ax_ov.set_xlabel("Column (px)")
    ax_ov.set_ylabel("Row (px)")
    ax_ov.legend(loc="upper right", fontsize=8, framealpha=0.6)
    plt.tight_layout()

    ov_path = out_dir / f"{cand_id}_overview.png"
    plt.savefig(ov_path, bbox_inches="tight", dpi=140)
    plt.close(fig_ov)
    saved_files.append(ov_path)

    # B) 3 Zoom PNGs at 25%, 50%, 75%
    zoom_fractions = [(0.25, "25"), (0.50, "50"), (0.75, "75")]
    for frac, pct_str in zoom_fractions:
        zf_r = ep0_r + frac * dr
        zf_c = ep0_c + frac * dc

        crop, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub, zf_r, zf_c, 400)
        norm_zoom, vmin_z, vmax_z = make_asinh_norm(crop, vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0)

        # 1. Zoom with thin, semi-transparent overlay
        fig_z, ax_z = plt.subplots(figsize=(6, 6), dpi=140)
        if norm_zoom is not None:
            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], norm=norm_zoom)
        else:
            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=vmin_z, vmax=vmax_z)

        # Line overlay
        ax_z.plot([ep0_c, ep1_c], [ep0_r, ep1_r], color="cyan", linewidth=1.1, alpha=0.55, linestyle="--", label="Line")

        # In-bounds member centroids
        zoom_mems = [m for m in member_centroids if (r0_cr <= m[0] <= r1_cr and c0_cr <= m[1] <= c1_cr)]
        if zoom_mems:
            zm_rows = [m[0] for m in zoom_mems]
            zm_cols = [m[1] for m in zoom_mems]
            ax_z.scatter(zm_cols, zm_rows, s=50, facecolors="none", edgecolors="lime", linewidth=1.2, alpha=0.75, label="Member Centroid")

        ax_z.set_xlim(c0_cr, c1_cr)
        ax_z.set_ylim(r0_cr, r1_cr)
        ax_z.set_title(f"{cand_id} Zoom @ {pct_str}% Length\nCenter: (r={zf_r:.1f}, c={zf_c:.1f}) | asinh stretch", fontsize=9.5)
        ax_z.set_xlabel("Column (px)")
        ax_z.set_ylabel("Row (px)")
        if zoom_mems:
            ax_z.legend(loc="upper right", fontsize=8, framealpha=0.5)
        plt.tight_layout()

        z_path = out_dir / f"{cand_id}_zoom_{pct_str}pct.png"
        plt.savefig(z_path, bbox_inches="tight", dpi=140)
        plt.close(fig_z)
        saved_files.append(z_path)

        # 2. Clean zoom (no overlays)
        fig_zc, ax_zc = plt.subplots(figsize=(6, 6), dpi=140)
        if norm_zoom is not None:
            ax_zc.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], norm=norm_zoom)
        else:
            ax_zc.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=vmin_z, vmax=vmax_z)
        ax_zc.set_xlim(c0_cr, c1_cr)
        ax_zc.set_ylim(r0_cr, r1_cr)
        ax_zc.set_title(f"{cand_id} Zoom @ {pct_str}% Length (Clean)", fontsize=9.5)
        ax_zc.set_xlabel("Column (px)")
        ax_zc.set_ylabel("Row (px)")
        plt.tight_layout()

        zc_path = out_dir / f"{cand_id}_zoom_{pct_str}pct_clean.png"
        plt.savefig(zc_path, bbox_inches="tight", dpi=140)
        plt.close(fig_zc)
        saved_files.append(zc_path)

    # C) Strip PNG (15 px width on line stacked with two parallel strips offset +-25 px, same stretch)
    t_vals = np.arange(0.0, length_px + 1.0, 1.0)
    w_strip = np.arange(-7.0, 8.0, 1.0)  # 15 px width (-7 to +7)

    # Grid for center strip (W_offset = 0)
    T_g, W_g = np.meshgrid(t_vals, w_strip)
    # Transverse displacement: perpendicular vector is (-uc, ur)
    Y_c = (ep0_r + T_g * ur) - W_g * uc
    X_c = (ep0_c + T_g * uc) + W_g * ur
    strip_center = map_coordinates(sub, [Y_c.ravel(), X_c.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_c.shape)

    # Left control strip (offset -25 px)
    W_g_m25 = W_g - 25.0
    Y_m25 = (ep0_r + T_g * ur) - W_g_m25 * uc
    X_m25 = (ep0_c + T_g * uc) + W_g_m25 * ur
    strip_m25 = map_coordinates(sub, [Y_m25.ravel(), X_m25.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_m25.shape)

    # Right control strip (offset +25 px)
    W_g_p25 = W_g + 25.0
    Y_p25 = (ep0_r + T_g * ur) - W_g_p25 * uc
    X_p25 = (ep0_c + T_g * uc) + W_g_p25 * ur
    strip_p25 = map_coordinates(sub, [Y_p25.ravel(), X_p25.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_p25.shape)

    # Downsample strip horizontally if long
    n_t = len(t_vals)
    bin_f = max(1, int(math.ceil(n_t / 1200.0)))
    n_trim = (n_t // bin_f) * bin_f
    if bin_f > 1:
        s_m25_disp = strip_m25[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
        s_c_disp = strip_center[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
        s_p25_disp = strip_p25[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
    else:
        s_m25_disp = strip_m25
        s_c_disp = strip_center
        s_p25_disp = strip_p25

    # Common stretch across strips
    all_strip_pixels = np.concatenate([strip_center.ravel(), strip_m25.ravel(), strip_p25.ravel()])
    norm_strip, vmin_s, vmax_s = make_asinh_norm(all_strip_pixels, vmin_pct=3.0, vmax_pct=99.5, linear_width=2.5)

    # Perpendicular profile across -40 to +40 px
    w_prof = np.arange(-40.0, 41.0, 1.0)
    T_p_grid, W_p_grid = np.meshgrid(t_vals, w_prof)
    Y_prof = (ep0_r + T_p_grid * ur) - W_p_grid * uc
    X_prof = (ep0_c + T_p_grid * uc) + W_p_grid * ur
    prof_map = map_coordinates(sub, [Y_prof.ravel(), X_prof.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_prof.shape)
    perp_profile = np.median(prof_map, axis=1)

    fig_s = plt.figure(figsize=(13, 7), dpi=140)
    gs = fig_s.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.2, 2.0], hspace=0.35)

    ax_s1 = fig_s.add_subplot(gs[0])
    ax_s2 = fig_s.add_subplot(gs[1])
    ax_s3 = fig_s.add_subplot(gs[2])
    ax_s4 = fig_s.add_subplot(gs[3])

    im_kwargs = {"cmap": "gray", "origin": "lower", "aspect": "auto"}
    if norm_strip is not None:
        im_kwargs["norm"] = norm_strip
    else:
        im_kwargs["vmin"] = vmin_s
        im_kwargs["vmax"] = vmax_s

    ax_s1.imshow(s_m25_disp, **im_kwargs)
    ax_s1.set_title(f"Parallel Control Strip: offset = -25 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold")
    ax_s1.set_ylabel("w [-32..-18]", fontsize=7.5)

    ax_s2.imshow(s_c_disp, **im_kwargs)
    ax_s2.set_title(f"On-Line Streak Strip: offset = 0 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold", color="blue")
    ax_s2.set_ylabel("w [-7..+7]", fontsize=7.5)

    ax_s3.imshow(s_p25_disp, **im_kwargs)
    ax_s3.set_title(f"Parallel Control Strip: offset = +25 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold")
    ax_s3.set_ylabel("w [+18..+32]", fontsize=7.5)
    ax_s3.set_xlabel(f"Along-line distance t ({'binned x' + str(bin_f) if bin_f > 1 else 'px'})", fontsize=8)

    # Panel 4: Perpendicular profile (median across line)
    ax_s4.plot(w_prof, perp_profile, color="black", lw=1.5, label="Median Along-Line Profile")
    ax_s4.axvline(0.0, color="blue", lw=1.2, linestyle="--", alpha=0.7, label="Line Center (0 px)")
    ax_s4.axvspan(-7.0, 7.0, color="blue", alpha=0.08, label="On-Line Width (15 px)")
    ax_s4.axvline(-25.0, color="red", lw=1.0, linestyle=":", alpha=0.7, label="Control Offsets (±25 px)")
    ax_s4.axvline(25.0, color="red", lw=1.0, linestyle=":", alpha=0.7)
    ax_s4.set_title("Transverse Profile (Median Along Line vs Perpendicular Offset w)", fontsize=9.5, fontweight="bold")
    ax_s4.set_xlabel("Perpendicular Offset w (px)")
    ax_s4.set_ylabel("Median ADU")
    ax_s4.grid(True, linestyle=":", alpha=0.6)
    ax_s4.legend(loc="upper right", fontsize=7.5, framealpha=0.7)

    fig_s.suptitle(f"{cand_id} Strip Verification ({frame_name}) | Length={length_px:.1f}px | Angle={float(row['angle_deg']):.1f}°", fontsize=11, fontweight="bold")

    strip_path = out_dir / f"{cand_id}_strip.png"
    plt.savefig(strip_path, bbox_inches="tight", dpi=140)
    plt.close(fig_s)
    saved_files.append(strip_path)

    return saved_files


def process_duplicate_target(row_270, row_10323, fits_path, out_dir):
    """Generate overview and 300x300 zoom for frame 20221026-0059 (candidates 270 and 10323)."""
    saved_files = []
    frame_name = "20221026-0059.fits"

    data, header = load_fits(fits_path)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    h_img, w_img = data.shape

    # 1. Overview PNG with both candidates drawn in different colors
    scale_ds = max(1, int(math.ceil(max(h_img, w_img) / 1000.0)))
    h_ds = h_img // scale_ds
    w_ds = w_img // scale_ds
    frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
    vmin_ov = float(np.percentile(frame_ds, 1.0))
    vmax_ov = float(np.percentile(frame_ds, 99.5))

    fig_ov, ax_ov = plt.subplots(figsize=(9, 9), dpi=140)
    ax_ov.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin_ov, vmax=vmax_ov)

    # Candidate 270 (Color 1: Red/Coral)
    r270_c0, r270_c1 = float(row_270["bbox_c0"]), float(row_270["bbox_c1"])
    r270_r0, r270_r1 = float(row_270["bbox_r0"]), float(row_270["bbox_r1"])
    r270_cy, r270_cx = float(row_270["centroid_row"]), float(row_270["centroid_col"])
    # 270 is a long streak across the frame
    ax_ov.plot([r270_c0, r270_c1], [r270_r1, r270_r0], color="red", linewidth=1.2, alpha=0.6, label="Candidate 270 (streak, L=4663px)")
    ax_ov.text(r270_cx + 40, r270_cy, "270", color="red", fontsize=9, fontweight="bold", alpha=0.85)

    # Candidate 10323 (Color 2: Cyan)
    r10_c0, r10_c1 = float(row_10323["bbox_c0"]), float(row_10323["bbox_c1"])
    r10_r0, r10_r1 = float(row_10323["bbox_r0"]), float(row_10323["bbox_r1"])
    r10_cy, r10_cx = float(row_10323["centroid_row"]), float(row_10323["centroid_col"])
    rect_10 = patches.Rectangle(
        (r10_c0, r10_r0), max(20.0, r10_c1 - r10_c0), max(20.0, r10_r1 - r10_r0),
        linewidth=1.3, edgecolor="cyan", facecolor="none", alpha=0.85, label="Candidate 10323 (unlabeled, L=13px)"
    )
    ax_ov.add_patch(rect_10)
    ax_ov.text(r10_cx + 40, r10_cy, "10323", color="cyan", fontsize=9, fontweight="bold", alpha=0.9)

    ax_ov.set_xlim(0, w_img)
    ax_ov.set_ylim(0, h_img)
    ax_ov.set_title(f"Overview: 20221026-0059.fits | Candidates 270 vs 10323 (Duplicate Pair)\nCandidate 270 (Red) vs Candidate 10323 (Cyan)", fontsize=10)
    ax_ov.set_xlabel("Column (px)")
    ax_ov.set_ylabel("Row (px)")
    ax_ov.legend(loc="upper right", fontsize=8, framealpha=0.6)
    plt.tight_layout()

    ov_path = out_dir / "20221026-0059_270_vs_10323_overview.png"
    plt.savefig(ov_path, bbox_inches="tight", dpi=140)
    plt.close(fig_ov)
    saved_files.append(ov_path)

    # 2. Zoom PNG of ~300x300 px centered between them
    # Candidate 10323 is at (1980.0, 415.33), and Candidate 270's streak passes directly through at d_perp=1.2 px.
    # Centering at (1980.0, 415.33) places both candidates right in the center of the 300x300 zoom!
    zoom_center_r = r10_cy
    zoom_center_c = r10_cx

    crop, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub, zoom_center_r, zoom_center_c, 300)
    norm_z, vmin_z, vmax_z = make_asinh_norm(crop, vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0)

    # A) Zoom with thin, semi-transparent overlays
    fig_z, ax_z = plt.subplots(figsize=(6, 6), dpi=140)
    if norm_z is not None:
        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], norm=norm_z)
    else:
        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=vmin_z, vmax=vmax_z)

    # Candidate 270 streak line segment passing through zoom (Color 1: Red)
    ang_rad = math.radians(float(row_270["angle_from_vertical"]))
    # In 20221026-0059, the streak leans from upper-left (high row, low col) to lower-right (low row, high col):
    # dr/dc ~ -cos(ang)/sin(ang)
    ur_270 = -math.cos(ang_rad)
    uc_270 = math.sin(ang_rad)
    p0_r = zoom_center_r - 200.0 * ur_270
    p0_c = zoom_center_c - 200.0 * uc_270
    p1_r = zoom_center_r + 200.0 * ur_270
    p1_c = zoom_center_c + 200.0 * uc_270
    ax_z.plot([p0_c, p1_c], [p0_r, p1_r], color="red", linewidth=1.1, alpha=0.55, linestyle="--", label="Streak 270")

    # Candidate 10323 bounding box (Color 2: Cyan)
    rect_z10 = patches.Rectangle(
        (r10_c0, r10_r0), r10_c1 - r10_c0, r10_r1 - r10_r0,
        linewidth=1.2, edgecolor="cyan", facecolor="none", alpha=0.75, label="Cand 10323 Bbox"
    )
    ax_z.add_patch(rect_z10)
    ax_z.scatter([r10_cx], [r10_cy], s=40, facecolors="none", edgecolors="cyan", linewidth=1.1, alpha=0.75)
    ax_z.text(r10_c1 + 5, r10_r1, "10323", color="cyan", fontsize=8.5, fontweight="bold", alpha=0.85)
    ax_z.text(c0_cr + 10, r1_cr - 25, "Streak 270", color="red", fontsize=8.5, fontweight="bold", alpha=0.85)

    ax_z.set_xlim(c0_cr, c1_cr)
    ax_z.set_ylim(r0_cr, r1_cr)
    ax_z.set_title("20221026-0059 Zoom (300x300 px) | 270 vs 10323\nd_perp = 1.2 px, overlap = 100% | asinh stretch", fontsize=9.5)
    ax_z.set_xlabel("Column (px)")
    ax_z.set_ylabel("Row (px)")
    ax_z.legend(loc="lower right", fontsize=8, framealpha=0.5)
    plt.tight_layout()

    z_path = out_dir / "20221026-0059_270_vs_10323_zoom.png"
    plt.savefig(z_path, bbox_inches="tight", dpi=140)
    plt.close(fig_z)
    saved_files.append(z_path)

    # B) Clean zoom (no overlays)
    fig_zc, ax_zc = plt.subplots(figsize=(6, 6), dpi=140)
    if norm_z is not None:
        ax_zc.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], norm=norm_z)
    else:
        ax_zc.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=vmin_z, vmax=vmax_z)
    ax_zc.set_xlim(c0_cr, c1_cr)
    ax_zc.set_ylim(r0_cr, r1_cr)
    ax_zc.set_title("20221026-0059 Zoom (300x300 px, Clean)", fontsize=9.5)
    ax_zc.set_xlabel("Column (px)")
    ax_zc.set_ylabel("Row (px)")
    plt.tight_layout()

    zc_path = out_dir / "20221026-0059_270_vs_10323_zoom_clean.png"
    plt.savefig(zc_path, bbox_inches="tight", dpi=140)
    plt.close(fig_zc)
    saved_files.append(zc_path)

    return saved_files


def main():
    print("=" * 80)
    print("TASK G9b: SAVE CHECK IMAGES FOR 2 LINES AND 1 DUPLICATE PAIR")
    print("=" * 80)

    # Load config for blob detection member lookup
    cfg_path = base_dir / "src/config.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    # 1. Read candidates_v2.csv only
    input_csv = base_dir / "data/d0b/candidates_v2.csv"
    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        sys.exit(1)

    df_v2 = pd.read_csv(input_csv)
    print(f"Loaded candidates_v2.csv ({len(df_v2)} candidates)")

    out_dir = base_dir / "data/d0b/viz_g9b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Check FITS file discovery for target frames
    target_frames = {
        "20230615-0006": find_fits_file("20230615-0006"),
        "20240502-0026": find_fits_file("20240502-0026"),
        "20221026-0059": find_fits_file("20221026-0059"),
    }

    print("\nFITS File Search Status:")
    print("-" * 50)
    for stem, p in target_frames.items():
        status_str = f"FOUND: {p}" if (p and p.exists()) else "NOT FOUND"
        print(f"  Frame {stem}.fits: {status_str}")

    all_saved_files = []

    # 2. Target (a): Line 20230615-0006_L1
    print("\nProcessing Target (a): Line 20230615-0006_L1...")
    line_a = df_v2[df_v2["cand_v2_id"] == "20230615-0006_L1"]
    if not line_a.empty and target_frames["20230615-0006"]:
        files_a = process_line_target(line_a.iloc[0], target_frames["20230615-0006"], out_dir, cfg)
        all_saved_files.extend(files_a)
    else:
        print("  Skipping: row or FITS missing.")

    # 3. Target (b): Line 20240502-0026_L1
    print("\nProcessing Target (b): Line 20240502-0026_L1...")
    line_b = df_v2[df_v2["cand_v2_id"] == "20240502-0026_L1"]
    if not line_b.empty and target_frames["20240502-0026"]:
        files_b = process_line_target(line_b.iloc[0], target_frames["20240502-0026"], out_dir, cfg)
        all_saved_files.extend(files_b)
    else:
        print("  Skipping: row or FITS missing.")

    # 4. Target (c): Frame 20221026-0059.fits, candidates 270 and 10323
    print("\nProcessing Target (c): 20221026-0059.fits (candidates 270 and 10323)...")
    frame_c_df = df_v2[df_v2["frame_name"] == "20221026-0059.fits"]
    cand_270 = frame_c_df[frame_c_df["candidate_id"] == 270.0]
    cand_10323 = frame_c_df[frame_c_df["candidate_id"] == 10323.0]

    if not cand_270.empty and not cand_10323.empty and target_frames["20221026-0059"]:
        files_c = process_duplicate_target(cand_270.iloc[0], cand_10323.iloc[0], target_frames["20221026-0059"], out_dir)
        all_saved_files.extend(files_c)
    else:
        print("  Skipping: candidates or FITS missing.")

    print("\n" + "=" * 80)
    print(f"Summary: Successfully saved {len(all_saved_files)} images to data/d0b/viz_g9b/")
    print("=" * 80)
    for p in all_saved_files:
        print(f"  {p.relative_to(base_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
