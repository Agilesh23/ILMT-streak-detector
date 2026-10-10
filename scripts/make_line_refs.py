# TASK D0b-ref: reference images for line checks
"""
Generate 4-panel reference diagnostic images for candidate lines in data/d0b/lines.csv:
1. Plain frame downsampled, robust stretch (percentiles 1 and 99.5), line endpoints marked on border.
2. Straightened strip (±60 px) along fitted line, bilinear interpolation, Gaussian blur sigma 1.5,
   robust stretch (percentiles 5 and 99.5), binned along length to <=2000 px, inlier extent ticked on top border.
3. Control strip shifted +150 px perpendicular, same stretch as panel 2.
4. Perpendicular profile (-60 to +60 px) for real strip and control strip on one axis.
"""

import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
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
from scripts.diag_line_group import (
    find_lines_ransac,
    PERP_TOL_PX,
    MIN_INLIERS,
    MIN_SPAN_PX,
    MAX_LINES,
    RANSAC_ITERS,
)


def get_frame_boundary_intersections(mean_y: float, mean_x: float, uy: float, ux: float, h: int, w: int):
    """
    Compute intersection points of the infinite line (mean_y + t*uy, mean_x + t*ux)
    with the 4 outer image borders [0, h] x [0, w].
    Returns (t_b1, t_b2, (y_b1, x_b1), (y_b2, x_b2)) where t_b1 < t_b2.
    """
    ts = []
    # y = 0
    if abs(uy) > 1e-7:
        t = (0.0 - mean_y) / uy
        x = mean_x + t * ux
        if 0.0 <= x <= w:
            ts.append((t, 0.0, x, "bottom"))
    # y = h
    if abs(uy) > 1e-7:
        t = (float(h) - mean_y) / uy
        x = mean_x + t * ux
        if 0.0 <= x <= w:
            ts.append((t, float(h), x, "top"))
    # x = 0
    if abs(ux) > 1e-7:
        t = (0.0 - mean_x) / ux
        y = mean_y + t * uy
        if 0.0 <= y <= h:
            ts.append((t, y, 0.0, "left"))
    # x = w
    if abs(ux) > 1e-7:
        t = (float(w) - mean_x) / ux
        y = mean_y + t * uy
        if 0.0 <= y <= h:
            ts.append((t, y, float(w), "right"))

    # Deduplicate points that are close
    unique_ts = []
    for item in sorted(ts, key=lambda x: x[0]):
        if not unique_ts or abs(item[0] - unique_ts[-1][0]) > 1.0:
            unique_ts.append(item)

    if len(unique_ts) >= 2:
        return unique_ts[0], unique_ts[-1]
    return None, None


def main():
    config_path = base_dir / "src/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})
    edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
    border_margin_px = det_cfg.get("border_margin_px", 15)
    border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
    left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
    left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
    bypass_cfg = det_cfg.get("long_thin_bypass", None)
    bleed_veto_cfg = det_cfg.get("bleed_veto", None)

    # Read data/d0b/lines.csv
    lines_csv_path = base_dir / "data/d0b/lines.csv"
    if not lines_csv_path.exists():
        print(f"Error: {lines_csv_path} does not exist.")
        sys.exit(1)

    df_lines = pd.read_csv(lines_csv_path)

    # Discover FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        fits_by_stem[p.stem.lower()] = p

    out_dir = base_dir / "outputs/d0b/ref"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    # Process each frame in lines.csv
    for stem, group in df_lines.groupby("frame", sort=False):
        fits_file = fits_by_stem.get(str(stem).lower())
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS file for {stem} not found.")
            continue

        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = data.shape

        # Rebuild candidates exactly as diag_line_group.py does
        raw_candidates, _, _ = find_blob_candidates(
            data,
            bg,
            std,
            edge_mask_max_x=edge_mask_max_x,
            border_margin_px=border_margin_px,
            border_max_length_px=border_max_length_px,
            left_edge_max_col=left_edge_max_col,
            left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
            long_thin_bypass=bypass_cfg,
            bleed_veto=bleed_veto_cfg,
        )

        # Reproduce lines identically using fixed seed 42
        rng = np.random.default_rng(42)
        lines = find_lines_ransac(raw_candidates, rng=rng)
        lines_by_id = {l["line_id"]: l for l in lines}

        # Process each target line in lines.csv for this frame
        for _, row in group.iterrows():
            line_id = int(row["line_id"])
            n_inliers = int(row["n_inliers"])
            span_px = float(row["span_px"])

            if line_id not in lines_by_id:
                print(f"Warning: Line {line_id} in {stem} not found by RANSAC reproduce.")
                continue

            line = lines_by_id[line_id]
            mean_y = line["mean_y"]
            mean_x = line["mean_x"]
            uy = line["uy"]
            ux = line["ux"]
            t_min = line["t_min"]
            t_max = line["t_max"]

            # Intersect with frame boundary
            pt_b1, pt_b2 = get_frame_boundary_intersections(mean_y, mean_x, uy, ux, h_img, w_img)
            if pt_b1 is not None and pt_b2 is not None:
                t_b1 = pt_b1[0]
                t_b2 = pt_b2[0]
            else:
                t_b1 = t_min - 300.0
                t_b2 = t_max + 300.0

            # 1. Plain frame downsampled
            scale_ds = max(1, int(np.ceil(max(h_img, w_img) / 1000.0)))
            h_ds = h_img // scale_ds
            w_ds = w_img // scale_ds
            frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(
                h_ds, scale_ds, w_ds, scale_ds
            ).mean(axis=(1, 3))

            vmin_frame = float(np.percentile(frame_ds, 1.0))
            vmax_frame = float(np.percentile(frame_ds, 99.5))

            # 2. Straightened strip setup
            t_strip_start = max(t_min - 300.0, t_b1)
            t_strip_end = min(t_max + 300.0, t_b2)
            if t_strip_end <= t_strip_start:
                t_strip_start = t_min - 100.0
                t_strip_end = t_max + 100.0

            t_vals = np.arange(t_strip_start, t_strip_end + 1.0, 1.0)
            w_vals = np.arange(-60.0, 61.0, 1.0)  # 121 px width
            T_grid, W_grid = np.meshgrid(t_vals, w_vals)

            # Sample real strip: line at W = 0
            # Centerline: mean_y + T*uy, mean_x + T*ux. Transverse offset: -W*ux, +W*uy
            Y_real = mean_y + T_grid * uy - W_grid * ux
            X_real = mean_x + T_grid * ux + W_grid * uy

            strip_real_raw = map_coordinates(sub, [Y_real.ravel(), X_real.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_real.shape)
            strip_real_blur = gaussian_filter(strip_real_raw, sigma=1.5)

            vmin_strip = float(np.percentile(strip_real_blur, 5.0))
            vmax_strip = float(np.percentile(strip_real_blur, 99.5))

            # Bin along length so width is at most 2000 px
            n_len = len(t_vals)
            bin_f = max(1, int(np.ceil(n_len / 2000.0)))
            n_trim = (n_len // bin_f) * bin_f
            if bin_f > 1:
                strip_real_disp = strip_real_blur[:, :n_trim].reshape(121, n_trim // bin_f, bin_f).mean(axis=2)
            else:
                strip_real_disp = strip_real_blur

            # Inlier extent positions on binned horizontal axis
            col_inlier_start = (t_min - t_strip_start) / float(bin_f)
            col_inlier_end = (t_max - t_strip_start) / float(bin_f)

            # 3. Control strip shifted +150 px perpendicular
            W_ctrl_grid = W_grid + 150.0
            Y_ctrl = mean_y + T_grid * uy - W_ctrl_grid * ux
            X_ctrl = mean_x + T_grid * ux + W_ctrl_grid * uy

            strip_ctrl_raw = map_coordinates(sub, [Y_ctrl.ravel(), X_ctrl.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_ctrl.shape)
            strip_ctrl_blur = gaussian_filter(strip_ctrl_raw, sigma=1.5)

            if bin_f > 1:
                strip_ctrl_disp = strip_ctrl_blur[:, :n_trim].reshape(121, n_trim // bin_f, bin_f).mean(axis=2)
            else:
                strip_ctrl_disp = strip_ctrl_blur

            # 4. Perpendicular profiles
            prof_real = np.mean(strip_real_raw, axis=1)
            prof_ctrl = np.mean(strip_ctrl_raw, axis=1)

            # Build Figure with GridSpec
            fig = plt.figure(figsize=(15, 12), dpi=140)
            gs = GridSpec(4, 2, figure=fig, height_ratios=[4.5, 4.5, 1.3, 1.3])

            title_main = f"{stem} | Line {line_id} | n_inliers={n_inliers} | span={span_px:.1f}px | angle={line['angle_deg']:.1f}°"
            fig.suptitle(title_main, fontsize=13, fontweight="bold", y=0.98)

            # Panel 1: Plain frame overview
            ax1 = fig.add_subplot(gs[0:2, 0])
            ax1.imshow(
                frame_ds,
                cmap="gray",
                origin="lower",
                extent=[0, w_img, 0, h_img],
                vmin=vmin_frame,
                vmax=vmax_frame,
            )
            ax1.set_title("1. Plain Frame (endpoints ticked on border)", fontsize=10, fontweight="bold")
            ax1.set_xlim(0, w_img)
            ax1.set_ylim(0, h_img)
            ax1.set_xlabel("X (px)", fontsize=9)
            ax1.set_ylabel("Y (px)", fontsize=9)

            # Mark only the two boundary intersection points on border with small ticks
            tick_len = 50.0
            if pt_b1 is not None and pt_b2 is not None:
                for pt in [pt_b1, pt_b2]:
                    _, y_b, x_b, border = pt
                    if border == "bottom":
                        ax1.plot([x_b, x_b], [0, tick_len], color="red", linewidth=2.5)
                    elif border == "top":
                        ax1.plot([x_b, x_b], [h_img - tick_len, h_img], color="red", linewidth=2.5)
                    elif border == "left":
                        ax1.plot([0, tick_len], [y_b, y_b], color="red", linewidth=2.5)
                    elif border == "right":
                        ax1.plot([w_img - tick_len, w_img], [y_b, y_b], color="red", linewidth=2.5)

            # Panel 4: Perpendicular profile (upper right)
            ax4 = fig.add_subplot(gs[0:2, 1])
            ax4.plot(w_vals, prof_real, color="cyan", linewidth=2.0, label="Real Strip (centered on line)")
            ax4.plot(w_vals, prof_ctrl, color="orange", linewidth=1.8, linestyle="--", label="Control Strip (+150px offset)")
            ax4.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            ax4.set_title("4. Perpendicular Profile (Mean ADU vs Offset)", fontsize=10, fontweight="bold")
            ax4.set_xlabel("Perpendicular offset (px)", fontsize=9)
            ax4.set_ylabel("Mean brightness (ADU)", fontsize=9)
            ax4.set_xlim(-60, 60)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

            # Panel 2: Straightened strip
            ax2 = fig.add_subplot(gs[2, :])
            ax2.imshow(
                strip_real_disp,
                cmap="gray",
                origin="lower",
                extent=[0, strip_real_disp.shape[1], -60, 60],
                vmin=vmin_strip,
                vmax=vmax_strip,
                aspect="auto",
            )
            ax2.set_title("2. Straightened Strip (±60px, Gaussian σ=1.5, binned along length)", fontsize=10, fontweight="bold")
            ax2.set_ylabel("Offset (px)", fontsize=8)
            ax2.set_xlabel("Length along line (px)", fontsize=8)
            # Ticks on top border for inlier extent
            ax2.plot([col_inlier_start, col_inlier_start], [45, 60], color="red", linewidth=2.5)
            ax2.plot([col_inlier_end, col_inlier_end], [45, 60], color="red", linewidth=2.5)

            # Panel 3: Control strip
            ax3 = fig.add_subplot(gs[3, :])
            ax3.imshow(
                strip_ctrl_disp,
                cmap="gray",
                origin="lower",
                extent=[0, strip_ctrl_disp.shape[1], -60, 60],
                vmin=vmin_strip,
                vmax=vmax_strip,
                aspect="auto",
            )
            ax3.set_title("3. Control Strip (+150px perpendicular offset, same stretch)", fontsize=10, fontweight="bold")
            ax3.set_ylabel("Offset (px)", fontsize=8)
            ax3.set_xlabel("Length along line (px)", fontsize=8)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out_file = out_dir / f"{stem}_L{line_id}_ref.png"
            fig.savefig(out_file)
            plt.close(fig)

            written_files.append(out_file)
            print(f"Written: {out_file}")

    print(f"\nTotal reference figures written: {len(written_files)}")


if __name__ == "__main__":
    main()
