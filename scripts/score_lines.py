# TASK D0b-score: whole-line brightness score
"""
Diagnostic script for whole-line brightness scoring and segment consistency analysis.
Scores candidate lines across full frame crossing:
- Strip sampling: w from -100 to +100 px, frame edge to edge, bilinear interpolation on data - bg.
- S(w0) = median over along-line bins (50 px) of mean over rows with |w - w0| <= 3.
- Signal = S(0); Background = S(w0) for w0 in {+-15, +-22, ..., +-99} (step 7).
- z_line = (Signal - mean(background)) / std(background).
- 8 equal-length segments: z_seg_1..z_seg_8 and n_segments_z_gt_3.
"""

import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
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
from scripts.make_line_refs import get_frame_boundary_intersections

# Hard-coded ground-truth verdicts
REAL_LINES = {
    ("20240209-0036", 1),
    ("20240211-0013", 1),
    ("20240216-0030", 1),
    ("20230615-0006", 1),
    ("20240502-0026", 1),
}

# Background offset grid: +-15, +-22, ..., +-99 (step 7)
POS_OFFSETS = list(range(15, 100, 7))  # [15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99]
NEG_OFFSETS = [-w for w in POS_OFFSETS]
BG_OFFSETS = sorted(NEG_OFFSETS + POS_OFFSETS)


def compute_S(strip: np.ndarray, w_vals: np.ndarray, w0: float, bin_len: int = 50) -> float:
    """
    Compute S(w0) = median over along-line bins (bin length 50 px)
    of the mean over rows with |w - w0| <= 3.
    """
    row_mask = np.abs(w_vals - w0) <= 3.0
    if not np.any(row_mask):
        return 0.0

    # Mean over rows with |w - w0| <= 3 across each along-line column
    col_means = np.mean(strip[row_mask, :], axis=0)  # shape (n_cols,)
    n_cols = len(col_means)
    if n_cols == 0:
        return 0.0

    n_bins = max(1, n_cols // bin_len)
    bin_means = []
    for b in range(n_bins):
        c0 = b * bin_len
        c1 = min((b + 1) * bin_len, n_cols) if b == n_bins - 1 else (b + 1) * bin_len
        if c1 > c0:
            bin_means.append(float(np.mean(col_means[c0:c1])))

    return float(np.median(bin_means)) if bin_means else 0.0


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

    # Search for FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        fits_by_stem[p.stem.lower()] = p

    out_csv_dir = base_dir / "data/d0b"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = base_dir / "outputs/d0b/score"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Process lines grouped by frame
    for stem, group in df_lines.groupby("frame", sort=False):
        fits_file = fits_by_stem.get(str(stem).lower())
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS file for {stem} not found.")
            continue

        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = data.shape

        # Rebuild candidates identically to make_line_refs.py / diag_line_group.py
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

        for _, row in group.iterrows():
            line_id = int(row["line_id"])
            csv_n_inliers = int(row["n_inliers"])
            csv_span_px = float(row["span_px"])
            csv_rms_px = float(row["rms_residual_px"])

            if line_id not in lines_by_id:
                print(f"Warning: Line {line_id} in {stem} not found by RANSAC.")
                continue

            line = lines_by_id[line_id]
            mean_y = line["mean_y"]
            mean_x = line["mean_x"]
            uy = line["uy"]
            ux = line["ux"]

            # Ground truth verdict
            verdict = "REAL" if (str(stem), line_id) in REAL_LINES else "FALSE"

            # Frame boundary intersections: full line across frame edge to edge
            pt_b1, pt_b2 = get_frame_boundary_intersections(mean_y, mean_x, uy, ux, h_img, w_img)
            if pt_b1 is not None and pt_b2 is not None:
                t_start = pt_b1[0]
                t_end = pt_b2[0]
            else:
                t_start = line["t_min"]
                t_end = line["t_max"]

            # Perpendicular offsets w from -100 to +100 px (step 1 px, 201 rows)
            w_vals = np.arange(-100.0, 101.0, 1.0)
            t_vals = np.arange(t_start, t_end + 1.0, 1.0)
            T_grid, W_grid = np.meshgrid(t_vals, w_vals)

            # Sample strip along the full line via bilinear interpolation on sub (data - bg)
            # Centerline: mean_y + T*uy, mean_x + T*ux; Transverse: -W*ux, +W*uy
            Y_strip = mean_y + T_grid * uy - W_grid * ux
            X_strip = mean_x + T_grid * ux + W_grid * uy

            strip = map_coordinates(
                sub,
                [Y_strip.ravel(), X_strip.ravel()],
                order=1,
                mode="constant",
                cval=0.0,
            ).reshape(Y_strip.shape)

            # Whole-line signal and background
            signal = compute_S(strip, w_vals, 0.0, bin_len=50)
            bg_vals = [compute_S(strip, w_vals, w0, bin_len=50) for w0 in BG_OFFSETS]
            mu_bg = float(np.mean(bg_vals))
            sigma_bg = float(np.std(bg_vals))
            z_line = float((signal - mu_bg) / sigma_bg) if sigma_bg > 1e-6 else 0.0

            # 8 equal-length segments along the frame crossing
            n_cols = strip.shape[1]
            seg_edges = np.linspace(0, n_cols, 9, dtype=int)
            z_segs = []

            for s in range(8):
                c0 = seg_edges[s]
                c1 = seg_edges[s + 1]
                seg_strip = strip[:, c0:c1]
                sig_seg = compute_S(seg_strip, w_vals, 0.0, bin_len=50)
                bg_segs = [compute_S(seg_strip, w_vals, w0, bin_len=50) for w0 in BG_OFFSETS]
                mu_bg_seg = float(np.mean(bg_segs))
                sigma_bg_seg = float(np.std(bg_segs))
                z_s = float((sig_seg - mu_bg_seg) / sigma_bg_seg) if sigma_bg_seg > 1e-6 else 0.0
                z_segs.append(round(z_s, 2))

            n_gt_3 = sum(1 for z in z_segs if z > 3.0)

            # Dense profile S(w0) for diagnostic visualization
            dense_w0s = np.arange(-99.0, 100.0, 2.0)
            dense_S = [compute_S(strip, w_vals, w0, bin_len=50) for w0 in dense_w0s]

            # Generate diagnostic plot: outputs/d0b/score/<frame>_L<id>_score.png
            fig, (ax_prof, ax_bars) = plt.subplots(1, 2, figsize=(14, 5), dpi=130)

            fig.suptitle(
                f"{stem} | Line {line_id} | Verdict: {verdict} | z_line = {z_line:.2f} | "
                f"n_inliers = {csv_n_inliers} | span = {csv_span_px:.1f}px",
                fontsize=12,
                fontweight="bold",
            )

            # Panel A: Profile S(w0) vs w0
            ax_prof.plot(dense_w0s, dense_S, color="steelblue", linewidth=1.8, label="S(w0) profile")
            ax_prof.scatter(BG_OFFSETS, bg_vals, color="orange", s=25, zorder=4, label=f"BG set (μ={mu_bg:.2f}, σ={sigma_bg:.2f})")
            ax_prof.scatter([0.0], [signal], color="red", s=80, marker="*", zorder=5, label=f"Signal S(0) = {signal:.2f}")
            ax_prof.axvline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_prof.axhline(mu_bg, color="gray", linestyle=":", linewidth=1.0, label=f"BG mean = {mu_bg:.2f}")
            ax_prof.set_xlabel("Perpendicular offset w0 (px)", fontsize=10)
            ax_prof.set_ylabel("S(w0) brightness (ADU)", fontsize=10)
            ax_prof.set_title(f"Perpendicular Profile S(w0) | z_line = {z_line:.2f}", fontsize=11, fontweight="bold")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

            # Panel B: Bar chart of 8 segment z values
            seg_indices = np.arange(1, 9)
            bar_colors = ["forestgreen" if z > 3.0 else "indianred" for z in z_segs]
            bars = ax_bars.bar(seg_indices, z_segs, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.8)
            ax_bars.axhline(3.0, color="darkred", linestyle="--", linewidth=1.5, label="Threshold z = 3.0")
            ax_bars.axhline(0.0, color="gray", linewidth=0.8)

            for bar, z_val in zip(bars, z_segs):
                y_pos = max(z_val, 0) + 0.3
                ax_bars.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{z_val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            ax_bars.set_xlabel("Segment (1 to 8 along frame crossing)", fontsize=10)
            ax_bars.set_ylabel("Segment z-score", fontsize=10)
            ax_bars.set_title(f"Segment Consistency | {n_gt_3}/8 segments with z > 3.0", fontsize=11, fontweight="bold")
            ax_bars.set_xticks(seg_indices)
            ax_bars.grid(True, axis="y", alpha=0.3)
            ax_bars.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

            fig.tight_layout()
            out_img_path = out_img_dir / f"{stem}_L{line_id}_score.png"
            fig.savefig(out_img_path)
            plt.close(fig)

            row_record = {
                "frame": str(stem),
                "line_id": line_id,
                "verdict": verdict,
                "n_inliers": csv_n_inliers,
                "span_px": csv_span_px,
                "rms_residual_px": csv_rms_px,
                "z_line": round(z_line, 2),
                "z_seg_1": z_segs[0],
                "z_seg_2": z_segs[1],
                "z_seg_3": z_segs[2],
                "z_seg_4": z_segs[3],
                "z_seg_5": z_segs[4],
                "z_seg_6": z_segs[5],
                "z_seg_7": z_segs[6],
                "z_seg_8": z_segs[7],
                "n_segments_z_gt_3": n_gt_3,
            }
            results.append(row_record)

    # Save output CSV: data/d0b/line_scores.csv
    out_csv_path = out_csv_dir / "line_scores.csv"
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv_path, index=False)

    # Print summary table sorted by z_line descending
    df_sorted = df_out.sort_values(by="z_line", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("TASK D0b-score: WHOLE-LINE BRIGHTNESS SCORES (sorted by z_line descending)")
    print("=" * 90)
    print(f"{'Frame':<15} {'Line':<5} {'Verdict':<7} {'z_line':>8} {'Inliers':>8} {'Span (px)':>10} {'RMS (px)':>9} {'z>3 Segs':>9}")
    print("-" * 90)

    for _, r in df_sorted.iterrows():
        print(
            f"{r['frame']:<15} {r['line_id']:<5} {r['verdict']:<7} {r['z_line']:>8.2f} "
            f"{r['n_inliers']:>8d} {r['span_px']:>10.1f} {r['rms_residual_px']:>9.2f} {r['n_segments_z_gt_3']:>9d}"
        )

    print("=" * 90)

    # Compute separation metrics
    real_z = df_out[df_out["verdict"] == "REAL"]["z_line"].values
    false_z = df_out[df_out["verdict"] == "FALSE"]["z_line"].values

    max_z_false = float(np.max(false_z)) if len(false_z) > 0 else float("nan")
    min_z_real = float(np.min(real_z)) if len(real_z) > 0 else float("nan")

    print(f"\nLargest z among FALSE lines: {max_z_false:.2f}")
    print(f"Smallest z among REAL lines:  {min_z_real:.2f}")
    sep_gap = min_z_real - max_z_false
    print(f"Separation gap (min_REAL - max_FALSE): {sep_gap:.2f}")
    print(f"\nSaved CSV: {out_csv_path}")
    print(f"Saved {len(results)} diagnostic figures in {out_img_dir}")


if __name__ == "__main__":
    main()
