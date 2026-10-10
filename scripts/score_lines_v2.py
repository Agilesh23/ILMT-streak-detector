# TASK D0b-score2: whole-line score with peak-offset search
"""
Diagnostic script for whole-line brightness scoring with peak-offset search:
- Evaluates S(w0) over integer offsets w0 in [-20, +20] px to find peak_offset = argmax S(w0).
- Background offsets: w0 in {+-30, +-37, +-44, ..., +-93} (step 7).
- z_peak = (S(peak_offset) - mean(bg)) / std(bg).
- Evaluates peak search and z_peak across 8 equal-length segments along the line crossing.
- Computes linear slope of peak_offset_seg versus segment index (px per segment) for segments with z > 3.
- Joins z_line from data/d0b/line_scores.csv as z_line_fixed0.
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
from scripts.score_lines import compute_S

# Hard-coded ground-truth verdicts
REAL_LINES = {
    ("20240209-0036", 1),
    ("20240211-0013", 1),
    ("20240216-0030", 1),
    ("20230615-0006", 1),
    ("20240502-0026", 1),
}

# Peak search range: integer w0 from -20 to +20
PEAK_SEARCH_RANGE = list(range(-20, 21))

# Background offset grid: +-30, +-37, ..., +-93 (step 7)
POS_BG_OFFSETS = list(range(30, 94, 7))  # [30, 37, 44, 51, 58, 65, 72, 79, 86, 93]
NEG_BG_OFFSETS = [-w for w in POS_BG_OFFSETS]
BG_OFFSETS_V2 = sorted(NEG_BG_OFFSETS + POS_BG_OFFSETS)


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

    # Read data/d0b/line_scores.csv to join old z_line as z_line_fixed0
    scores_v1_path = base_dir / "data/d0b/line_scores.csv"
    fixed0_dict = {}
    if scores_v1_path.exists():
        df_v1 = pd.read_csv(scores_v1_path)
        for _, r in df_v1.iterrows():
            fixed0_dict[(str(r["frame"]), int(r["line_id"]))] = float(r["z_line"])

    # Search for FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        fits_by_stem[p.stem.lower()] = p

    out_csv_dir = base_dir / "data/d0b"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = base_dir / "outputs/d0b/score2"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for stem, group in df_lines.groupby("frame", sort=False):
        fits_file = fits_by_stem.get(str(stem).lower())
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS file for {stem} not found.")
            continue

        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = data.shape

        # Rebuild candidates identically
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

        # Reproduce lines identically using seed 42
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

            verdict = "REAL" if (str(stem), line_id) in REAL_LINES else "FALSE"
            z_line_fixed0 = fixed0_dict.get((str(stem), line_id), np.nan)

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

            # Sample strip along the full line via bilinear interpolation on data - bg
            Y_strip = mean_y + T_grid * uy - W_grid * ux
            X_strip = mean_x + T_grid * ux + W_grid * uy

            strip = map_coordinates(
                sub,
                [Y_strip.ravel(), X_strip.ravel()],
                order=1,
                mode="constant",
                cval=0.0,
            ).reshape(Y_strip.shape)

            # 1. Whole-line peak search over w0 in [-20, +20]
            search_S = {w0: compute_S(strip, w_vals, float(w0), bin_len=50) for w0 in PEAK_SEARCH_RANGE}
            peak_offset = max(search_S.keys(), key=lambda w0: search_S[w0])
            peak_val = search_S[peak_offset]

            # Background offsets {+-30, +-37, ..., +-93}
            bg_vals = [compute_S(strip, w_vals, float(w0), bin_len=50) for w0 in BG_OFFSETS_V2]
            mu_bg = float(np.mean(bg_vals))
            sigma_bg = float(np.std(bg_vals))
            z_peak = float((peak_val - mu_bg) / sigma_bg) if sigma_bg > 1e-6 else 0.0

            # 2. Segment analysis (8 equal-length segments along the line)
            n_cols = strip.shape[1]
            seg_edges = np.linspace(0, n_cols, 9, dtype=int)
            z_seg_peaks = []
            peak_offsets_seg = []

            for s in range(8):
                c0 = seg_edges[s]
                c1 = seg_edges[s + 1]
                seg_strip = strip[:, c0:c1]

                # Peak search in this segment
                seg_search_S = {w0: compute_S(seg_strip, w_vals, float(w0), bin_len=50) for w0 in PEAK_SEARCH_RANGE}
                p_off_seg = max(seg_search_S.keys(), key=lambda w0: seg_search_S[w0])
                p_val_seg = seg_search_S[p_off_seg]

                # Background in this segment
                bg_seg_vals = [compute_S(seg_strip, w_vals, float(w0), bin_len=50) for w0 in BG_OFFSETS_V2]
                mu_bg_s = float(np.mean(bg_seg_vals))
                sigma_bg_s = float(np.std(bg_seg_vals))
                z_s = float((p_val_seg - mu_bg_s) / sigma_bg_s) if sigma_bg_s > 1e-6 else 0.0

                z_seg_peaks.append(round(z_s, 2))
                peak_offsets_seg.append(int(p_off_seg))

            n_gt_3 = sum(1 for z in z_seg_peaks if z > 3.0)

            # Slope of peak_offset_seg vs segment index (1..8) for segments with z_seg_peak > 3
            seg_indices_valid = []
            offsets_valid = []
            for s_idx in range(8):
                if z_seg_peaks[s_idx] > 3.0:
                    seg_indices_valid.append(s_idx + 1)  # 1-based index
                    offsets_valid.append(peak_offsets_seg[s_idx])

            if len(seg_indices_valid) >= 3:
                poly = np.polyfit(seg_indices_valid, offsets_valid, 1)
                slope_val = round(float(poly[0]), 3)
            else:
                slope_val = np.nan

            # Dense S(w0) profile from -100 to +100 for visualization
            w0_dense = list(range(-100, 101))
            dense_S = [compute_S(strip, w_vals, float(w0), bin_len=50) for w0 in w0_dense]

            # 3. Visualization: outputs/d0b/score2/<frame>_L<id>_score2.png
            fig, (ax_prof, ax_bars) = plt.subplots(1, 2, figsize=(15, 5), dpi=130)

            slope_str = f"{slope_val:+.2f}" if not np.isnan(slope_val) else "NaN"
            fig.suptitle(
                f"{stem} | Line {line_id} ({verdict}) | z_peak = {z_peak:.2f} (z_fixed0 = {z_line_fixed0:.2f}) | "
                f"peak = {peak_offset:+d}px | slope = {slope_str} px/seg",
                fontsize=12,
                fontweight="bold",
            )

            # Panel A: S(w0) profile from -100 to +100
            ax_prof.plot(w0_dense, dense_S, color="steelblue", linewidth=1.8, label="S(w0) profile")
            ax_prof.scatter(BG_OFFSETS_V2, bg_vals, color="orange", s=25, zorder=4, label=f"BG set (μ={mu_bg:.2f}, σ={sigma_bg:.2f})")
            ax_prof.scatter([peak_offset], [peak_val], color="red", s=100, marker="*", zorder=5, label=f"Peak w0={peak_offset:+d} (S={peak_val:.2f})")
            ax_prof.axvline(peak_offset, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_prof.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, label="Center w0=0")
            ax_prof.axvspan(-20, 20, color="lightblue", alpha=0.15, label="Search band [±20px]")
            ax_prof.axhline(mu_bg, color="gray", linestyle=":", linewidth=1.0, label=f"BG mean = {mu_bg:.2f}")

            ax_prof.set_xlabel("Perpendicular offset w0 (px)", fontsize=10)
            ax_prof.set_ylabel("S(w0) brightness (ADU)", fontsize=10)
            ax_prof.set_xlim(-100, 100)
            ax_prof.set_title(f"S(w0) Profile [-100, +100] | z_peak = {z_peak:.2f}", fontsize=11, fontweight="bold")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend(loc="upper right", fontsize=8.0, framealpha=0.85)

            # Panel B: 8 segment bars with peak offsets above them
            seg_indices = np.arange(1, 9)
            bar_colors = ["forestgreen" if z > 3.0 else "indianred" for z in z_seg_peaks]
            bars = ax_bars.bar(seg_indices, z_seg_peaks, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.8)
            ax_bars.axhline(3.0, color="darkred", linestyle="--", linewidth=1.5, label="Threshold z = 3.0")
            ax_bars.axhline(0.0, color="gray", linewidth=0.8)

            for bar, z_val, p_off in zip(bars, z_seg_peaks, peak_offsets_seg):
                y_pos = max(z_val, 0) + 0.3
                ax_bars.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos,
                    f"w={p_off:+d}\n({z_val:.1f})",
                    ha="center",
                    va="bottom",
                    fontsize=8.0,
                    fontweight="bold",
                )

            ax_bars.set_xlabel("Segment (1 to 8 along frame crossing)", fontsize=10)
            ax_bars.set_ylabel("Segment z_seg_peak", fontsize=10)
            ax_bars.set_title(f"Segment Peak z-scores | {n_gt_3}/8 with z > 3 | slope: {slope_str} px/seg", fontsize=11, fontweight="bold")
            ax_bars.set_xticks(seg_indices)
            ax_bars.grid(True, axis="y", alpha=0.3)
            ax_bars.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

            fig.tight_layout()
            out_img_path = out_img_dir / f"{stem}_L{line_id}_score2.png"
            fig.savefig(out_img_path)
            plt.close(fig)

            row_record = {
                "frame": str(stem),
                "line_id": line_id,
                "verdict": verdict,
                "n_inliers": csv_n_inliers,
                "span_px": csv_span_px,
                "rms_residual_px": csv_rms_px,
                "z_line_fixed0": round(z_line_fixed0, 2) if not np.isnan(z_line_fixed0) else np.nan,
                "peak_offset": peak_offset,
                "z_peak": round(z_peak, 2),
                "n_segments_z_gt_3": n_gt_3,
                "slope_px_per_segment": slope_val,
                "z_seg_peak_1": z_seg_peaks[0],
                "z_seg_peak_2": z_seg_peaks[1],
                "z_seg_peak_3": z_seg_peaks[2],
                "z_seg_peak_4": z_seg_peaks[3],
                "z_seg_peak_5": z_seg_peaks[4],
                "z_seg_peak_6": z_seg_peaks[5],
                "z_seg_peak_7": z_seg_peaks[6],
                "z_seg_peak_8": z_seg_peaks[7],
                "peak_offset_seg_1": peak_offsets_seg[0],
                "peak_offset_seg_2": peak_offsets_seg[1],
                "peak_offset_seg_3": peak_offsets_seg[2],
                "peak_offset_seg_4": peak_offsets_seg[3],
                "peak_offset_seg_5": peak_offsets_seg[4],
                "peak_offset_seg_6": peak_offsets_seg[5],
                "peak_offset_seg_7": peak_offsets_seg[6],
                "peak_offset_seg_8": peak_offsets_seg[7],
            }
            results.append(row_record)

    # Save output CSV: data/d0b/line_scores_v2.csv
    out_csv_path = out_csv_dir / "line_scores_v2.csv"
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv_path, index=False)

    # Print summary table sorted by z_peak descending
    df_sorted = df_out.sort_values(by="z_peak", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 95)
    print("TASK D0b-score2: WHOLE-LINE SCORES WITH PEAK SEARCH (sorted by z_peak descending)")
    print("=" * 95)
    print(f"{'Frame':<15} {'Line':<5} {'Verdict':<7} {'z_peak':>8} {'z_fixed0':>9} {'peak_off':>9} {'z>3 Segs':>9} {'Slope (px/seg)':>15}")
    print("-" * 95)

    for _, r in df_sorted.iterrows():
        sl_str = f"{r['slope_px_per_segment']:+.2f}" if not np.isnan(r['slope_px_per_segment']) else "NaN"
        zf_str = f"{r['z_line_fixed0']:.2f}" if not np.isnan(r['z_line_fixed0']) else "NaN"
        print(
            f"{r['frame']:<15} {r['line_id']:<5} {r['verdict']:<7} {r['z_peak']:>8.2f} "
            f"{zf_str:>9} {r['peak_offset']:>+9d} {r['n_segments_z_gt_3']:>9d} {sl_str:>15}"
        )

    print("=" * 95)

    # Separation metrics on z_peak
    real_z = df_out[df_out["verdict"] == "REAL"]["z_peak"].values
    false_z = df_out[df_out["verdict"] == "FALSE"]["z_peak"].values

    max_z_false = float(np.max(false_z)) if len(false_z) > 0 else float("nan")
    min_z_real = float(np.min(real_z)) if len(real_z) > 0 else float("nan")
    gap = min_z_real - max_z_false

    print(f"\nLargest z_peak among FALSE lines: {max_z_false:.2f}")
    print(f"Smallest z_peak among REAL lines:  {min_z_real:.2f}")
    print(f"Separation gap (min_REAL - max_FALSE): {gap:.2f}")
    print(f"\nSaved CSV: {out_csv_path}")
    print(f"Saved {len(results)} diagnostic figures in {out_img_dir}")


if __name__ == "__main__":
    main()
