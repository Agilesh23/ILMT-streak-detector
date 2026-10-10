import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from skimage.transform import radon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)

written_files = []

# =========================================================================
# PART A & B: Frame 20221101-0030.fits
# =========================================================================
fname_main = "20221101-0030.fits"
d_main, _ = load_fits(frame_dir / fname_main)
bg_main, std_main = estimate_background(d_main, method="background2d")
sub_main = d_main - bg_main
frame_max_main = float(np.max(d_main))

# PART A: Overview
# Fresh run detection
cands_main, _, _ = find_blob_candidates(d_main, bg_main, std_main, long_thin_bypass=cfg["detection"]["long_thin_bypass"])
stitched_main = stitch_collinear_fragments(cands_main, d_main, bg_main, std_main)
selected_main, _ = rank_and_select_candidates(stitched_main, min_abs_score=cfg["ranking"]["min_abs_score"])

csv_main = df_all[df_all["frame_name"] == fname_main]

# 3x3 block-mean downscale (4096 -> 1365 <= 1400)
down_a = 3
n_pix_a = 4096 // down_a
sub_down_a = sub_main[:n_pix_a * down_a, :n_pix_a * down_a].reshape(n_pix_a, down_a, n_pix_a, down_a).mean(axis=(1, 3))

vmin_a, vmax_a = np.percentile(sub_down_a, [5, 99.5])

fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
ax.imshow(np.clip(sub_down_a, vmin_a, vmax_a), cmap="gray", origin="lower", extent=[0, 4096, 0, 4096])

for cand in selected_main:
    cid = cand["label"]
    r0, c0, r1, c1 = cand.get("bbox_full", cand.get("bbox_local"))
    
    match_csv = csv_main[csv_main["candidate_id"] == cid]
    if len(match_csv) > 0 and match_csv.iloc[0]["label"] == "fp":
        edge_col = "#FFA500" # Orange
    else:
        edge_col = "#00FFFF" # Cyan
        
    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor=edge_col, facecolor="none", lw=1.2)
    ax.add_patch(rect)
    ax.text(c0, max(0, r0 - 30), f"{cid}", color=edge_col, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))

ax.set_title("20221101-0030 Overview | Orange: FP in CSV | Cyan: Not in CSV (Axes in Full-Res Pixels)", fontsize=11, fontweight="bold")
ax.set_xlabel("X (full-res px)", fontsize=10)
ax.set_ylabel("Y (full-res px)", fontsize=10)

out_overview = review_dir / "frame_20221101-0030_overview.png"
plt.savefig(out_overview, dpi=100)
plt.close()
written_files.append(out_overview)

# PART B: Faint view (4x4 block-mean, gaussian sigma 1.5, stretch 20 to 99.8)
down_b = 4
n_pix_b = 4096 // down_b # 1024
sub_down_b = sub_main[:n_pix_b * down_b, :n_pix_b * down_b].reshape(n_pix_b, down_b, n_pix_b, down_b).mean(axis=(1, 3))
sub_smooth_b = gaussian_filter(sub_down_b, sigma=1.5)

vmin_b, vmax_b = np.percentile(sub_smooth_b, [20, 99.8])

# Full faint
fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
ax.imshow(np.clip(sub_smooth_b, vmin_b, vmax_b), cmap="gray", origin="lower", extent=[0, 4096, 0, 4096])
ax.set_title("20221101-0030 Faint Long Line View (Smoothed 4x4 Downscale, 20-99.8% Stretch)", fontsize=11, fontweight="bold")
ax.set_xlabel("X (full-res px)", fontsize=10)
ax.set_ylabel("Y (full-res px)", fontsize=10)

out_faint = review_dir / "frame_20221101-0030_faint.png"
plt.savefig(out_faint, dpi=100)
plt.close()
written_files.append(out_faint)

# Four quadrants (upscaled to 1200 px wide)
quadrants = [
    ("q1", sub_smooth_b[512:1024, 512:1024], [2048, 4096, 2048, 4096], "Q1 (Top-Right: r2048-4096, c2048-4096)"),
    ("q2", sub_smooth_b[512:1024, 0:512], [0, 2048, 2048, 4096], "Q2 (Top-Left: r2048-4096, c0-2048)"),
    ("q3", sub_smooth_b[0:512, 0:512], [0, 2048, 0, 2048], "Q3 (Bottom-Left: r0-2048, c0-2048)"),
    ("q4", sub_smooth_b[0:512, 512:1024], [2048, 4096, 0, 2048], "Q4 (Bottom-Right: r0-2048, c2048-4096)")
]

for q_name, q_data, q_extent, q_title in quadrants:
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.imshow(np.clip(q_data, vmin_b, vmax_b), cmap="gray", origin="lower", extent=q_extent)
    ax.set_title(f"20221101-0030 Faint {q_title}", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (full-res px)", fontsize=10)
    ax.set_ylabel("Y (full-res px)", fontsize=10)
    out_q = review_dir / f"frame_20221101-0030_faint_{q_name}.png"
    plt.savefig(out_q, dpi=100)
    plt.close()
    written_files.append(out_q)

# =========================================================================
# PART C: LINE SEARCH VIA RADON TRANSFORM
# =========================================================================
def radon_line_search(fname, d, bg, std, known_ang=None):
    sub = d - bg
    n_pix = 1024
    sub_down = sub[:n_pix*4, :n_pix*4].reshape(n_pix, 4, n_pix, 4).mean(axis=(1, 3))
    std_down = std[:n_pix*4, :n_pix*4].reshape(n_pix, 4, n_pix, 4).mean(axis=(1, 3)) / 2.0
    sub_smooth = gaussian_filter(sub_down, sigma=1.5)

    # Mask pixels above 5 sigma and 40 px around saturated cores
    sat_mask = (d >= 0.9 * np.max(d))
    sat_down = sat_mask[:n_pix*4, :n_pix*4].reshape(n_pix, 4, n_pix, 4).any(axis=(1, 3))
    sat_dil = maximum_filter(sat_down, size=21) # ~40 px in full res

    bright_mask = (sub_down > 5.0 * np.median(std_down)) | sat_dil
    img_for_radon = np.where(bright_mask, 0.0, sub_smooth)

    thetas = np.arange(0.0, 180.0, 0.5)
    sinogram = radon(img_for_radon, theta=thetas, circle=False)
    n_rho = sinogram.shape[0]
    rho_center = (n_rho - 1) / 2.0

    flat_idx = np.argsort(sinogram.ravel())[::-1]
    candidates = []
    known_rank = None

    for idx in flat_idx:
        r, a = np.unravel_index(idx, sinogram.shape)
        theta_deg = thetas[a]
        theta_rad = np.radians(theta_deg)
        alpha_h = 90.0 - theta_deg
        if alpha_h > 90.0: alpha_h -= 180.0
        elif alpha_h < -90.0: alpha_h += 180.0

        offset_full = float((r - rho_center) * 4.0)

        # Suppress nearby duplicates
        if any(abs(alpha_h - c["ang_h"]) < 4.0 and abs(offset_full - c["offset"]) < 100.0 for c in candidates):
            continue

        # Intersection with [0, 4096] x [0, 4096]
        # Line: (x - 2048)*cos(theta) + (y - 2048)*sin(theta) = offset
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        pts = []
        if abs(sin_t) > 1e-5:
            # x = 0
            y = 2048.0 + (offset_full + 2048.0 * cos_t) / sin_t
            if 0.0 <= y <= 4096.0: pts.append((0.0, y))
            # x = 4096
            y = 2048.0 + (offset_full - 2048.0 * cos_t) / sin_t
            if 0.0 <= y <= 4096.0: pts.append((4096.0, y))
        if abs(cos_t) > 1e-5:
            # y = 0
            x = 2048.0 + (offset_full + 2048.0 * sin_t) / cos_t
            if 0.0 <= x <= 4096.0: pts.append((x, 0.0))
            # y = 4096
            x = 2048.0 + (offset_full - 2048.0 * sin_t) / cos_t
            if 0.0 <= x <= 4096.0: pts.append((x, 4096.0))

        pts = list(dict.fromkeys([(round(p[0], 2), round(p[1], 2)) for p in pts]))
        if len(pts) < 2:
            continue
        p0, p1 = pts[0], pts[1]
        line_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        # Mean SNR along line
        n_samples = max(10, int(line_len / 4.0))
        ts = np.linspace(0.0, 1.0, n_samples)
        xs_d = (p0[0] + ts * (p1[0] - p0[0])) / 4.0
        ys_d = (p0[1] + ts * (p1[1] - p0[1])) / 4.0

        snr_vals = map_coordinates(sub_down / np.median(std_down), [ys_d, xs_d], order=1)
        mask_vals = map_coordinates(bright_mask.astype(float), [ys_d, xs_d], order=0) > 0.5
        unmasked = snr_vals[~mask_vals]
        mean_snr = float(np.mean(unmasked)) if len(unmasked) > 0 else 0.0

        cand_dict = {
            "ang_h": alpha_h,
            "offset": offset_full,
            "mean_snr": mean_snr,
            "length": line_len,
            "sin_val": float(sinogram[r, a])
        }
        candidates.append(cand_dict)

        if known_ang is not None and known_rank is None:
            if abs(alpha_h - known_ang) <= 2.5:
                known_rank = len(candidates)

        if len(candidates) == 12:
            break

    return candidates, known_rank

# Run controls
frames_part_c = [
    ("20230317-0008.fits", -24.8),
    ("20240405-0004.fits", 22.6),
    ("20221101-0030.fits", None)
]

c_results = {}
for fn, ka in frames_part_c:
    if fn == fname_main:
        d, bg, std = d_main, bg_main, std_main
    else:
        d, _ = load_fits(frame_dir / fn)
        bg, std = estimate_background(d, method="background2d")
    cands, rank = radon_line_search(fn, d, bg, std, ka)
    c_results[fn] = (cands, rank)

# Print written files
print("=== FILES WRITTEN ===")
for wf in written_files:
    print(wf.as_posix())

# Print numbers for Part C
print("\n=== PART C: RADON LINE SEARCH ===")
for fn, ka in frames_part_c:
    cands, rank = c_results[fn]
    print(f"\n[{fn}]" + (f" (Known streak {ka}° rank: #{rank})" if rank else " (No known streak)"))
    for i, c in enumerate(cands[:5]):
        print(f"  Rank {i+1}: angle={c['ang_h']:6.1f}° | offset={c['offset']:7.1f} px | mean_snr={c['mean_snr']:4.2f} | length={c['length']:6.1f} px")
