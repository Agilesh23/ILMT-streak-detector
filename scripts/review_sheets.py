import sys
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background

# Load labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)

# Frames for PART A
frames_part_a = [
    "20221025-0028.fits",
    "20230519-0009.fits",
    "20230317-0008.fits",
    "20221026-0057.fits",
    "20230602-0022.fits",
    "20231115-0003.fits",
    "20240114-0028.fits",
    "20240405-0004.fits"
]

# Physical streak groups
streak_groups = {
    3673: "g_1",
    16881: "g_2",
    14253: "g_2",
    3618: "g_3",
    1175: "g_4",
    14613: "g_5",
    12522: "g_5",
    17987: "g_5",
    9558: "g_5",
    10716: "g_5"
}

# Suspect FP list
suspect_fps = {
    "20230317-0008.fits": [66, 3767, 7217],
    "20221026-0057.fits": [770, 11455, 18383],
    "20240114-0028.fits": [18665],
    "20231115-0003.fits": [15632]
}

# 8 new long candidates
new_long_candidates = {
    "20221025-0028.fits": [(6527, (1038, 0, 2957, 4096), "duplicate_streak")],
    "20230519-0009.fits": [(5768, (1271, 784, 3443, 3779), "duplicate_streak")],
    "20221026-0057.fits": [
        (1183, (347, 1110, 427, 3328), "horizontal_artifact"),
        (1714, (607, 540, 2938, 599), "bleed_trail")
    ],
    "20230602-0022.fits": [(11716, (2224, 2042, 4032, 4096), "fp_corridor")],
    "20231115-0003.fits": [
        (888, (153, 0, 2329, 2463), "collinear_frag"),
        (12982, (2334, 2473, 3769, 4096), "collinear_frag")
    ],
    "20240114-0028.fits": [(5446, (966, 0, 2690, 4096), "diagonal_streak")]
}

written_files = []

# Cache frame backgrounds
frame_cache = {}
for fname in frames_part_a:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    sub = d - bg
    frame_max = float(np.max(d))
    frame_cache[fname] = {
        "data": d, "bg": bg, "std": std, "sub": sub, "frame_max": frame_max
    }

# =========================================================================
# PART A: OVERVIEW PLOTS
# =========================================================================
downscale = 3  # 4096 // 3 = 1365 px <= 1400 px
for fname in frames_part_a:
    fc = frame_cache[fname]
    sub = fc["sub"]
    d = fc["data"]
    frame_max = fc["frame_max"]
    
    # Block mean downscale
    n_pix = 4096 // downscale
    sub_crop = sub[:n_pix * downscale, :n_pix * downscale]
    sub_down = sub_crop.reshape(n_pix, downscale, n_pix, downscale).mean(axis=(1, 3))
    
    vmin, vmax = np.percentile(sub_down, [5, 99.5])
    
    fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
    ax.imshow(np.clip(sub_down, vmin, vmax), cmap="gray", origin="lower", extent=[0, n_pix, 0, n_pix])
    
    # 1. GREEN = labeled streak rows
    streaks_df = df_all[(df_all["frame_name"] == fname) & (df_all["label"] == "streak")]
    for _, row in streaks_df.iterrows():
        cid = int(row["candidate_id"])
        r0, c0, r1, c1 = row["bbox_r0"] / downscale, row["bbox_c0"] / downscale, row["bbox_r1"] / downscale, row["bbox_c1"] / downscale
        rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="#00FF00", facecolor="none", lw=1.2)
        ax.add_patch(rect)
        grp = streak_groups.get(cid, None)
        lbl_text = f"{cid} {grp}" if grp else f"{cid}"
        ax.text(c0, max(0, r0 - 10), lbl_text, color="#00FF00", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))
        
    # 2. ORANGE = suspect fp rows
    if fname in suspect_fps:
        for fpid in suspect_fps[fname]:
            r_fp = df_all[(df_all["frame_name"] == fname) & (df_all["candidate_id"] == fpid)]
            if len(r_fp) > 0:
                row = r_fp.iloc[0]
                r0, c0, r1, c1 = row["bbox_r0"] / downscale, row["bbox_c0"] / downscale, row["bbox_r1"] / downscale, row["bbox_c1"] / downscale
                rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="#FFA500", facecolor="none", lw=1.2)
                ax.add_patch(rect)
                note_str = str(row["notes"]) if pd.notna(row["notes"]) else "fp"
                ax.text(c0, min(n_pix - 10, r1 + 3), f"{fpid} ({note_str})", color="#FFA500", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))
                
    # 3. CYAN = new long candidates
    if fname in new_long_candidates:
        for cid, (r0_f, c0_f, r1_f, c1_f), note_str in new_long_candidates[fname]:
            r0, c0, r1, c1 = r0_f / downscale, c0_f / downscale, r1_f / downscale, c1_f / downscale
            rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="#00FFFF", facecolor="none", lw=1.2)
            ax.add_patch(rect)
            ax.text(c0 + 5, r0 + 5, f"{cid} ({note_str})", color="#00FFFF", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))
            
    # 4. YELLOW star = centre of each saturated blob (pixels >= 0.9 * max, area > 200 px)
    sat_mask = (d >= 0.9 * frame_max)
    lbl_sat = label(sat_mask)
    sat_props = regionprops(lbl_sat)
    for p in sat_props:
        if p.area > 200:
            cy_d = p.centroid[0] / downscale
            cx_d = p.centroid[1] / downscale
            ax.plot(cx_d, cy_d, marker="*", color="#FFFF00", markersize=12, markeredgecolor="black", markeredgewidth=0.8)
            
    # Custom Legend & Title
    legend_elements = [
        Line2D([0], [0], color="#00FF00", lw=2, label="Green: Labeled Streak (CSV)"),
        Line2D([0], [0], color="#FFA500", lw=2, label="Orange: Suspect FP"),
        Line2D([0], [0], color="#00FFFF", lw=2, label="Cyan: New Long Candidate"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFFF00", markeredgecolor="black", markersize=12, label="Yellow Star: Sat Blob (>200px)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.8)
    
    stem = fname.replace(".fits", "")
    ax.set_title(f"Overview: {fname} | Green=Streak | Orange=Suspect FP | Cyan=New Long | Yellow*=Saturated", fontsize=11, fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    out_path = review_dir / f"overview_{stem}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    written_files.append(out_path)

# =========================================================================
# HELPER: Draw Scale Bar
# =========================================================================
def add_scale_bar(ax, crop_h, crop_w, length_px=50):
    x1 = crop_w - 20
    x0 = x1 - length_px
    y = 20
    ax.plot([x0, x1], [y, y], color="#FFFF00", lw=3.0)
    ax.text((x0 + x1) / 2.0, y + 8, f"{length_px} px", color="#FFFF00", fontsize=9, ha="center", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="black", alpha=0.6, edgecolor="none"))

# =========================================================================
# PART B: ZOOM SHEETS
# =========================================================================

# --- 1. ZOOM A: STREAKS ---
zoom_a_ids = [3673, 16881, 14253, 3618, 1175, 14613, 12522, 17987, 9558, 10716]
fig = plt.figure(figsize=(18, 10), dpi=100)
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.45], hspace=0.28, wspace=0.15)

for idx, cid in enumerate(zoom_a_ids):
    row_idx = idx // 5
    col_idx = idx % 5
    ax = fig.add_subplot(gs[row_idx, col_idx])
    
    r_match = df_all[df_all["candidate_id"] == cid].iloc[0]
    fname = r_match["frame_name"]
    fc = frame_cache[fname]
    sub = fc["sub"]
    
    cy = float(r_match["centroid_row"])
    cx = float(r_match["centroid_col"])
    
    # 300 x 300 crop around centroid
    half = 150
    r0 = max(0, min(4096 - 300, int(round(cy)) - half))
    c0 = max(0, min(4096 - 300, int(round(cx)) - half))
    r1 = r0 + 300
    c1 = c0 + 300
    crop = sub[r0:r1, c0:c1]
    
    vmin, vmax = np.percentile(crop, [5, 99.5])
    ax.imshow(np.clip(crop, vmin, vmax), cmap="gray", origin="lower")
    
    # Draw candidate bbox
    br0, bc0, br1, bc1 = r_match["bbox_r0"], r_match["bbox_c0"], r_match["bbox_r1"], r_match["bbox_c1"]
    rect = patches.Rectangle((bc0 - c0, br0 - r0), bc1 - bc0, br1 - br0, edgecolor="#00FF00", facecolor="none", lw=1.2)
    ax.add_patch(rect)
    
    add_scale_bar(ax, 300, 300, length_px=50)
    
    grp = streak_groups.get(cid, "streak")
    stem = fname.replace(".fits", "")
    ax.set_title(f"{stem} | {cid} | streak | {grp}", fontsize=9, fontweight="bold")
    ax.axis("off")

# Extra panel for 1175: strip along rows 300 to 500 across cols 0 to 4096
ax_strip_a = fig.add_subplot(gs[2, :])
sub_57 = frame_cache["20221026-0057.fits"]["sub"]
strip_a = sub_57[300:500, 0:4096]
vmin_s, vmax_s = np.percentile(strip_a, [5, 99.5])
ax_strip_a.imshow(np.clip(strip_a, vmin_s, vmax_s), cmap="gray", origin="lower", extent=[0, 4096, 300, 500], aspect="auto")

# Mark Giant Saturated Star at row 381, col 574
ax_strip_a.plot(574, 381, marker="*", color="#FFFF00", markersize=14, markeredgecolor="black", markeredgewidth=1.0)
ax_strip_a.text(574, 400, "Giant Star Disc (r381, c574)", color="#FFFF00", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))

# Mark Streak 1175
rect_1175 = patches.Rectangle((3628, 341), 4096 - 3628, 401 - 341, edgecolor="#00FF00", facecolor="none", lw=1.5)
ax_strip_a.add_patch(rect_1175)
ax_strip_a.text(3628, 415, "Streak 1175", color="#00FF00", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))

ax_strip_a.set_title("20221026-0057 | 1175 | streak | Row band 300-500 full context (Giant Star marked)", fontsize=10, fontweight="bold")
ax_strip_a.set_xlabel("Column (px)", fontsize=9)
ax_strip_a.set_ylabel("Row (px)", fontsize=9)

out_zoom_a = review_dir / "zoom_A_streaks.png"
plt.savefig(out_zoom_a, dpi=100)
plt.close()
written_files.append(out_zoom_a)


# --- 2. ZOOM B: SUSPECT FP ---
zoom_b_ids = [66, 3767, 7217, 770, 11455, 18383, 18665, 15632]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), dpi=100)
axes = axes.ravel()

for idx, cid in enumerate(zoom_b_ids):
    ax = axes[idx]
    r_match = df_all[df_all["candidate_id"] == cid].iloc[0]
    fname = r_match["frame_name"]
    fc = frame_cache[fname]
    sub = fc["sub"]
    
    br0, bc0, br1, bc1 = r_match["bbox_r0"], r_match["bbox_c0"], r_match["bbox_r1"], r_match["bbox_c1"]
    
    # 18665 is long (bleed trail) -> use 800 x 800 around middle of bbox
    if cid == 18665:
        crop_sz = 800
        mid_r = (br0 + br1) / 2.0
        mid_c = (bc0 + bc1) / 2.0
        half = 400
        r0 = max(0, min(4096 - crop_sz, int(round(mid_r)) - half))
        c0 = max(0, min(4096 - crop_sz, int(round(mid_c)) - half))
    else:
        crop_sz = 300
        cy = float(r_match["centroid_row"])
        cx = float(r_match["centroid_col"])
        half = 150
        r0 = max(0, min(4096 - crop_sz, int(round(cy)) - half))
        c0 = max(0, min(4096 - crop_sz, int(round(cx)) - half))
        
    r1 = r0 + crop_sz
    c1 = c0 + crop_sz
    crop = sub[r0:r1, c0:c1]
    
    vmin, vmax = np.percentile(crop, [5, 99.5])
    ax.imshow(np.clip(crop, vmin, vmax), cmap="gray", origin="lower")
    
    rect = patches.Rectangle((bc0 - c0, br0 - r0), bc1 - bc0, br1 - br0, edgecolor="#FFA500", facecolor="none", lw=1.2)
    ax.add_patch(rect)
    
    add_scale_bar(ax, crop_sz, crop_sz, length_px=50)
    
    note_str = str(r_match["notes"]) if pd.notna(r_match["notes"]) else "fp"
    stem = fname.replace(".fits", "")
    ax.set_title(f"{stem} | {cid} | fp | {note_str}", fontsize=9, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
out_zoom_b = review_dir / "zoom_B_suspect_fp.png"
plt.savefig(out_zoom_b, dpi=100)
plt.close()
written_files.append(out_zoom_b)


# --- 3. ZOOM C: NEW LONG CANDIDATES ---
zoom_c_items = [
    ("20221025-0028.fits", 6527, (1038, 0, 2957, 4096), "duplicate_streak"),
    ("20230519-0009.fits", 5768, (1271, 784, 3443, 3779), "duplicate_streak"),
    ("20221026-0057.fits", 1183, (347, 1110, 427, 3328), "horizontal_artifact"),
    ("20221026-0057.fits", 1714, (607, 540, 2938, 599), "bleed_trail"),
    ("20230602-0022.fits", 11716, (2224, 2042, 4032, 4096), "fp_corridor"),
    ("20231115-0003.fits", 888, (153, 0, 2329, 2463), "collinear_frag"),
    ("20231115-0003.fits", 12982, (2334, 2473, 3769, 4096), "collinear_frag"),
    ("20240114-0028.fits", 5446, (966, 0, 2690, 4096), "diagonal_streak")
]

fig = plt.figure(figsize=(18, 11), dpi=100)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.45], hspace=0.28, wspace=0.15)

for idx, (fname, cid, (br0, bc0, br1, bc1), note_str) in enumerate(zoom_c_items):
    row_idx = idx // 4
    col_idx = idx % 4
    ax = fig.add_subplot(gs[row_idx, col_idx])
    
    fc = frame_cache[fname]
    sub = fc["sub"]
    
    crop_sz = 800
    mid_r = (br0 + br1) / 2.0
    mid_c = (bc0 + bc1) / 2.0
    half = 400
    r0 = max(0, min(4096 - crop_sz, int(round(mid_r)) - half))
    c0 = max(0, min(4096 - crop_sz, int(round(mid_c)) - half))
    r1 = r0 + crop_sz
    c1 = c0 + crop_sz
    crop = sub[r0:r1, c0:c1]
    
    vmin, vmax = np.percentile(crop, [5, 99.5])
    ax.imshow(np.clip(crop, vmin, vmax), cmap="gray", origin="lower")
    
    # Draw candidate bbox (offset by crop)
    rect = patches.Rectangle((bc0 - c0, br0 - r0), bc1 - bc0, br1 - br0, edgecolor="#00FFFF", facecolor="none", lw=1.2)
    ax.add_patch(rect)
    
    add_scale_bar(ax, crop_sz, crop_sz, length_px=50)
    
    stem = fname.replace(".fits", "")
    ax.set_title(f"{stem} | {cid} | new_long | {note_str}", fontsize=9, fontweight="bold")
    ax.axis("off")

# Extra panel for 1183: strip along rows 300 to 500 across cols 0 to 4096
ax_strip_c = fig.add_subplot(gs[2, :])
sub_57 = frame_cache["20221026-0057.fits"]["sub"]
strip_c = sub_57[300:500, 0:4096]
vmin_s, vmax_s = np.percentile(strip_c, [5, 99.5])
ax_strip_c.imshow(np.clip(strip_c, vmin_s, vmax_s), cmap="gray", origin="lower", extent=[0, 4096, 300, 500], aspect="auto")

# Mark Giant Saturated Star at row 381, col 574
ax_strip_c.plot(574, 381, marker="*", color="#FFFF00", markersize=14, markeredgecolor="black", markeredgewidth=1.0)
ax_strip_c.text(574, 400, "Giant Star Disc (r381, c574)", color="#FFFF00", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))

# Mark Candidate 1183
rect_1183 = patches.Rectangle((1110, 347), 3328 - 1110, 427 - 347, edgecolor="#00FFFF", facecolor="none", lw=1.5)
ax_strip_c.add_patch(rect_1183)
ax_strip_c.text(1110, 440, "Candidate 1183 (horizontal artifact)", color="#00FFFF", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.7, edgecolor="none"))

ax_strip_c.set_title("20221026-0057 | 1183 | new_long | Row band 300-500 full context (Giant Star marked)", fontsize=10, fontweight="bold")
ax_strip_c.set_xlabel("Column (px)", fontsize=9)
ax_strip_c.set_ylabel("Row (px)", fontsize=9)

out_zoom_c = review_dir / "zoom_C_new_long.png"
plt.savefig(out_zoom_c, dpi=100)
plt.close()
written_files.append(out_zoom_c)

# Print written files
for wf in written_files:
    print(wf.as_posix())
