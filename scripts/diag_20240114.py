import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240114-0028.fits")

with open(base_dir / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg.get("detection", {})
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)

data, header = load_fits(frame_path)
bg, std = estimate_background(data, method="background2d")
sub = data - bg

# Fresh pipeline run
cands_raw, count_before, count_shape = find_blob_candidates(
    data, bg, std,
    n_sigma=3.0,
    min_aspect_ratio=3.0,
    min_eccentricity=0.9,
    max_area=15000.0,
    edge_mask_max_x=edge_mask_max_x,
    border_margin_px=border_margin_px,
    border_max_length_px=border_max_length_px,
    left_edge_max_col=left_edge_max_col,
    left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg
)

stitched = stitch_collinear_fragments(cands_raw, data, bg, std)
selected, ranked = rank_and_select_candidates(stitched, min_abs_score=min_abs_score)

# Line geometry: (col 0, row 1000) to (col 4096, row 2650)
p0 = np.array([0.0, 1000.0]) # (col, row)
p1 = np.array([4096.0, 2650.0])
line_vec = p1 - p0
line_len = np.hypot(line_vec[0], line_vec[1])
line_u = line_vec / line_len
line_perp = np.array([-line_u[1], line_u[0]]) # (-dy, dx)

def dist_to_segment(cx, cy):
    # point (cx, cy)
    pt = np.array([cx, cy])
    proj = np.dot(pt - p0, line_u)
    proj_clamped = np.clip(proj, 0.0, line_len)
    closest = p0 + proj_clamped * line_u
    return np.hypot(pt[0] - closest[0], pt[1] - closest[1])

df_labels = pd.concat([
    pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv"),
    pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
]).reset_index(drop=True)

df_this = df_labels[df_labels["frame_name"] == "20240114-0028.fits"]

print("=== CANDIDATES WITHIN 30 PX OF LINE ===")
# Check in cands_raw (after masks/shape), stitched, and selected
for stage_name, c_list in [("After masks & shape filter", cands_raw), ("After stitching", stitched), ("Final selected", selected)]:
    print(f"\n--- Stage: {stage_name} (Total cands: {len(c_list)}) ---")
    near_line = []
    for c in c_list:
        cy, cx = c["centroid"]
        d = dist_to_segment(cx, cy)
        if d <= 30.0:
            cid = c.get("label", c.get("candidate_id", None))
            bbox = c.get("bbox_full", c.get("bbox_local"))
            score = c.get("score", None)
            
            # Check CSV label
            csv_match = df_this[df_this["candidate_id"] == cid]
            lbl = csv_match["label"].values[0] if len(csv_match) > 0 else "unlabeled"
            near_line.append((cid, bbox, score, lbl, d, cy, cx))
            
    if not near_line:
        print("None found within 30 px.")
    else:
        for item in near_line:
            sc_str = f"{item[2]:.2f}" if item[2] is not None else "N/A"
            print(f"  ID {item[0]}: bbox={item[1]}, score={sc_str}, label={item[3]}, dist={item[4]:.1f}px, centroid=({item[5]:.1f}, {item[6]:.1f})")

# 2. Candidate 18665 audit
print("\n=== CANDIDATE 18665 AUDIT ===")
row_18665 = df_this[df_this["candidate_id"] == 18665].iloc[0]
print(f"Stored Centroid (row, col): ({row_18665['centroid_row']}, {row_18665['centroid_col']})")
print(f"Stored Bbox (r0, c0, r1, c1): ({row_18665['bbox_r0']}, {row_18665['bbox_c0']}, {row_18665['bbox_r1']}, {row_18665['bbox_c1']})")
print(f"Stored Score: {row_18665['score']}, Length: {row_18665['length_px']}, Angle: {row_18665['angle_deg']}")

# Fresh crop of exactly that bbox
r0, c0, r1, c1 = int(row_18665['bbox_r0']), int(row_18665['bbox_c0']), int(row_18665['bbox_r1']), int(row_18665['bbox_c1'])
crop = sub[r0:r1, c0:c1]
# Percentile stretch
vmin, vmax = np.percentile(crop, [2, 98])
crop_norm = np.clip((crop - vmin) / max(vmax - vmin, 1e-4), 0, 1)

out_crop_path = base_dir / "outputs/c18665_bbox.png"
plt.figure(figsize=(4, 6))
plt.imshow(crop_norm, cmap='gray', origin='lower')
plt.title(f"18665 Bbox Crop\n[{r0}:{r1}, {c0}:{c1}]")
plt.axis('off')
plt.tight_layout()
plt.savefig(out_crop_path, dpi=150)
plt.close()
print(f"Saved fresh crop to {out_crop_path}")

# Check saved cutout vs context
saved_cutout_path = base_dir / row_18665["cutout_path"]
saved_ctx_path = base_dir / row_18665["context_path"]
print(f"Saved cutout path: {saved_cutout_path} (exists: {saved_cutout_path.exists()})")
print(f"Saved context path: {saved_ctx_path} (exists: {saved_ctx_path.exists()})")

# Check what the saved cutout dimensions/content are
img_cut = Image.open(saved_cutout_path)
print(f"Saved cutout size (W, H): {img_cut.size}")

# 3. Why was the diagonal line not selected?
print("\n=== DIAGONAL LINE STAGE ANALYSIS ===")
# Sample along the diagonal line in data and sub
s_samples = np.arange(0, line_len, 5.0)
xs_line = p0[0] + s_samples * line_u[0]
ys_line = p0[1] + s_samples * line_u[1]
from scipy.ndimage import map_coordinates
line_vals = map_coordinates(sub, [ys_line, xs_line], order=1)
thresh_vals = map_coordinates(std * 3.0, [ys_line, xs_line], order=1)
print(f"Diagonal line mean intensity: {np.mean(line_vals):.2f}, mean std*3: {np.mean(thresh_vals):.2f}")
print(f"Fraction of line pixels > 3 sigma: {np.mean(line_vals > thresh_vals):.3f}")

# Check connected components in raw threshold mask along that line
from skimage.measure import label, regionprops
raw_mask = sub > (3.0 * std)
# Edge mask column clipping
raw_mask[:, int(edge_mask_max_x):] = False
raw_lbl = label(raw_mask)
raw_props = regionprops(raw_lbl)
print(f"Total raw blobs > 3 sigma (before shape filter): {len(raw_props)}")

blobs_near_line = []
for p in raw_props:
    cy, cx = p.centroid
    if dist_to_segment(cx, cy) <= 30.0:
        blobs_near_line.append(p)
print(f"Blobs > 3 sigma with centroid within 30px of line: {len(blobs_near_line)}")
for p in blobs_near_line[:10]:
    major = p.axis_major_length
    minor = max(p.axis_minor_length, 0.1)
    aspect = major / minor
    ecc = p.eccentricity
    print(f"  Blob label {p.label}: area={p.area}, major={major:.1f}, aspect={aspect:.2f}, ecc={ecc:.3f}, centroid=({cy:.1f}, {cx:.1f})")

# 4. Save overview png
print("\n=== SAVING OVERVIEW PNG ===")
out_overview = base_dir / "outputs/frame_20240114-0028_overview.png"
# Downscaled 4096 full frame
downscale = 4
sub_down = sub[::downscale, ::downscale]
vmin, vmax = np.percentile(sub_down, [1, 99])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.clip(sub_down, vmin, vmax), cmap='gray', origin='lower')

# Draw diagonal line
ax.plot([p0[0] / downscale, p1[0] / downscale], [p0[1] / downscale, p1[1] / downscale], 'r--', lw=1.5, label='Diagonal (0,1000)-(4096,2650)')

# Draw boxes for all selected candidates
for c in selected:
    cid = c.get("label", c.get("candidate_id"))
    r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
    w_box = (c1 - c0) / downscale
    h_box = (r1 - r0) / downscale
    rect = plt.Rectangle((c0 / downscale, r0 / downscale), w_box, h_box, edgecolor='cyan', facecolor='none', lw=1.2)
    ax.add_patch(rect)
    ax.text(c0 / downscale, (r1 / downscale) + 3, str(cid), color='yellow', fontsize=8, fontweight='bold')

ax.set_title("20240114-0028 Overview: Selected Candidates (cyan) & Diagonal (red)")
ax.axis('off')
plt.tight_layout()
plt.savefig(out_overview, dpi=150)
plt.close()
print(f"Saved {out_overview}")
