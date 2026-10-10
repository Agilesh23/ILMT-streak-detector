import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg["detection"]

# Labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)

# 8 long blobs + 18665:
blobs_to_investigate = [
    ("20221025-0028.fits", 6527),
    ("20230519-0009.fits", 5768),
    ("20221026-0057.fits", 1183),
    ("20221026-0057.fits", 1714),
    ("20230602-0022.fits", 11716),
    ("20231115-0003.fits", 888),
    ("20231115-0003.fits", 12982),
    ("20240114-0028.fits", 5446),
    ("20240114-0028.fits", 18665)
]

unique_frames = list(dict.fromkeys([b[0] for b in blobs_to_investigate] + list(df_all["frame_name"].unique())))

print("Loading frames and segmenting blobs...", flush=True)
frame_data = {}
for fname in unique_frames:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    sub = d - bg
    thresh_map = 3.0 * std
    binary = sub > thresh_map
    struct = generate_binary_structure(2, 1)
    binary = binary_dilation(binary, structure=struct, iterations=2)
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=sub)
    
    # Store lookup by label
    prop_dict = {p.label: p for p in props}
    frame_max = float(np.max(d))
    frame_data[fname] = {
        "data": d, "bg": bg, "std": std, "sub": sub,
        "props": prop_dict, "frame_max": frame_max
    }

print("\n" + "="*50, flush=True)
print("=== ITEM 1: PCA OF MASK-PIXEL COORDINATES ===", flush=True)
print("="*50, flush=True)

pca_results = {}

for fname, blabel in blobs_to_investigate:
    fd = frame_data[fname]
    p = fd["props"][blabel]
    coords = p.coords # (row, col)
    
    pts_x = coords[:, 1].astype(np.float64) # col
    pts_y = coords[:, 0].astype(np.float64) # row
    cx = float(np.mean(pts_x))
    cy = float(np.mean(pts_y))
    
    dx = pts_x - cx
    dy = pts_y - cy
    cov = np.cov(np.vstack([dx, dy]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    vec = eigvecs[:, 0] # principal direction (vx, vy)
    if vec[0] < 0: vec = -vec
    
    vx, vy = float(vec[0]), float(vec[1])
    
    # Angle from horizontal in [-90, 90]
    ang_h_deg = float(np.degrees(np.arctan2(vy, vx)))
    if ang_h_deg > 90.0: ang_h_deg -= 180.0
    elif ang_h_deg < -90.0: ang_h_deg += 180.0
    
    # Angle from vertical in [0, 90]
    ang_v_deg = float(abs(90.0 - abs(ang_h_deg)))
    
    major = 4.0 * np.sqrt(max(0.0, eigvals[0]))
    minor = 4.0 * np.sqrt(max(0.0, eigvals[1]))
    err_bound_deg = float(np.degrees(np.arctan(minor / major)))
    
    pca_results[(fname, blabel)] = {
        "cx": cx, "cy": cy, "vx": vx, "vy": vy,
        "ang_h": ang_h_deg, "ang_v": ang_v_deg,
        "major": major, "minor": minor,
        "err_bound": err_bound_deg,
        "bbox": p.bbox, "area": p.area,
        "coords": coords
    }
    
    print(f"({fname}, {blabel:<5}) | ang_h: {ang_h_deg:6.2f} deg | ang_v: {ang_v_deg:5.2f} deg | "
          f"minor/major: {minor/major:5.3f} | err_bound: {err_bound_deg:5.2f} deg | bbox: {p.bbox}", flush=True)

# Save outputs/long_pca_check.png for the 8 new long blobs
print("\nSaving outputs/long_pca_check.png...", flush=True)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
downscale = 4

for idx, (fname, blabel) in enumerate(blobs_to_investigate[:8]):
    ax = axes[idx]
    fd = frame_data[fname]
    sub_down = fd["sub"][::downscale, ::downscale]
    vmin, vmax = np.percentile(sub_down, [2, 98])
    ax.imshow(np.clip(sub_down, vmin, vmax), cmap="gray", origin="lower")
    
    res = pca_results[(fname, blabel)]
    cx, cy = res["cx"], res["cy"]
    vx, vy = res["vx"], res["vy"]
    half_L = res["major"] / 2.0
    
    x0 = (cx - half_L * vx) / downscale
    y0 = (cy - half_L * vy) / downscale
    x1 = (cx + half_L * vx) / downscale
    y1 = (cy + half_L * vy) / downscale
    
    ax.plot([x0, x1], [y0, y1], "r-", lw=1.8, label="PCA Axis")
    
    r0, c0, r1, c1 = res["bbox"]
    rect = plt.Rectangle((c0/downscale, r0/downscale), (c1-c0)/downscale, (r1-r0)/downscale,
                         edgecolor="cyan", facecolor="none", lw=1.0, ls="--")
    ax.add_patch(rect)
    
    ax.set_title(f"({fname.replace('.fits','')}, {blabel})\nθ_h={res['ang_h']:.1f}° | θ_v={res['ang_v']:.1f}° | err={res['err_bound']:.1f}°", fontsize=9)
    ax.axis("off")

plt.tight_layout()
out_png = base_dir / "outputs/long_pca_check.png"
plt.savefig(out_png, dpi=150)
plt.close()
print(f"Saved {out_png}", flush=True)

print("\n" + "="*50, flush=True)
print("=== ITEM 2: BLEED VETO, CORRECTED ===", flush=True)
print("="*50, flush=True)

# Bleed veto rule:
# PCA angle within 10 degrees of vertical AND blob contains saturated pixels (>= 0.9 * frame_max)
flagged_labeled = []
for idx, row in df_all.iterrows():
    fname = row["frame_name"]
    cid = row["candidate_id"]
    lbl = row["label"]
    fd = frame_data[fname]
    if cid not in fd["props"]:
        continue
    p = fd["props"][cid]
    coords = p.coords
    
    pts_x = coords[:, 1].astype(np.float64)
    pts_y = coords[:, 0].astype(np.float64)
    cx = np.mean(pts_x)
    cy = np.mean(pts_y)
    dx = pts_x - cx
    dy = pts_y - cy
    cov = np.cov(np.vstack([dx, dy]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigvals)[::-1]
    vec = eigvecs[:, idx_sort[0]]
    if vec[0] < 0: vec = -vec
    
    ang_h = float(np.degrees(np.arctan2(vec[1], vec[0])))
    if ang_h > 90.0: ang_h -= 180.0
    elif ang_h < -90.0: ang_h += 180.0
    ang_v = float(abs(90.0 - abs(ang_h)))
    
    # Check saturated pixels in blob
    blob_pixels = fd["data"][coords[:, 0], coords[:, 1]]
    max_blob_val = float(np.max(blob_pixels))
    sat_thresh = 0.9 * fd["frame_max"]
    has_sat = max_blob_val >= sat_thresh
    
    if ang_v <= 10.0 and has_sat:
        flagged_labeled.append({
            "frame": fname, "id": cid, "label": lbl,
            "ang_v": ang_v, "max_val": max_blob_val, "sat_thresh": sat_thresh,
            "frame_max": fd["frame_max"]
        })

print(f"Total labeled rows flagged: {len(flagged_labeled)}")
for f in flagged_labeled:
    print(f"  Flagged Labeled: {f}")

# Check 8 new blobs:
flagged_new = []
for fname, blabel in blobs_to_investigate[:8]:
    fd = frame_data[fname]
    res = pca_results[(fname, blabel)]
    coords = res["coords"]
    blob_pixels = fd["data"][coords[:, 0], coords[:, 1]]
    max_blob_val = float(np.max(blob_pixels))
    sat_thresh = 0.9 * fd["frame_max"]
    has_sat = max_blob_val >= sat_thresh
    ang_v = res["ang_v"]
    
    if ang_v <= 10.0 and has_sat:
        flagged_new.append({
            "frame": fname, "id": blabel,
            "ang_v": ang_v, "max_val": max_blob_val, "sat_thresh": sat_thresh,
            "frame_max": fd["frame_max"]
        })

print(f"Total new blobs flagged: {len(flagged_new)}")
for f in flagged_new:
    print(f"  Flagged New: {f}")

# Check 18665 specifically:
res_18665 = pca_results[("20240114-0028.fits", 18665)]
fd_18665 = frame_data["20240114-0028.fits"]
max_val_18665 = float(np.max(fd_18665["data"][res_18665["coords"][:, 0], res_18665["coords"][:, 1]]))
thresh_18665 = 0.9 * fd_18665["frame_max"]
flagged_18665 = (res_18665["ang_v"] <= 10.0) and (max_val_18665 >= thresh_18665)
print(f"Candidate 18665: ang_v = {res_18665['ang_v']:.2f} deg, max_val = {max_val_18665:.1f}, thresh = {thresh_18665:.1f}, flagged = {flagged_18665}")

print("\n" + "="*50, flush=True)
print("=== ITEM 3: 20221026-0057 GIANT STAR DISC & HORIZONTAL LINE ===", flush=True)
print("="*50, flush=True)

# Giant star DISC in 20221026-0057
# Saturated halo/disc around rows 200-550, cols 400-750
fd_57 = frame_data["20221026-0057.fits"]
d_57 = fd_57["data"]
sub_57 = fd_57["sub"]
std_57 = fd_57["std"]
global_std_57 = float(np.median(std_57))

# Saturated star disc:
# The star's disc has values > 5000 ADU and area > 500 px
# Combine the two halves across the bleed stripe or take the contiguous saturated core
lbl_disc = label(d_57 > 5000)
props_disc = regionprops(lbl_disc, intensity_image=d_57)
star_blobs = [p for p in props_disc if p.area > 500 and p.centroid[0] < 800 and p.centroid[1] > 300 and p.centroid[1] < 800]
print(f"Found {len(star_blobs)} star disc blobs > 500 px:")
for sb in star_blobs:
    print(f"   bbox = {sb.bbox}, centroid = ({sb.centroid[0]:.1f}, {sb.centroid[1]:.1f}), area = {sb.area}, max = {sb.intensity_max:.1f}")

# Joint disc combining these components:
all_disc_coords = np.vstack([sb.coords for sb in star_blobs])
disc_r0, disc_c0 = np.min(all_disc_coords, axis=0)
disc_r1, disc_c1 = np.max(all_disc_coords, axis=0) + 1
disc_centroid_row = float(np.mean(all_disc_coords[:, 0]))
disc_centroid_col = float(np.mean(all_disc_coords[:, 1]))
disc_bbox = (int(disc_r0), int(disc_c0), int(disc_r1), int(disc_c1))
print(f"Combined Giant Star DISC: centroid = (row {disc_centroid_row:.1f}, col {disc_centroid_col:.1f}), bbox = {disc_bbox}, total_area = {len(all_disc_coords)}")

# Peak row of the horizontal line (artifact / blob 1183)
# Rows 250-550 median profile over cols 100-4000 (skip stars)
from scipy.ndimage import maximum_filter
star_mask = (d_57 >= 25000) | (sub_57 > 5.0 * std_57)
star_mask_dil = maximum_filter(star_mask, size=121)
row_indices = np.arange(250, 551)
row_meds = []
for r in row_indices:
    valid_cols = np.where(~star_mask_dil[r, 100:4001])[0] + 100
    row_meds.append(float(np.median(sub_57[r, valid_cols])) if len(valid_cols) > 0 else 0.0)

row_meds = np.array(row_meds)
peak_idx = int(np.argmax(row_meds))
horiz_peak_row = int(row_indices[peak_idx])
diff_row = abs(horiz_peak_row - disc_centroid_row)
within_30 = diff_row <= 30.0

print(f"Horizontal Line Peak Row: {horiz_peak_row} (signal = {row_meds[peak_idx]:.2f} ADU, {row_meds[peak_idx]/global_std_57:.2f} sigma)")
print(f"Difference |peak_row - disc_centroid_row| = {diff_row:.1f} px")
print(f"Is peak row within 30 px of disc centroid row? {within_30}")

print("\n" + "="*50, flush=True)
print("=== ITEM 4: SEED 15632 (20231115-0003) & STITCHED BLOB 888 ===", flush=True)
print("="*50, flush=True)

fd_15 = frame_data["20231115-0003.fits"]
d_15 = fd_15["data"]
sub_15 = fd_15["sub"]
props_15 = fd_15["props"]

# Seed 15632
p_15632 = props_15[15632]
seed_cy, seed_cx = p_15632.centroid
print(f"Seed 15632: centroid = (row {seed_cy:.2f}, col {seed_cx:.2f}), bbox = {p_15632.bbox}, area = {p_15632.area}")

# Stitched blob 888 (mask pixels of 888 + 12982)
coords_888 = props_15[888].coords
coords_12982 = props_15[12982].coords
coords_stitched = np.vstack([coords_888, coords_12982])

pts_x_st = coords_stitched[:, 1].astype(np.float64)
pts_y_st = coords_stitched[:, 0].astype(np.float64)
st_cx = float(np.mean(pts_x_st))
st_cy = float(np.mean(pts_y_st))

cov_st = np.cov(np.vstack([pts_x_st - st_cx, pts_y_st - st_cy]))
eigvals_st, eigvecs_st = np.linalg.eigh(cov_st)
idx_st = np.argsort(eigvals_st)[::-1]
vec_st = eigvecs_st[:, idx_st[0]]
if vec_st[0] < 0: vec_st = -vec_st

st_vx, st_vy = float(vec_st[0]), float(vec_st[1])
st_ang_h = float(np.degrees(np.arctan2(st_vy, st_vx)))
if st_ang_h > 90: st_ang_h -= 180
elif st_ang_h < -90: st_ang_h += 180

# Perpendicular distance from seed 15632 centroid to PCA line of stitched 888
# normal vector (-st_vy, st_vx)
perp_dist_seed = abs((seed_cx - st_cx) * (-st_vy) + (seed_cy - st_cy) * st_vx)
print(f"Stitched blob 888 PCA line: center = ({st_cx:.1f}, {st_cy:.1f}), angle_h = {st_ang_h:.2f} deg")
print(f"Perpendicular distance from Seed 15632 to PCA line of stitched 888 = {perp_dist_seed:.3f} px")

# Save outputs/seed_15632.png: 400 x 400 zoom plus the line
zoom_half = 200
r_min = max(0, int(round(seed_cy)) - zoom_half)
r_max = min(d_15.shape[0], int(round(seed_cy)) + zoom_half)
c_min = max(0, int(round(seed_cx)) - zoom_half)
c_max = min(d_15.shape[1], int(round(seed_cx)) + zoom_half)

patch = sub_15[r_min:r_max, c_min:c_max]
vmin, vmax = np.percentile(patch, [2, 98])

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(np.clip(patch, vmin, vmax), cmap="gray", origin="lower", extent=[c_min, c_max, r_min, r_max])

# Plot PCA line across zoom
t_vals = np.linspace(-3000, 3000, 500)
line_x = st_cx + t_vals * st_vx
line_y = st_cy + t_vals * st_vy
mask_in = (line_x >= c_min) & (line_x <= c_max) & (line_y >= r_min) & (line_y <= r_max)
ax.plot(line_x[mask_in], line_y[mask_in], "r-", lw=1.5, label="Stitched 888 PCA line")

# Mark seed 15632
sb0, sb1, sb2, sb3 = p_15632.bbox
seed_rect = plt.Rectangle((sb1, sb0), sb3 - sb1, sb2 - sb0, edgecolor="cyan", facecolor="none", lw=1.5, label="Seed 15632 bbox")
ax.add_patch(seed_rect)
ax.scatter([seed_cx], [seed_cy], color="yellow", s=30, marker="+", label=f"Centroid ({seed_cy:.1f}, {seed_cx:.1f})")

ax.set_title(f"Seed 15632 Zoom (20231115-0003)\nPerp Dist to Stitched 888 PCA Line: {perp_dist_seed:.2f} px", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
out_seed_png = base_dir / "outputs/seed_15632.png"
plt.savefig(out_seed_png, dpi=150)
plt.close()
print(f"Saved {out_seed_png}", flush=True)

print("\nALL ITEMS COMPLETED SUCCESSFULLY.", flush=True)
