import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import median_filter, binary_dilation
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
from characterize import fit_weighted_tls_line

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg["detection"]

# Labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)

# 8 new long blobs:
blobs_to_investigate = [
    ("20221025-0028.fits", 6527),
    ("20230519-0009.fits", 5768),
    ("20221026-0057.fits", 1183),
    ("20221026-0057.fits", 1714),
    ("20230602-0022.fits", 11716),
    ("20231115-0003.fits", 888),
    ("20231115-0003.fits", 12982),
    ("20240114-0028.fits", 5446)
]

unique_frames = list(dict.fromkeys([b[0] for b in blobs_to_investigate]))

print("Loading frames and finding candidates...", flush=True)
frame_cache = {}
for fname in unique_frames:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    cands, _, _ = find_blob_candidates(d, bg, std, long_thin_bypass=det_cfg["long_thin_bypass"])
    stitched = stitch_collinear_fragments(cands, d, bg, std)
    frame_cache[fname] = {
        "data": d, "bg": bg, "std": std, "cands": cands, "stitched": stitched
    }

print("\n" + "="*50, flush=True)
print("=== STEP 1: REFIT AXIS OF 8 LONG CANDIDATES ===", flush=True)
print("="*50, flush=True)

refit_results = {}

for fname, blabel in blobs_to_investigate:
    fc = frame_cache[fname]
    d = fc["data"]
    bg = fc["bg"]
    cands = fc["cands"]
    cand = [c for c in cands if c["label"] == blabel][0]
    r0, c0, r1, c1 = cand["bbox_full"]
    cy, cx = cand["centroid"]
    dy = r1 - r0
    dx = c1 - c0
    
    sub = d - bg
    
    # Extract visible ridge points per 50 px block within a corridor along candidate
    ridge_pts = [] # list of (col, row)
    
    if dx >= dy:
        # Near-horizontal or diagonal: slope m
        # In case the candidate spans across bbox corners:
        # Check if slope is positive or negative:
        # Test candidate orientation using covariance of pixels > 2 sigma in bbox
        sub_c = sub[r0:r1, c0:c1]
        std_c = fc["std"][r0:r1, c0:c1]
        y_idx, x_idx = np.where(sub_c > 2.0 * std_c)
        if len(y_idx) > 10:
            cov = np.cov(np.vstack([x_idx, y_idx]))
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, np.argmax(eigvals)]
            m = v[1] / v[0] if abs(v[0]) > 1e-4 else 0.0
        else:
            m = (r1 - r0) / (c1 - c0)
            
        for cb in range(c0, c1, 50):
            ce = min(cb + 50, c1)
            cmid = (cb + ce) / 2.0
            y_approx = cy + m * (cmid - cx)
            y_lo = max(0, int(y_approx - 35))
            y_hi = min(d.shape[0], int(y_approx + 35))
            if y_hi <= y_lo:
                continue
            block = sub[y_lo:y_hi, cb:ce]
            prof = np.median(block, axis=1)
            prof_filt = median_filter(prof, size=5)
            r_peak = y_lo + np.argmax(prof_filt)
            ridge_pts.append((cmid, float(r_peak)))
    else:
        # Near-vertical (e.g. 1714): step in 50 px row blocks
        m_inv = (c1 - c0) / (r1 - r0)
        for rb in range(r0, r1, 50):
            re = min(rb + 50, r1)
            rmid = (rb + re) / 2.0
            x_approx = cx + m_inv * (rmid - cy)
            x_lo = max(0, int(x_approx - 35))
            x_hi = min(d.shape[1], int(x_approx + 35))
            if x_hi <= x_lo:
                continue
            block = sub[rb:re, x_lo:x_hi]
            prof = np.median(block, axis=0)
            prof_filt = median_filter(prof, size=5)
            c_peak = x_lo + np.argmax(prof_filt)
            ridge_pts.append((float(c_peak), rmid))
            
    ridge_pts = np.array(ridge_pts)
    
    center, direction, _ = fit_weighted_tls_line(ridge_pts, np.ones(len(ridge_pts)))
    cx_ref, cy_ref = center
    vx, vy = direction
    
    # Calculate angle from HORIZONTAL:
    ang_h_deg = np.degrees(np.arctan2(vy, vx))
    if ang_h_deg > 90:
        ang_h_deg -= 180
    elif ang_h_deg < -90:
        ang_h_deg += 180
        
    # Project ridge points onto line to get endpoints & length
    t = (ridge_pts[:, 0] - cx_ref) * vx + (ridge_pts[:, 1] - cy_ref) * vy
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    length_px = t_max - t_min
    
    start_pt = (cx_ref + t_min * vx, cy_ref + t_min * vy) # (col, row)
    end_pt = (cx_ref + t_max * vx, cy_ref + t_max * vy)     # (col, row)
    
    # Perpendicular distance from refit line to visible ridge points
    perp_dists = np.abs((ridge_pts[:, 0] - cx_ref) * (-vy) + (ridge_pts[:, 1] - cy_ref) * vx)
    med_perp_dist = float(np.median(perp_dists))
    max_perp_dist = float(np.max(perp_dists))
    
    passes = med_perp_dist < 5.0
    
    refit_results[(fname, blabel)] = {
        "center": center, "direction": (vx, vy),
        "angle_h_deg": ang_h_deg,
        "start_pt": start_pt, "end_pt": end_pt,
        "length_px": length_px,
        "med_perp_dist": med_perp_dist,
        "passes": passes,
        "ridge_pts": ridge_pts,
        "bbox": (r0, c0, r1, c1)
    }
    
    status_str = "PASS" if passes else "FAIL"
    print(f"({fname}, {blabel:<5}) | angle_h: {ang_h_deg:6.2f} deg | length: {length_px:6.1f} px | "
          f"start(c,r): ({start_pt[0]:6.1f}, {start_pt[1]:6.1f}) | end(c,r): ({end_pt[0]:6.1f}, {end_pt[1]:6.1f}) | "
          f"med_dist: {med_perp_dist:4.2f} px [{status_str}]", flush=True)

# Stitched 888+12982 in 20231115-0003
r_pts_888 = refit_results[("20231115-0003.fits", 888)]["ridge_pts"]
r_pts_12982 = refit_results[("20231115-0003.fits", 12982)]["ridge_pts"]
r_pts_stitched = np.vstack([r_pts_888, r_pts_12982])
c_st, dir_st, _ = fit_weighted_tls_line(r_pts_stitched, np.ones(len(r_pts_stitched)))
ang_st = np.degrees(np.arctan2(dir_st[1], dir_st[0]))
if ang_st > 90: ang_st -= 180
elif ang_st < -90: ang_st += 180
t_st = (r_pts_stitched[:, 0] - c_st[0]) * dir_st[0] + (r_pts_stitched[:, 1] - c_st[1]) * dir_st[1]
refit_results[("20231115-0003.fits", "stitched_888_12982")] = {
    "center": c_st, "direction": dir_st, "angle_h_deg": ang_st,
    "start_pt": (c_st[0] + np.min(t_st)*dir_st[0], c_st[1] + np.min(t_st)*dir_st[1]),
    "end_pt": (c_st[0] + np.max(t_st)*dir_st[0], c_st[1] + np.max(t_st)*dir_st[1]),
    "length_px": float(np.max(t_st) - np.min(t_st))
}
print(f"(20231115-0003.fits, stitched_888_12982) | angle_h: {ang_st:6.2f} deg | length: {np.max(t_st)-np.min(t_st):6.1f} px", flush=True)

# Save outputs/long_refit_check.png
print("\nSaving outputs/long_refit_check.png...", flush=True)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
downscale = 4

for idx, (fname, blabel) in enumerate(blobs_to_investigate):
    ax = axes[idx]
    fc = frame_cache[fname]
    sub = fc["data"] - fc["bg"]
    sub_down = sub[::downscale, ::downscale]
    vmin, vmax = np.percentile(sub_down, [2, 98])
    ax.imshow(np.clip(sub_down, vmin, vmax), cmap="gray", origin="lower")
    
    res = refit_results[(fname, blabel)]
    s_c, s_r = res["start_pt"]
    e_c, e_r = res["end_pt"]
    
    # Draw refit line in red
    ax.plot([s_c / downscale, e_c / downscale], [s_r / downscale, e_r / downscale], "r-", lw=2.0, label="Refit TLS")
    
    # Draw ridge points in cyan dots
    rpts = res["ridge_pts"]
    ax.scatter(rpts[:, 0] / downscale, rpts[:, 1] / downscale, color="cyan", s=2, alpha=0.7)
    
    ax.set_title(f"({fname.replace('.fits','')}, {blabel})\nθ_h={res['angle_h_deg']:.1f}° | med_dist={res['med_perp_dist']:.2f}px", fontsize=9)
    ax.axis("off")

plt.tight_layout()
out_png = base_dir / "outputs/long_refit_check.png"
plt.savefig(out_png, dpi=150)
plt.close()
print(f"Saved {out_png}", flush=True)

print("\n" + "="*50, flush=True)
print("=== STEP 2: DISTANCES FROM SEEDS TO REFIT LINES ===", flush=True)
print("="*50, flush=True)

def dist_point_to_line(px, py, cx, cy, vx, vy):
    return abs((px - cx) * (-vy) + (py - cy) * vx)

# 20221025-0028: 16881 and 14253 to 6527
r6527 = refit_results[("20221025-0028.fits", 6527)]
for cid in [16881, 14253]:
    row = df_all[(df_all["frame_name"] == "20221025-0028.fits") & (df_all["candidate_id"] == cid)].iloc[0]
    scx, scy = row["centroid_col"], row["centroid_row"]
    d_perp = dist_point_to_line(scx, scy, r6527["center"][0], r6527["center"][1], r6527["direction"][0], r6527["direction"][1])
    print(f"20221025-0028: Seed {cid} to refit line 6527 = {d_perp:6.2f} px")

# 20230519-0009: 3673 to 5768
r5768 = refit_results[("20230519-0009.fits", 5768)]
row3673 = df_all[(df_all["frame_name"] == "20230519-0009.fits") & (df_all["candidate_id"] == 3673)].iloc[0]
scx, scy = row3673["centroid_col"], row3673["centroid_row"]
d_perp = dist_point_to_line(scx, scy, r5768["center"][0], r5768["center"][1], r5768["direction"][0], r5768["direction"][1])
print(f"20230519-0009: Seed 3673 to refit line 5768 = {d_perp:6.2f} px")

# 20231115-0003: 15632 and 15312 to line 888 / 12982 / stitched
r888 = refit_results[("20231115-0003.fits", 888)]
r12982 = refit_results[("20231115-0003.fits", 12982)]
rst = refit_results[("20231115-0003.fits", "stitched_888_12982")]
for cid in [15632, 15312]:
    row = df_all[(df_all["frame_name"] == "20231115-0003.fits") & (df_all["candidate_id"] == cid)].iloc[0]
    scx, scy = row["centroid_col"], row["centroid_row"]
    d_888 = dist_point_to_line(scx, scy, r888["center"][0], r888["center"][1], r888["direction"][0], r888["direction"][1])
    d_12982 = dist_point_to_line(scx, scy, r12982["center"][0], r12982["center"][1], r12982["direction"][0], r12982["direction"][1])
    d_st = dist_point_to_line(scx, scy, rst["center"][0], rst["center"][1], rst["direction"][0], rst["direction"][1])
    print(f"20231115-0003: Seed {cid} to refit 888={d_888:6.2f} px | to refit 12982={d_12982:6.2f} px | to refit stitched={d_st:6.2f} px")

# 20230602-0022: 19325 and 17284 to 11716
r11716 = refit_results[("20230602-0022.fits", 11716)]
for cid in [19325, 17284]:
    row = df_all[(df_all["frame_name"] == "20230602-0022.fits") & (df_all["candidate_id"] == cid)].iloc[0]
    scx, scy = row["centroid_col"], row["centroid_row"]
    d_perp = dist_point_to_line(scx, scy, r11716["center"][0], r11716["center"][1], r11716["direction"][0], r11716["direction"][1])
    print(f"20230602-0022: Seed {cid} to refit line 11716 = {d_perp:6.2f} px")

print("\n" + "="*50, flush=True)
print("=== STEP 3: HORIZONTAL ARTIFACT IN 20221026-0057 ===", flush=True)
print("="*50, flush=True)

d_57 = frame_cache["20221026-0057.fits"]["data"]
bg_57 = frame_cache["20221026-0057.fits"]["bg"]
std_57 = frame_cache["20221026-0057.fits"]["std"]
sub_57 = d_57 - bg_57
global_std_57 = float(np.median(std_57))

# Saturated star disc
# The giant star has saturated core disc around col 575
# Find saturated core centroid
sat_mask_57 = (d_57 >= 25000)
sat_props_57 = regionprops(label(sat_mask_57))
giant_prop = sorted(sat_props_57, key=lambda p: p.area, reverse=True)[0]

# Locate maximum brightness or centroid of the disc
# Inside the giant star bbox, the disc has the broadest saturated width across columns
sub_giant = d_57[giant_prop.bbox[0]:giant_prop.bbox[2], max(0, giant_prop.bbox[1]-50):min(4096, giant_prop.bbox[3]+50)]
# Row of peak intensity / disc center
disc_row_local = np.argmax(np.mean(sub_giant, axis=1))
giant_disc_row = giant_prop.bbox[0] + disc_row_local
print(f"Giant star disc: row = {giant_disc_row} (centroid row = {giant_prop.centroid[0]:.1f}, bbox = {giant_prop.bbox})")

# Mask 60 px around stars
star_mask = (d_57 >= 25000) | (sub_57 > 5.0 * std_57)
# Use uniform filter / max filter for fast 60 px dilation
from scipy.ndimage import maximum_filter
star_mask_dilated = maximum_filter(star_mask, size=121) # 60 px radius

# Rows 250 to 550, cols 100 to 4000
row_indices = np.arange(250, 551)
row_medians_all = []
row_medians_left = [] # cols 0-500
row_medians_right = [] # cols 700-4000

for r in row_indices:
    valid_cols_all = np.where(~star_mask_dilated[r, 100:4001])[0] + 100
    row_medians_all.append(float(np.median(sub_57[r, valid_cols_all])) if len(valid_cols_all) > 0 else 0.0)
        
    valid_cols_left = np.where(~star_mask_dilated[r, 0:501])[0]
    row_medians_left.append(float(np.median(sub_57[r, valid_cols_left])) if len(valid_cols_left) > 0 else 0.0)
        
    valid_cols_right = np.where(~star_mask_dilated[r, 700:4001])[0] + 700
    row_medians_right.append(float(np.median(sub_57[r, valid_cols_right])) if len(valid_cols_right) > 0 else 0.0)

row_medians_all = np.array(row_medians_all)
row_medians_left = np.array(row_medians_left)
row_medians_right = np.array(row_medians_right)

peak_idx = int(np.argmax(row_medians_all))
peak_row = row_indices[peak_idx]
peak_signal = row_medians_all[peak_idx]
peak_sigma = peak_signal / global_std_57

left_peak_signal = row_medians_left[peak_idx]
left_peak_sigma = left_peak_signal / global_std_57

right_peak_signal = row_medians_right[peak_idx]
right_peak_sigma = right_peak_signal / global_std_57

print(f"Across cols 100-4000: Peak row = {peak_row}, signal = {peak_signal:.2f} ADU, height = {peak_sigma:.2f} sigma")
print(f"Left of giant star (cols 0-500) at row {peak_row}: signal = {left_peak_signal:.2f} ADU, height = {left_peak_sigma:.2f} sigma")
print(f"Right of giant star (cols 700-4000) at row {peak_row}: signal = {right_peak_signal:.2f} ADU, height = {right_peak_sigma:.2f} sigma")
present_both = (left_peak_sigma > 2.0) and (right_peak_sigma > 2.0)
print(f"Present both left and right: {present_both} (left = {left_peak_sigma:.2f} sigma, right = {right_peak_sigma:.2f} sigma)")
print(f"Giant star disc row: row {giant_disc_row}")

print("\n" + "="*50, flush=True)
print("=== STEP 4: BLEED VETO TEST ===", flush=True)
print("="*50, flush=True)

# Bleed veto rule:
# angle within 10 degrees of vertical AND the line contains saturated pixels (>= 25000)
# Angle within 10 deg of vertical means: |angle_from_vertical| <= 10 deg,
# which corresponds to |angle_h| >= 80 deg.

flagged_labeled = []
for idx, row in df_all.iterrows():
    fname = row["frame_name"]
    cid = row["candidate_id"]
    lbl = row["label"]
    ang_h = row["angle_deg"]
    ang_from_vert = min(abs(90.0 - abs(ang_h)), abs(ang_h - 90.0))
    
    r0, c0, r1, c1 = int(row["bbox_r0"]), int(row["bbox_c0"]), int(row["bbox_r1"]), int(row["bbox_c1"])
    if fname in frame_cache:
        fd = frame_cache[fname]["data"]
    else:
        fd, _ = load_fits(frame_dir / fname)
    has_sat = bool(np.any(fd[r0:r1, c0:c1] >= 25000))
    
    if ang_from_vert <= 10.0 and has_sat:
        flagged_labeled.append((fname, cid, lbl, ang_from_vert, has_sat))

print(f"Flagged labeled candidates count: {len(flagged_labeled)}")
for f in flagged_labeled:
    print(f"  Labeled flagged: {f}")

flagged_new = []
for fname, blabel in blobs_to_investigate:
    res = refit_results[(fname, blabel)]
    ang_h = res["angle_h_deg"]
    ang_from_vert = min(abs(90.0 - abs(ang_h)), abs(ang_h - 90.0))
    r0, c0, r1, c1 = res["bbox"]
    fd = frame_cache[fname]["data"]
    has_sat = bool(np.any(fd[r0:r1, c0:c1] >= 25000))
    
    if ang_from_vert <= 10.0 and has_sat:
        flagged_new.append((fname, blabel, ang_from_vert, has_sat))

print(f"Flagged new candidates count: {len(flagged_new)}")
for f in flagged_new:
    print(f"  New flagged: {f}")

print("\nALL STEPS COMPLETED SUCCESSFULLY.", flush=True)
