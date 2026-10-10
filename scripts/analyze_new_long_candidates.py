import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

with open(base_dir / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg.get("detection", {})
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
bypass_cfg = det_cfg.get("long_thin_bypass", {})

df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)

# The 8 new long blobs from find_blob_candidates:
# (frame, blob_label)
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

# We need data for frames involved:
frames_needed = list(dict.fromkeys([b[0] for b in blobs_to_investigate] + ["20221101-0030.fits"]))

frame_data = {}
for fname in frames_needed:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    
    # Saturated cores
    sat_p = regionprops(label(d >= 25000))
    sat_c = [sp.centroid for sp in sat_p] # (row, col)
    
    # Run find_blob_candidates with bypass
    cands, _, _ = find_blob_candidates(
        d, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass=bypass_cfg
    )
    stitched = stitch_collinear_fragments(cands, d, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score)
    
    # Also find without bypass to identify newly selected candidates
    cands_no, _, _ = find_blob_candidates(
        d, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass_enabled=False
    )
    stitched_no = stitch_collinear_fragments(cands_no, d, bg, std)
    selected_no, _ = rank_and_select_candidates(stitched_no, min_abs_score=min_abs_score)
    old_cids = {c["label"] for c in selected_no}
    
    new_selected = [c for c in selected if c["label"] not in old_cids]
    
    frame_data[fname] = {
        "data": d, "bg": bg, "std": std, "sat_c": sat_c,
        "cands": cands, "stitched": stitched, "selected": selected,
        "new_selected": new_selected
    }

print("=== STEP 1: 8 NEW LONG CANDIDATES ===")
# For each of the 8:
# frame, PCA angle from vertical, bbox, aspect, distance from axis to nearest sat core
blob_info = {}

for fname, blabel in blobs_to_investigate:
    fd = frame_data[fname]
    cand = [c for c in fd["cands"] if c["label"] == blabel][0]
    
    # PCA angle from vertical:
    # Centroid: (cy, cx), BBox: (r0, c0, r1, c1)
    # Using candidate pixels or regionprop orientation
    # In skimage, orientation is angle between 0th axis (vertical/rows) and major axis, in radians, in [-pi/2, pi/2]
    # So angle from vertical in degrees = abs(orientation in deg)
    # Let's compute directly via PCA on pixels above threshold in bbox
    r0, c0, r1, c1 = cand["bbox_full"]
    sub_crop = (fd["data"][r0:r1, c0:c1] - fd["bg"][r0:r1, c0:c1])
    std_crop = fd["std"][r0:r1, c0:c1] if isinstance(fd["std"], np.ndarray) else fd["std"]
    y_idx, x_idx = np.where(sub_crop > 3.0 * std_crop)
    
    pts_y = r0 + y_idx
    pts_x = c0 + x_idx
    w = sub_crop[y_idx, x_idx]
    
    cy = np.average(pts_y, weights=w)
    cx = np.average(pts_x, weights=w)
    dy = pts_y - cy
    dx = pts_x - cx
    cov = np.cov(np.vstack([dx, dy]), aweights=w)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)] # (ux, uy)
    if vec[0] < 0:
        vec = -vec
        
    # Angle from horizontal: theta_h = arctan2(uy, ux)
    # Angle from vertical (y-axis): vertical unit vector is (0, 1) or (0, -1)
    # angle_from_vert = abs(90 - abs(theta_h)) or arccos(abs(uy))
    ang_h_deg = np.degrees(np.arctan2(vec[1], vec[0]))
    ang_from_vert_deg = float(np.degrees(np.arccos(abs(vec[1]))))
    
    # Distance from line axis to nearest saturated core:
    # Axis line passes through (cx, cy) with unit direction (ux, uy)
    ux, uy = vec[0], vec[1]
    perp_ux, perp_uy = -uy, ux
    
    min_sat_dist = 99999.0
    for scy, scx in fd["sat_c"]:
        # check distance from core to line
        d_perp = abs((scx - cx) * perp_ux + (scy - cy) * perp_uy)
        # check if projection lies within bbox span
        d_par = (scx - cx) * ux + (scy - cy) * uy
        # if touching saturated pixels (within 10 px of line)
        if d_perp < min_sat_dist:
            min_sat_dist = d_perp
            
    # Check if any saturated pixel lies inside the blob
    sat_in_blob = np.any(fd["data"][r0:r1, c0:c1] >= 25000)
    axis_sat_dist = 0.0 if sat_in_blob else min_sat_dist
    
    blob_info[(fname, blabel)] = {
        "cx": cx, "cy": cy, "ux": ux, "uy": uy,
        "ang_from_vert": ang_from_vert_deg,
        "ang_h": ang_h_deg,
        "bbox": (r0, c0, r1, c1),
        "aspect": cand["aspect_ratio"],
        "major": cand["axis_major_length"],
        "area": cand["area"],
        "sat_dist": axis_sat_dist
    }
    
    print(f"({fname}, {blabel:<5}) | angle_from_vert: {ang_from_vert_deg:5.1f}° | aspect: {cand['aspect_ratio']:6.1f} | major: {cand['axis_major_length']:6.1f}px | sat_dist: {axis_sat_dist:5.1f}px | bbox: {r0}:{r1}, {c0}:{c1}")

print("\n--- NEW SELECTED CANDIDATES PER FRAME ---")
for fname in frames_needed:
    new_sel = frame_data[fname]["new_selected"]
    print(f"[{fname}] New selected count: {len(new_sel)}")
    for c in new_sel:
        cid = c["label"]
        r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
        score = c.get("composite_score", c.get("score", 0.0))
        stitched_from = c.get("stitched_from", [cid])
        print(f"  Selected ID {cid}: bbox=({r0}, {c0}, {r1}, {c1}), score={score:.2f}, stitched_from={stitched_from}")

# Step 2: Duplicates in 20221025-0028 and 20230519-0009
print("\n=== STEP 2: DUPLICATES (20221025-0028 & 20230519-0009) ===")
# 20221025-0028: compare blob 6527 with 16881 and 14253
b28 = blob_info[("20221025-0028.fits", 6527)]
for cid in [16881, 14253]:
    r = df_all[(df_all["frame_name"] == "20221025-0028.fits") & (df_all["candidate_id"] == cid)].iloc[0]
    scx, scy = r["centroid_col"], r["centroid_row"]
    perp_d = abs((scx - b28["cx"]) * (-b28["uy"]) + (scy - b28["cy"]) * b28["ux"])
    ang_diff = abs(b28["ang_h"] - r["angle_deg"])
    ang_diff = min(ang_diff, 180.0 - ang_diff)
    print(f"20221025-0028 (6527 vs {cid}): perp_offset = {perp_d:5.2f} px, angle_diff = {ang_diff:5.2f}°")

# 20230519-0009: compare blob 5768 with 3673
b09 = blob_info[("20230519-0009.fits", 5768)]
r3673 = df_all[(df_all["frame_name"] == "20230519-0009.fits") & (df_all["candidate_id"] == 3673)].iloc[0]
scx, scy = r3673["centroid_col"], r3673["centroid_row"]
perp_d = abs((scx - b09["cx"]) * (-b09["uy"]) + (scy - b09["cy"]) * b09["ux"])
ang_diff = abs(b09["ang_h"] - r3673["angle_deg"])
ang_diff = min(ang_diff, 180.0 - ang_diff)
print(f"20230519-0009 (5768 vs 3673): perp_offset = {perp_d:5.2f} px, angle_diff = {ang_diff:5.2f}°")

# Step 3: Collinearity of labeled FP seeds with new lines
print("\n=== STEP 3: COLLINEARITY OF LABELED FP SEEDS WITH NEW LINES ===")
fp_checks = [
    ("20231115-0003.fits", [15632, 15312], [888, 12982]),
    ("20230602-0022.fits", [19325, 17284], [11716]),
    ("20221101-0030.fits", [9983], [])
]

for fname, fp_cids, new_blob_labels in fp_checks:
    for fpid in fp_cids:
        r = df_all[(df_all["frame_name"] == fname) & (df_all["candidate_id"] == fpid)].iloc[0]
        fcx, fcy = r["centroid_col"], r["centroid_row"]
        if not new_blob_labels:
            print(f"[{fname}] FP {fpid}: No new long lines in this frame.")
        else:
            for nbl in new_blob_labels:
                nb = blob_info[(fname, nbl)]
                p_dist = abs((fcx - nb["cx"]) * (-nb["uy"]) + (fcy - nb["cy"]) * nb["ux"])
                print(f"[{fname}] FP {fpid} to Line {nbl}: perp_dist = {p_dist:6.2f} px (centroid=({fcy:.1f}, {fcx:.1f}))")

# Step 4: 20231115-0003: Blobs 888 and 12982 collinearity & why +1
print("\n=== STEP 4: 20231115-0003 ANALYSIS ===")
b888 = blob_info[("20231115-0003.fits", 888)]
b12982 = blob_info[("20231115-0003.fits", 12982)]

p_dist_888_12982 = abs((b12982["cx"] - b888["cx"]) * (-b888["uy"]) + (b12982["cy"] - b888["cy"]) * b888["ux"])
ang_diff_888_12982 = abs(b888["ang_h"] - b12982["ang_h"])
ang_diff_888_12982 = min(ang_diff_888_12982, 180.0 - ang_diff_888_12982)
print(f"Blob 888 vs 12982: perp_offset = {p_dist_888_12982:.2f} px, angle_diff = {ang_diff_888_12982:.2f}°")

# Check stitching of 888 and 12982 in frame_data["20231115-0003.fits"]["stitched"]:
st_15 = frame_data["20231115-0003.fits"]["stitched"]
for s in st_15:
    from_list = s.get("stitched_from", [s["label"]])
    if 888 in from_list or 12982 in from_list:
        print(f"Stitched candidate {s['label']}: stitched_from={from_list}, score={s.get('composite_score', s.get('score')):.2f}")

# Step 5: 20221026-0057: Row band of giant saturated star, 1175, and blob A (1183)
print("\n=== STEP 5: 20221026-0057 ROW BANDS ===")
d_57 = frame_data["20221026-0057.fits"]["data"]
sat_p_57 = regionprops(label(d_57 >= 25000))
# Giant saturated star is the largest saturated core
giant_star = sorted(sat_p_57, key=lambda p: p.area, reverse=True)[0]
print(f"Giant Saturated Star: row band = [{giant_star.bbox[0]}, {giant_star.bbox[2]}], col band = [{giant_star.bbox[1]}, {giant_star.bbox[3]}], area = {giant_star.area}")

r1175 = df_all[(df_all["frame_name"] == "20221026-0057.fits") & (df_all["candidate_id"] == 1175)].iloc[0]
print(f"Streak 1175: row band = [{int(r1175['bbox_r0'])}, {int(r1175['bbox_r1'])}], col band = [{int(r1175['bbox_c0'])}, {int(r1175['bbox_c1'])}]")

b1183 = blob_info[("20221026-0057.fits", 1183)]
print(f"Blob A (1183): row band = [{b1183['bbox'][0]}, {b1183['bbox'][2]}], col band = [{b1183['bbox'][1]}, {b1183['bbox'][3]}]")

# Step 6: Save outputs/new_long_candidates.png
print("\n=== STEP 6: SAVING outputs/new_long_candidates.png ===")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

downscale = 4

for idx, (fname, blabel) in enumerate(blobs_to_investigate):
    ax = axes[idx]
    fd = frame_data[fname]
    sub = fd["data"] - fd["bg"]
    sub_down = sub[::downscale, ::downscale]
    vmin, vmax = np.percentile(sub_down, [2, 98])
    ax.imshow(np.clip(sub_down, vmin, vmax), cmap='gray', origin='lower')
    
    b = blob_info[(fname, blabel)]
    r0, c0, r1, c1 = b["bbox"]
    w_box = (c1 - c0) / downscale
    h_box = (r1 - r0) / downscale
    rect = plt.Rectangle((c0 / downscale, r0 / downscale), w_box, h_box, edgecolor='cyan', facecolor='none', lw=1.2)
    ax.add_patch(rect)
    
    # Draw fitted axis line across bbox
    t_span = np.array([-b["major"]/2.0, b["major"]/2.0])
    lx = (b["cx"] + t_span * b["ux"]) / downscale
    ly = (b["cy"] + t_span * b["uy"]) / downscale
    ax.plot(lx, ly, 'r--', lw=1.5)
    
    ax.set_title(f"({fname.replace('.fits','')}, {blabel})\nθ_vert={b['ang_from_vert']:.1f}° | Asp={b['aspect']:.0f}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
out_fig_path = base_dir / "outputs/new_long_candidates.png"
plt.savefig(out_fig_path, dpi=150)
plt.close()
print(f"Saved {out_fig_path}")
