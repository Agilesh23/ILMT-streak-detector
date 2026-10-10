import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops, label

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

# 1. Compute T from the 16 control lines in 20230317-0008
print("=== 1. COMPUTING T FROM 16 CONTROL LINES (20230317-0008) ===")
fpath_08 = frame_dir / "20230317-0008.fits"
data_08, _ = load_fits(fpath_08)
bg_08, std_08 = estimate_background(data_08, method="background2d")
sub_sigma_08 = (data_08 - bg_08) / np.maximum(std_08, 1e-4)

# Saturated star cores in 20230317-0008
sat_props_08 = regionprops(label(data_08 >= 25000))
sat_centers_08 = [sp.centroid for sp in sat_props_08]

x0_base, y0_base = 0.0, 1557.0
x1_base, y1_base = 3370.0, 0.0
dx_base = x1_base - x0_base
dy_base = y1_base - y0_base
len_base = np.hypot(dx_base, dy_base)
ux_08 = dx_base / len_base
uy_08 = dy_base / len_base
perp_ux_08 = -uy_08
perp_uy_08 = ux_08

offsets = [-500, -400, -300, -200, -160, -120, -80, -40, 40, 80, 120, 160, 200, 300, 400, 500]
half_width = 4.0
w_offsets = np.arange(-half_width, half_width + 1.0, 1.0)

all_200px_means = []

for off in offsets:
    s_dense = np.arange(-1000, len_base + 1000, 1.0)
    xs = x0_base + off * perp_ux_08 + s_dense * ux_08
    ys = y0_base + off * perp_uy_08 + s_dense * uy_08
    in_bounds = (xs >= 0) & (xs < 4096) & (ys >= 0) & (ys < 4096)
    xs = xs[in_bounds]
    ys = ys[in_bounds]
    
    # Transverse sample
    cy_grid = ys[:, None] + w_offsets[None, :] * perp_uy_08
    cx_grid = xs[:, None] + w_offsets[None, :] * perp_ux_08
    vals = map_coordinates(sub_sigma_08, np.array([cy_grid.ravel(), cx_grid.ravel()]), order=1, mode='constant', cval=0.0)
    vals = vals.reshape((len(xs), len(w_offsets)))
    med_snr = np.median(vals, axis=1)
    
    # Mask stars: 60px from sat cores or > 5 sigma
    mask = med_snr > 5.0
    for scy, scx in sat_centers_08:
        mask |= (np.hypot(xs - scx, ys - scy) <= 60.0)
        
    med_clean = med_snr.copy()
    med_clean[mask] = np.nan
    
    # Non-overlapping 200 px windows
    n_win = len(med_clean) // 200
    for w_i in range(n_win):
        chunk = med_clean[w_i * 200 : (w_i + 1) * 200]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= 100: # at least half unmasked
            all_200px_means.append(float(np.mean(valid)))

std_200px = float(np.std(all_200px_means))
T = round(3.0 * std_200px, 3)
print(f"200px control windows analyzed: {len(all_200px_means)}")
print(f"Std of 200px averages: {std_200px:.4f} sigma")
print(f"Chosen T = 3 * std = {T} sigma")

# Function to extend candidate
def extend_by_integration(c, data, bg, std, T_val):
    sub_sig = (data - bg) / np.maximum(std, 1e-4)
    h, w = data.shape
    
    # Saturated cores
    sat_p = regionprops(label(data >= 25000))
    sat_c = [sp.centroid for sp in sat_p]
    
    # Seed geometry from bbox/centroid/angle
    cy, cx = c["centroid_row"], c["centroid_col"]
    ang_deg = c["angle_deg"]
    ang_rad = np.radians(ang_deg)
    ux = np.cos(ang_rad)
    uy = np.sin(ang_rad)
    perp_ux = -uy
    perp_uy = ux
    
    seed_len = c["length_px"]
    half_seed = seed_len / 2.0
    
    # March bidirectionally from seed
    def march_direction(dir_sign):
        curr_dist = half_seed
        trailing_buffer = [] # 1px step samples
        consec_below = 0.0
        best_dist = half_seed
        
        while True:
            # Step in 50 px block
            block_steps = np.arange(curr_dist, curr_dist + 50.0, 1.0)
            xs_block = cx + dir_sign * block_steps * ux
            ys_block = cy + dir_sign * block_steps * uy
            
            # Check edge bounds
            in_b = (xs_block >= 0) & (xs_block < w) & (ys_block >= 0) & (ys_block < h)
            if not np.all(in_b):
                # hit edge
                valid_idx = np.where(in_b)[0]
                if len(valid_idx) > 0:
                    best_dist = block_steps[valid_idx[-1]]
                break
                
            # Sample transverse
            cy_g = ys_block[:, None] + w_offsets[None, :] * perp_uy
            cx_g = xs_block[:, None] + w_offsets[None, :] * perp_ux
            v = map_coordinates(sub_sig, np.array([cy_g.ravel(), cx_g.ravel()]), order=1, mode='constant', cval=0.0).reshape((len(block_steps), len(w_offsets)))
            med_snr = np.median(v, axis=1)
            
            # Star mask
            mask = med_snr > 5.0
            for scy, scx in sat_c:
                mask |= (np.hypot(xs_block - scx, ys_block - scy) <= 60.0)
                
            med_clean = np.where(mask, np.nan, med_snr)
            
            # Append to buffer
            trailing_buffer.extend(med_clean)
            
            # Evaluate trailing 200 px average
            last_200 = trailing_buffer[-200:]
            valid_200 = [val for val in last_200 if not np.isnan(val)]
            avg_200 = np.mean(valid_200) if len(valid_200) >= 50 else 0.0
            
            if avg_200 >= T_val:
                consec_below = 0.0
                best_dist = curr_dist + 50.0
            else:
                consec_below += 50.0
                if consec_below >= 300.0:
                    break
                    
            curr_dist += 50.0
            if curr_dist > 5000.0:
                break
                
        return best_dist

    dist_fwd = march_direction(+1.0)
    dist_bwd = march_direction(-1.0)
    
    p_start = (cx - dist_bwd * ux, cy - dist_bwd * uy)
    p_end = (cx + dist_fwd * ux, cy + dist_fwd * uy)
    tot_len = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
    
    # Measure mean SNR along extended line
    s_full = np.arange(0, tot_len, 2.0)
    xf = p_start[0] + s_full * (p_end[0] - p_start[0]) / max(tot_len, 1e-3)
    yf = p_start[1] + s_full * (p_end[1] - p_start[1]) / max(tot_len, 1e-3)
    cy_full = yf[:, None] + w_offsets[None, :] * perp_uy
    cx_full = xf[:, None] + w_offsets[None, :] * perp_ux
    vf = map_coordinates(sub_sig, np.array([cy_full.ravel(), cx_full.ravel()]), order=1, mode='constant', cval=0.0).reshape((len(s_full), len(w_offsets)))
    full_snrs = np.median(vf, axis=1)
    mean_full_snr = float(np.mean(full_snrs)) if len(full_snrs) > 0 else 0.0
    
    return p_start, p_end, round(tot_len, 1), round(mean_full_snr, 2)

# 2. Run on specified candidates:
# 3618 (20230317-0008), 5 fragments in 20240405-0004, 2 in 20221025-0028, and 3673
print("\n=== 2. EXTENSION ON SPECIFIED STREAKS ===")
target_runs = [
    ("20230317-0008.fits", [3618]),
    ("20240405-0004.fits", [14613, 12522, 17987, 9558, 10716]),
    ("20221025-0028.fits", [16881, 14253]),
    ("20230519-0009.fits", [3673])
]

df_labels = pd.concat([
    pd.read_csv("data/labeled/batch_1_v2_labels.csv"),
    pd.read_csv("data/labeled/batch_2_v2_labels.csv")
]).reset_index(drop=True)

res_rows = []
for fname, cids in target_runs:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    for cid in cids:
        c_row = df_labels[(df_labels["frame_name"] == fname) & (df_labels["candidate_id"] == cid)].iloc[0]
        p_start, p_end, tot_len, mean_snr = extend_by_integration(c_row, d, bg, std, T)
        p0_str = f"({p_start[0]:.1f}, {p_start[1]:.1f})"
        p1_str = f"({p_end[0]:.1f}, {p_end[1]:.1f})"
        res_rows.append({
            "frame": fname,
            "id": cid,
            "seed_len": c_row["length_px"],
            "start": p0_str,
            "end": p1_str,
            "length": tot_len,
            "mean_snr": mean_snr
        })

print(f"{'Frame':<18} | {'ID':<6} | {'Seed Len':<9} | {'Start (col, row)':<20} | {'End (col, row)':<20} | {'Length':<8} | {'Mean SNR':<8}")
print("-" * 105)
for r in res_rows:
    print(f"{r['frame']:<18} | {r['id']:<6} | {r['seed_len']:<9.1f} | {r['start']:<20} | {r['end']:<20} | {r['length']:<8.1f} | {r['mean_snr']:<8.2f}")

# 3. False-alarm test:
# 20 random spots and angles on empty sky (no stars within 100 px) in each of the 3 test frames.
print("\n=== 3. FALSE-ALARM TEST ===")
frames_3 = ["20221025-0028.fits", "20230519-0009.fits", "20230317-0008.fits"]
rng = np.random.RandomState(42)

for fname in frames_3:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    sub = d - bg
    
    # Detect all star positions (pixels > 5 sigma)
    stars = np.argwhere(sub > 5.0 * std)
    
    extended_count = 0
    tested_spots = 0
    
    while tested_spots < 20:
        rx = rng.uniform(200, 3800)
        ry = rng.uniform(200, 3800)
        
        # Check if empty sky: no stars within 100 px
        box_r0 = max(0, int(ry - 100))
        box_r1 = min(4096, int(ry + 100))
        box_c0 = max(0, int(rx - 100))
        box_c1 = min(4096, int(rx + 100))
        if np.any(sub[box_r0:box_r1, box_c0:box_c1] > 5.0 * (std[box_r0:box_r1, box_c0:box_c1] if isinstance(std, np.ndarray) else std)):
            continue
            
        r_angle = rng.uniform(0, 180)
        
        mock_c = {
            "centroid_row": ry,
            "centroid_col": rx,
            "angle_deg": r_angle,
            "length_px": 20.0
        }
        
        _, _, tot_len, _ = extend_by_integration(mock_c, d, bg, std, T)
        if (tot_len - 20.0) > 200.0:
            extended_count += 1
        tested_spots += 1
        
    print(f"[{fname}] Random empty-sky spots extending > 200px: {extended_count} / 20")

# Run on 81 FP seeds: list any growing beyond 500 px
print("\n--- 81 FP SEEDS (GROWN > 500 px) ---")
fps_over_500 = []
all_fps = df_labels[df_labels["label"] == "fp"]

for fname in all_fps["frame_name"].unique():
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    sub_fps = all_fps[all_fps["frame_name"] == fname]
    for _, c_row in sub_fps.iterrows():
        _, _, tot_len, m_snr = extend_by_integration(c_row, d, bg, std, T)
        if tot_len > 500.0:
            fps_over_500.append((c_row["frame_name"], int(c_row["candidate_id"]), c_row["length_px"], tot_len, m_snr))

print(f"Total FP seeds with grown length > 500px: {len(fps_over_500)}")
for fp in fps_over_500:
    print(f"  ({fp[0]}, {fp[1]}) | seed: {fp[2]:.1f}px, grown: {fp[3]:.1f}px, mean SNR: {fp[4]:.2f}")
