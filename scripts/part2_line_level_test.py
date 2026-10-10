import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops, label

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background

frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
offsets = [-500, -400, -300, -200, -160, -120, -80, -40, 40, 80, 120, 160, 200, 300, 400, 500]
half_width = 4.0
w_offsets = np.arange(-half_width, half_width + 1.0, 1.0)

lines_to_test = [
    {
        "name": "(a) 20230317-0008 (cand 3618 at -24.8 deg)",
        "fname": "20230317-0008.fits",
        "p0": np.array([0.0, 1557.1]),
        "p1": np.array([3370.0, 0.0])
    },
    {
        "name": "(b) 20240405-0004 (TLS line through 5 fragments)",
        "fname": "20240405-0004.fits",
        "p0": np.array([0.0, 1667.4]),
        "p1": np.array([4096.0, 3328.4])
    },
    {
        "name": "(c) 20221025-0028 (through 16881 and 14253)",
        "fname": "20221025-0028.fits",
        "p0": np.array([0.0, 1070.6]),
        "p1": np.array([4096.0, 2973.5])
    }
]

for cfg in lines_to_test:
    print(f"\n======================================================================")
    print(f"LINE: {cfg['name']}")
    print(f"======================================================================")
    
    fname = cfg["fname"]
    data, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(data, method="background2d")
    sub_sig = (data - bg) / np.maximum(std, 1e-4)
    
    sat_props = regionprops(label(data >= 25000))
    sat_c = [sp.centroid for sp in sat_props]
    
    p0 = cfg["p0"]
    p1 = cfg["p1"]
    line_vec = p1 - p0
    tot_len = float(np.hypot(line_vec[0], line_vec[1]))
    ux, uy = line_vec / tot_len
    perp_ux, perp_uy = -uy, ux
    
    # Pre-extract profile for center line (offset=0) and all 16 controls
    all_offsets = [0] + offsets
    line_profiles = {}
    
    s_dense = np.arange(0, tot_len, 1.0)
    
    for off in all_offsets:
        xs = p0[0] + off * perp_ux + s_dense * ux
        ys = p0[1] + off * perp_uy + s_dense * uy
        
        in_b = (xs >= 0) & (xs < 4096) & (ys >= 0) & (ys < 4096)
        
        cy_g = ys[:, None] + w_offsets[None, :] * perp_uy
        cx_g = xs[:, None] + w_offsets[None, :] * perp_ux
        
        vals = map_coordinates(sub_sig, np.array([cy_g.ravel(), cx_g.ravel()]), order=1, mode='constant', cval=0.0).reshape((len(s_dense), len(w_offsets)))
        med_snr = np.median(vals, axis=1)
        
        # Star masking
        mask = (med_snr > 5.0) | (~in_b)
        for scy, scx in sat_c:
            mask |= (np.hypot(xs - scx, ys - scy) <= 60.0)
            
        med_clean = np.where(mask, np.nan, med_snr)
        line_profiles[off] = med_clean

    def evaluate_windows(win_size):
        print(f"\n--- {win_size} px Windows ---")
        print(f"{'Window Start':<14} | {'Line Val':<10} | {'Ctrl Mean':<10} | {'Ctrl Std':<10} | {'Z-score':<8}")
        print("-" * 62)
        n_wins = int(tot_len // win_size)
        ctrl_win_vals = {off: [] for off in offsets}
        
        for w_i in range(n_wins):
            w_start = w_i * win_size
            w_end = w_start + win_size
            
            # Line val
            l_chk = line_profiles[0][w_start:w_end]
            l_v = l_chk[~np.isnan(l_chk)]
            line_val = float(np.mean(l_v)) if len(l_v) >= 50 else np.nan
            
            # Control vals
            c_means = []
            for off in offsets:
                c_chk = line_profiles[off][w_start:w_end]
                cv = c_chk[~np.isnan(c_chk)]
                c_val = float(np.mean(cv)) if len(cv) >= 50 else np.nan
                ctrl_win_vals[off].append(c_val)
                if not np.isnan(c_val):
                    c_means.append(c_val)
                    
            c_mean = float(np.mean(c_means)) if len(c_means) > 0 else np.nan
            c_std = float(np.std(c_means, ddof=1)) if len(c_means) > 1 else np.nan
            z = (line_val - c_mean) / c_std if (c_std > 0 and not np.isnan(line_val)) else np.nan
            
            z_str = f"{z:8.2f}" if not np.isnan(z) else "     NaN"
            print(f"{w_start:<14} | {line_val:<10.3f} | {c_mean:<10.3f} | {c_std:<10.3f} | {z_str}")
            
        return ctrl_win_vals

    ctrl_500 = evaluate_windows(500)
    _ = evaluate_windows(1000)
    
    # Empirical false alarm rate on 16 controls
    # Each control treated as candidate line, tested against other 15 controls
    n_500_wins = int(tot_len // 500)
    z_max_per_ctrl = []
    
    for i, target_off in enumerate(offsets):
        other_offs = [o for o in offsets if o != target_off]
        ctrl_z_max = -999.0
        for w_i in range(n_500_wins):
            target_v = ctrl_500[target_off][w_i]
            other_v = [ctrl_500[o][w_i] for o in other_offs if not np.isnan(ctrl_500[o][w_i])]
            if not np.isnan(target_v) and len(other_v) > 1:
                om = float(np.mean(other_v))
                os = float(np.std(other_v, ddof=1))
                if os > 0:
                    z = (target_v - om) / os
                    if z > ctrl_z_max:
                        ctrl_z_max = z
        z_max_per_ctrl.append(ctrl_z_max)
        
    z_max_arr = np.array(z_max_per_ctrl)
    fa_gt_3 = int(np.sum(z_max_arr > 3.0))
    fa_gt_4 = int(np.sum(z_max_arr > 4.0))
    fa_gt_5 = int(np.sum(z_max_arr > 5.0))
    
    print(f"\nEmpirical False Alarm Rate (16 controls, any 500px window):")
    print(f"  z > 3: {fa_gt_3} / 16 ({fa_gt_3/16*100:.1f}%)")
    print(f"  z > 4: {fa_gt_4} / 16 ({fa_gt_4/16*100:.1f}%)")
    print(f"  z > 5: {fa_gt_5} / 16 ({fa_gt_5/16*100:.1f}%)")
    print(f"  Max z-scores across 16 controls: {np.round(z_max_arr, 2)}")
