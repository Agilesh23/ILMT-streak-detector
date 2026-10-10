import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, uniform_filter1d
from skimage.measure import regionprops, label

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background

fpath = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")
data, header = load_fits(fpath)
bg, std = estimate_background(data, method="background2d")
sub_sigma = (data - bg) / np.maximum(std, 1e-4)

# Saturated star cores (pixels >= 25000)
sat_mask = data >= 25000
sat_props = regionprops(label(sat_mask))
sat_centers = [sp.centroid for sp in sat_props] # (row, col)

# Base line: (col 0, row 1557) to (col 3370, row 0)
x0_base, y0_base = 0.0, 1557.0
x1_base, y1_base = 3370.0, 0.0

dx_base = x1_base - x0_base
dy_base = y1_base - y0_base
len_base = np.hypot(dx_base, dy_base)
ux = dx_base / len_base
uy = dy_base / len_base

# Perpendicular unit vector:
# If line direction is (ux, uy), perp direction is (-uy, ux)
perp_ux = -uy
perp_uy = ux

offsets = [0, 40, -40, 80, -80, 120, -120, 160, -160, 200, -200, 300, -300, 400, -400, 500, -500]
# Sort for presentation
offsets_sorted = sorted(offsets)

half_width = 4.0
w_offsets = np.arange(-half_width, half_width + 1.0, 1.0)

results = []
profile_dict = {}

for offset in offsets_sorted:
    # Offset the line points by offset * (perp_ux, perp_uy)
    # x(s) = x0_base + offset * perp_ux + s * ux
    # y(s) = y0_base + offset * perp_uy + s * uy
    
    # We step along s from -1000 to len_base + 1000 in 1 px steps, keeping points inside [0, 4096] x [0, 4096]
    s_dense = np.arange(-1000, len_base + 1000, 1.0)
    xs_dense = x0_base + offset * perp_ux + s_dense * ux
    ys_dense = y0_base + offset * perp_uy + s_dense * uy
    
    in_bounds = (xs_dense >= 0) & (xs_dense < 4096) & (ys_dense >= 0) & (ys_dense < 4096)
    s_valid = s_dense[in_bounds]
    xs = xs_dense[in_bounds]
    ys = ys_dense[in_bounds]
    
    if len(s_valid) == 0:
        continue
        
    # Distance measured from the start of the valid segment (or from s=0)
    # Let's measure along the segment:
    seg_dist = np.arange(len(s_valid))
    
    # Sample transverse width +-4 px:
    coords_y = ys[:, None] + w_offsets[None, :] * perp_uy
    coords_x = xs[:, None] + w_offsets[None, :] * perp_ux
    
    vals = map_coordinates(sub_sigma, np.array([coords_y.ravel(), coords_x.ravel()]), order=1, mode='constant', cval=0.0)
    vals = vals.reshape((len(s_valid), len(w_offsets)))
    med_snr = np.median(vals, axis=1)
    
    # Star masking: mask 60 px around saturated cores and any pixel above 5 sigma
    # 1. Mask 60 px around saturated cores:
    mask_sat = np.zeros(len(s_valid), dtype=bool)
    for scy, scx in sat_centers:
        dist_to_core = np.hypot(xs - scx, ys - scy)
        mask_sat |= (dist_to_core <= 60.0)
        
    # 2. Mask any pixel above 5 sigma in the profile
    mask_5sig = (med_snr > 5.0)
    
    masked = mask_sat | mask_5sig
    
    # Apply mask
    snr_clean = med_snr.copy()
    snr_clean[masked] = np.nan
    
    # Valid (unmasked) samples
    unmasked = ~masked
    
    # Metrics:
    # 0-1800 px and 1800-end
    idx_0_1800 = (s_valid >= 0) & (s_valid < 1800) & unmasked
    idx_1800_end = (s_valid >= 1800) & unmasked
    idx_all = unmasked
    
    mean_0_1800 = np.nanmean(snr_clean[idx_0_1800]) if np.any(idx_0_1800) else np.nan
    mean_1800_end = np.nanmean(snr_clean[idx_1800_end]) if np.any(idx_1800_end) else np.nan
    frac_ge_1sig = np.mean(snr_clean[idx_all] >= 1.0) if np.any(idx_all) else np.nan
    
    results.append({
        "offset": offset,
        "mean_0_1800": mean_0_1800,
        "mean_1800_end": mean_1800_end,
        "frac_ge_1sig": frac_ge_1sig,
        "n_samples": len(s_valid)
    })
    
    # For smoothing, fill nan with 0 for running mean:
    snr_for_filter = np.where(np.isnan(snr_clean), 0.0, snr_clean)
    running_mean = uniform_filter1d(snr_for_filter, size=500, mode='nearest')
    profile_dict[offset] = (s_valid, running_mean)

# Print Table
print("=================== CONTROL TEST TABLE (20230317-0008) ===================")
print(f"{'Offset (px)':>11} | {'Mean SNR (0-1800px)':>20} | {'Mean SNR (1800-end)':>20} | {'Frac >= 1.0 sigma':>18}")
print("-" * 76)
for r in results:
    off_str = f"{r['offset']:+d} px" if r['offset'] != 0 else "0 px (diag)"
    print(f"{off_str:>11} | {r['mean_0_1800']:>20.3f} | {r['mean_1800_end']:>20.3f} | {r['frac_ge_1sig']:>18.3f}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
for offset in offsets_sorted:
    if offset in profile_dict:
        s_v, r_mean = profile_dict[offset]
        if offset == 0:
            ax.plot(s_v, r_mean, color="red", lw=2.5, label="Diagonal (offset = 0 px)", zorder=10)
        else:
            label_c = "Control lines (+/-40..500 px)" if offset == offsets_sorted[0] else ""
            ax.plot(s_v, r_mean, color="gray", alpha=0.45, lw=1.2, label=label_c)

ax.set_title("20230317-0008: 500 px Running Mean Profile vs Control Offsets", fontsize=12)
ax.set_xlabel("Distance along line (px)")
ax.set_ylabel("500 px Running Mean SNR (sigma)")
ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
out_png = Path("outputs/diagonal_control.png")
fig.savefig(out_png, dpi=140)
plt.close(fig)
print(f"Saved {out_png}")
