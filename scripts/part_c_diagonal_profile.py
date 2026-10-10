import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates, grow_streak

fpath = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")
data, header = load_fits(fpath)
bg, std = estimate_background(data, method="background2d")
sub_sigma = (data - bg) / np.maximum(std, 1e-4)

# Candidate 3618 location
# Centroid: row=1038.5, col=1122.5
c_row = 1038.5
c_col = 1122.5
angle_deg = -24.8
angle_rad = np.radians(angle_deg)
slope = np.tan(angle_rad)

# Direction vector (ux, uy) along the line:
# as col increases (x increases), row decreases (y decreases)
ux = np.cos(angle_rad) # positive
uy = np.sin(angle_rad) # negative: -0.4195
perp_ux = -uy
perp_uy = ux

# Find line endpoints within image bounds [0, 4096] x [0, 4096]
# Point at x: y(x) = c_row + slope * (x - c_col)
# x = 0 -> y = c_row - slope * c_col
# y = 0 -> x = c_col - c_row / slope
x_start = 0.0
y_start = c_row + slope * (x_start - c_col)
y_end = 0.0
x_end = c_col + (y_end - c_row) / slope

total_line_len = np.hypot(x_end - x_start, y_end - y_start)
print(f"Line from (col={x_start:.1f}, row={y_start:.1f}) to (col={x_end:.1f}, row={y_end:.1f})")
print(f"Total line length: {total_line_len:.1f} px, angle: {angle_deg:.1f} deg")

# Sample along line in 1.0 px steps
s_steps = np.arange(0, total_line_len, 1.0)
xs = x_start + s_steps * ((x_end - x_start) / total_line_len)
ys = y_start + s_steps * ((y_end - y_start) / total_line_len)

# Sample across streak width (+-4 px transverse, 9 samples)
half_width = 4.0
w_offsets = np.arange(-half_width, half_width + 1.0, 1.0)
snr_profile = []
raw_profile = []

for x, y in zip(xs, ys):
    sample_x = x + w_offsets * perp_ux
    sample_y = y + w_offsets * perp_uy
    vals = map_coordinates(sub_sigma, np.vstack([sample_y, sample_x]), order=1, mode='constant', cval=0.0)
    snr_profile.append(float(np.median(vals)))
    raw_vals = map_coordinates(data, np.vstack([sample_y, sample_x]), order=1, mode='constant', cval=0.0)
    raw_profile.append(float(np.median(raw_vals)))

snr_arr = np.array(snr_profile)
raw_arr = np.array(raw_profile)

# Saturated star regions (star at col ~863-870, row ~1145)
# Find distance along line to point closest to (col 863, row 1145):
star_col = 863.4
star_row = 1145.0
d_to_star = np.hypot(xs - star_col, ys - star_row)
min_star_dist = np.min(d_to_star)
star_s_idx = np.argmin(d_to_star)
star_s = s_steps[star_s_idx]
print(f"Nearest approach to saturated star at s={star_s:.1f}px (distance to star core: {min_star_dist:.1f}px)")

# Supported length: count pixels where SNR >= 1.0
supported_mask = snr_arr >= 1.0
supported_len = np.sum(supported_mask)
mean_snr = np.mean(snr_arr)
mean_supported_snr = np.mean(snr_arr[supported_mask]) if supported_len > 0 else 0.0

# Gaps: consecutive regions where SNR < 1.0
gaps = []
curr_gap = 0
for val in snr_arr:
    if val < 1.0:
        curr_gap += 1
    else:
        if curr_gap > 0:
            gaps.append(curr_gap)
            curr_gap = 0
if curr_gap > 0:
    gaps.append(curr_gap)

max_gap = max(gaps) if gaps else 0
print(f"Supported length (SNR >= 1.0 sigma): {supported_len} px / {len(snr_arr)} px ({supported_len/len(snr_arr)*100:.1f}%)")
print(f"Mean full SNR: {mean_snr:.2f} sigma, Mean supported SNR: {mean_supported_snr:.2f} sigma")
print(f"Number of gaps < 1.0 sigma: {len(gaps)}, Max gap: {max_gap} px, Gaps > 50px: {sum(g > 50 for g in gaps)}")

# Check why grow_streak stopped at 148 px:
cands_08, _, _ = find_blob_candidates(data, bg, std, edge_mask_max_x=3950, border_margin_px=15, border_max_length_px=100)
stitched_08 = stitch_collinear_fragments(cands_08, data, bg, std)
sel_08, _ = rank_and_select_candidates(stitched_08)
c3618 = next(c for c in sel_08 if c["label"] == 3618)

grown_3618 = grow_streak([c3618], data, bg, std)[0]
print(f"\ncand 3618 detected major length: {c3618['axis_major_length']:.1f}px, grown length: {grown_3618['total_length']:.1f}px")

# Trace march bidirectionally from 3618:
cy, cx = c3618["centroid"]
min_r, min_c, max_r, max_c = c3618.get("bbox_full", c3618.get("bbox_local"))
print(f"3618 bbox: ({min_r}, {min_c}) to ({max_r}, {max_c})")

# Let's inspect SNR profile around 3618 along the fitted line:
# Plot: 2 subplots:
# 1) Full-frame overlay showing the diagonal line and star
# 2) Profile vs distance along line, with saturated star region shaded
fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(18, 8))

vmin, vmax = float(np.percentile(data, 1)), float(np.percentile(data, 99))
ax_img.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
ax_img.plot([x_start, x_end], [y_start, y_end], color="cyan", lw=2, label=f"Fitted streak line (slope {angle_deg}°)")
ax_img.plot([grown_3618['p_start'][0], grown_3618['p_end'][0]], [grown_3618['p_start'][1], grown_3618['p_end'][1]], color="lime", lw=3.5, label=f"grow_streak segment ({grown_3618['total_length']:.1f}px)")
ax_img.scatter([star_col], [star_row], color="yellow", marker="*", s=150, zorder=5, label="Saturated star core")
ax_img.scatter([cx], [cy], color="red", marker="x", s=80, zorder=5, label="Cand 3618 centroid")
ax_img.set_xlim(0, 4096)
ax_img.set_ylim(0, 4096)
ax_img.set_title(f"20230317-0008 Full Frame Overlay\nFaint Diagonal Streak (slope {angle_deg}°)", fontsize=11)
ax_img.set_xlabel("Column (px)")
ax_img.set_ylabel("Row (px)")
ax_img.legend(loc="upper right", fontsize=9)

# Profile plot
ax_prof.plot(s_steps, snr_arr, color="blue", lw=1.2, label="Median transverse SNR")
ax_prof.axhline(y=1.0, color="gray", linestyle="--", label="1.0 sigma threshold")
ax_prof.axhline(y=1.5, color="orange", linestyle=":", label="1.5 sigma (grow_streak min_snr)")

# Shade saturated star region (within +-100px of nearest approach):
ax_prof.axvspan(star_s - 80, star_s + 80, color="yellow", alpha=0.3, label="Saturated star region")

# Mark cand 3618 position:
s_3618 = np.hypot(cx - x_start, cy - y_start)
ax_prof.axvline(x=s_3618, color="red", linestyle="-.", label=f"Cand 3618 (s={s_3618:.0f}px)")
ax_prof.axvspan(s_3618 - grown_3618['total_length']/2, s_3618 + grown_3618['total_length']/2, color="lime", alpha=0.2, label=f"Grown range ({grown_3618['total_length']:.0f}px)")

ax_prof.set_xlim(0, total_line_len)
ax_prof.set_ylim(-1, max(10, min(50, float(np.percentile(snr_arr, 99.5)))))
ax_prof.set_title(f"Median Streak Profile vs Distance along Line\nSupported length (SNR>1s): {supported_len}px | Mean SNR: {mean_snr:.2f}s", fontsize=11)
ax_prof.set_xlabel("Distance along line (px)")
ax_prof.set_ylabel("Median Transverse SNR (sigma)")
ax_prof.grid(True, alpha=0.3)
ax_prof.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
out_diag = Path("outputs/diagonal_profile_20230317-0008.png")
fig.savefig(out_diag, dpi=140)
plt.close(fig)
print(f"Saved {out_diag}")
