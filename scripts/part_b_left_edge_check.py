import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background

fpath = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")
data, header = load_fits(fpath)
bg, std = estimate_background(data, method="background2d")
sub = data - bg

# 1. Panels:
# Cand 66 bbox: r0=19, r1=350, c0=0, c1=39
# Star at r=839, c=41
# Cand 3767 bbox: r0=1073, r1=1169, c0=0, c1=12
# Cand 7217 bbox: r0=2025, r1=2225, c0=0, c1=54

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 4)

panels = [
    ("Cand 66 (rows 0-450, cols 0-200)", max(0, 19-100), min(4096, 350+100), (19, 0, 350, 39), gs[0, 0]),
    ("Saturated Star (rows 739-939, cols 0-200)", 839-100, 839+100, (834, 38, 844, 45), gs[0, 1]),
    ("Cand 3767 (rows 973-1269, cols 0-200)", max(0, 1073-100), min(4096, 1169+100), (1073, 0, 1169, 12), gs[0, 2]),
    ("Cand 7217 (rows 1925-2325, cols 0-200)", max(0, 2025-100), min(4096, 2225+100), (2025, 0, 2225, 54), gs[0, 3]),
]

for title, r_start, r_end, bbox, pos in panels:
    ax = fig.add_subplot(pos)
    strip = data[r_start:r_end, 0:200]
    p_vmin, p_vmax = float(np.percentile(strip, 5)), float(np.percentile(strip, 98))
    ax.imshow(strip, cmap="gray", origin="lower", extent=[0, 200, r_start, r_end], vmin=p_vmin, vmax=p_vmax, aspect='auto', interpolation='nearest')
    
    r0, c0, r1, c1 = bbox
    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.axvline(x=41, color="yellow", linestyle=":", alpha=0.7, label="Star col=41")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    if pos == gs[0, 0]:
        ax.legend(loc="upper right", fontsize=8)

# Lower panel: Mean background-subtracted brightness vs column for cols 0-200 (median over rows 100-3900)
ax_prof = fig.add_subplot(gs[1, :])
sub_strip = sub[100:3900, 0:200]
# Median over rows 100-3900 across columns
col_prof = np.median(sub_strip, axis=0)
col_mean_prof = np.mean(sub_strip, axis=0)

cols = np.arange(200)
ax_prof.plot(cols, col_prof, color="blue", lw=2, label="Median brightness over rows 100-3900")
ax_prof.plot(cols, col_mean_prof, color="cyan", lw=1, linestyle="--", alpha=0.7, label="Mean brightness over rows 100-3900")
ax_prof.axvline(x=41, color="yellow", linestyle=":", lw=2, label="Star column 41")
ax_prof.axvline(x=12, color="magenta", linestyle=":", lw=1.5, label="Cand 3767 edge (col 12)")
ax_prof.set_xlim(0, 200)
ax_prof.set_title("Column Brightness Profile (cols 0-200, rows 100-3900)", fontsize=11)
ax_prof.set_xlabel("Column (px)")
ax_prof.set_ylabel("Background-subtracted Brightness (ADU)")
ax_prof.grid(True, alpha=0.3)
ax_prof.legend(loc="upper right", fontsize=9)

plt.tight_layout()
out_p = Path("outputs/left_edge_check.png")
fig.savefig(out_p, dpi=130)
plt.close(fig)
print(f"Saved {out_p}")

# Analysis printouts:
print("\n--- Column 0-200 Analysis ---")
print(f"Brightness at col 0-10: {np.mean(col_prof[0:10]):.2f} ADU")
print(f"Brightness at col 41 (star col): {col_prof[41]:.2f} ADU")
print(f"Brightness at col 100-200: {np.mean(col_prof[100:200]):.2f} ADU")
print(f"Star at row 839 col 41. Cand 3767 is at cols {0}..{12}.")
print(f"Is Cand 3767 (cols 0-12) in the same column as star (col 41)? {'YES' if 0 <= 41 <= 12 else 'NO (30 px offset)'}")
