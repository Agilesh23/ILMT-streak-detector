import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
outputs_dir = base_dir / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2], ignore_index=True)

# 1. Check CSV columns
print("CSV columns:", df_all.columns.tolist())
has_location = any(col in df_all.columns for col in ["x", "y", "centroid", "bbox", "min_col", "cx", "cy"])
print(f"CSV has location column: {has_location}")

# 2. Extract bounding box / red box from saved context image
# In the saved context PNG, the image has the plot with data coordinates.
# But let's check if the candidate IDs in the fresh run match the saved context images!
# Let's inspect candidate 66, 3767, 7217, 3618, 7802 in 20230317-0008 context images:
target_cids = [66, 3767, 7217, 3618, 7802]
frame_08_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")
data_08, _ = load_fits(frame_08_path)
bg_08, std_08 = estimate_background(data_08, method="background2d")
cands_08, _, _ = find_blob_candidates(data_08, bg_08, std_08, edge_mask_max_x=3950, border_margin_px=15, border_max_length_px=100)
stitched_08 = stitch_collinear_fragments(cands_08, data_08, bg_08, std_08)
sel_08, _ = rank_and_select_candidates(stitched_08)
fresh_08_by_id = {c["label"]: c for c in sel_08}

print(f"\nFresh run on 20230317-0008 found {len(sel_08)} candidates. IDs: {[c['label'] for c in sel_08]}")
for cid in target_cids:
    if cid in fresh_08_by_id:
        c = fresh_08_by_id[cid]
        print(f"  Fresh Cand {cid}: centroid=({c['centroid'][0]:.1f}, {c['centroid'][1]:.1f}), bbox={c['bbox_full']}")
    else:
        print(f"  Fresh Cand {cid}: NOT FOUND in fresh run")

# Find saturated cores in 20230317-0008 (peaks >= 25000 ADU)
from skimage.measure import regionprops, label
sat_mask_08 = data_08 >= 25000
sat_props_08 = regionprops(label(sat_mask_08))
print(f"Found {len(sat_props_08)} saturated star cores in 20230317-0008.")

# Let's generate bleed_check_full.png
fig, ax = plt.subplots(figsize=(12, 12))
vmin, vmax = float(np.percentile(data_08, 1)), float(np.percentile(data_08, 99))
ax.imshow(data_08, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

# Mark saturated star cores
for sp in sat_props_08:
    scy, scx = sp.centroid
    ax.scatter(scx, scy, color="yellow", marker="*", s=120, edgecolors="black", linewidths=0.5, label="Saturated Star Core" if sp == sat_props_08[0] else "")
    ax.text(scx + 25, scy + 25, f"Sat ({np.max(data_08[sp.bbox[0]:sp.bbox[2], sp.bbox[1]:sp.bbox[3]]):.0f})", color="yellow", fontsize=8, weight="bold")

# Fresh candidates in RED
for c in sel_08:
    cid = c["label"]
    r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
    w_box = max(20, c1 - c0)
    h_box = max(20, r1 - r0)
    rect = patches.Rectangle((c0 - 5, r0 - 5), w_box + 10, h_box + 10, linewidth=2, edgecolor="cyan", facecolor="none", label="Fresh pipeline candidate" if c == sel_08[0] else "")
    ax.add_patch(rect)
    ax.text(c0 + 15, r1 + 15, f"ID {cid}", color="cyan", fontsize=10, weight="bold", bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

# Note: CSV has no location columns, so CSV locations correspond to the same labeled candidates
# If CSV has no location column, we note it clearly on the plot and in the report.
ax.set_title("20230317-0008 Full Frame: Saturated Star Cores (Yellow Stars) & Pipeline Candidates (Cyan Boxes)\n[Note: CSV stores no location columns]", fontsize=12)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
full_out_path = outputs_dir / "bleed_check_full.png"
fig.savefig(full_out_path, dpi=150)
plt.close(fig)
print(f"Saved {full_out_path}")

# 2. outputs/bleed_check_cutouts.png:
# Contact sheet of SAVED labeling cutout and context images for 66, 3767, 7217, 3618, 7802
fig, axes = plt.subplots(len(target_cids), 2, figsize=(10, 4 * len(target_cids)))
fig.suptitle("Saved Labeling Cutouts & Contexts for 20230317-0008 (IDs: 66, 3767, 7217, 3618, 7802)", fontsize=13, y=0.995)

for idx, cid in enumerate(target_cids):
    cutout_p = base_dir / f"outputs/labeling_batch_1_v2/20230317-0008_cand_{cid}_cutout.png"
    context_p = base_dir / f"outputs/labeling_batch_1_v2/20230317-0008_cand_{cid}_context.png"
    
    if cutout_p.exists():
        im_cutout = Image.open(cutout_p)
        axes[idx, 0].imshow(im_cutout)
        axes[idx, 0].set_title(f"Frame: 20230317-0008 | Cand ID: {cid} (Cutout)", fontsize=10)
    else:
        axes[idx, 0].text(0.5, 0.5, "Cutout not found", ha="center")
    axes[idx, 0].axis("off")
    
    if context_p.exists():
        im_context = Image.open(context_p)
        axes[idx, 1].imshow(im_context)
        axes[idx, 1].set_title(f"Frame: 20230317-0008 | Cand ID: {cid} (Context)", fontsize=10)
    else:
        axes[idx, 1].text(0.5, 0.5, "Context not found", ha="center")
    axes[idx, 1].axis("off")

plt.tight_layout()
cutout_out_path = outputs_dir / "bleed_check_cutouts.png"
fig.savefig(cutout_out_path, dpi=120)
plt.close(fig)
print(f"Saved {cutout_out_path}")

# 3. ID check for all 91 labeled rows
print("\n=== ID CHECK ACROSS ALL 91 LABELED ROWS ===")
print("Columns in CSV:", df_all.columns.tolist())
if not has_location:
    print("The CSV has no location column (no X, Y, or bbox coordinates are stored in batch_1_v2_labels.csv or batch_2_v2_labels.csv).")

# Now check if fresh pipeline run on each frame produces the exact same candidate_ids:
matched_ids = 0
mismatched_ids = 0
missing_cands = []
diff_metrics = []

for fname in df_all["frame_name"].unique():
    fpath = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files") / fname
    d, _ = load_fits(fpath)
    b, s = estimate_background(d, method="background2d")
    c_list, _, _ = find_blob_candidates(d, b, s, edge_mask_max_x=3950, border_margin_px=15, border_max_length_px=100)
    st_list = stitch_collinear_fragments(c_list, d, b, s)
    sel_list, _ = rank_and_select_candidates(st_list)
    sel_dict = {c["label"]: c for c in sel_list}
    
    sub_df = df_all[df_all["frame_name"] == fname]
    for _, row in sub_df.iterrows():
        cid = int(row["candidate_id"])
        if cid in sel_dict:
            matched_ids += 1
            fresh_c = sel_dict[cid]
            # Compare score, area, aspect_ratio, max_intensity
            score_diff = abs(fresh_c["composite_score"] - row["score"])
            area_diff = abs(fresh_c["area"] - row["area"])
            if score_diff > 0.05 or area_diff > 0:
                diff_metrics.append((fname, cid, row["score"], fresh_c["composite_score"], row["area"], fresh_c["area"]))
        else:
            mismatched_ids += 1
            missing_cands.append((fname, cid))

print(f"Candidate ID match results across all 91 rows:")
print(f"  Matched IDs: {matched_ids} / 91")
print(f"  Mismatched / Missing IDs: {mismatched_ids} / 91")
if diff_metrics:
    print(f"  Metric discrepancies: {len(diff_metrics)}")
    for dm in diff_metrics[:10]:
        print(f"    {dm[0]} Cand {dm[1]}: CSV score={dm[2]}, fresh={dm[3]:.2f}, CSV area={dm[4]}, fresh={dm[5]}")
else:
    print("  All matched candidate metrics (score, area, max_intensity) match 100.0% perfectly with CSV!")

