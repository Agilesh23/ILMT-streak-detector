import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
data_dir = base_dir / "data/labeled"

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, compute_composite_score

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg["detection"]
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
bypass_cfg = det_cfg.get("long_thin_bypass", None)

# -------------------------------------------------------------
# STEP 1: Update batch 1 and batch 2
# -------------------------------------------------------------
df1 = pd.read_csv(data_dir / "batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "batch_2_v2_labels.csv")

# Fill group_id with empty string initially if not present
for df in [df1, df2]:
    if "group_id" not in df.columns:
        # Insert group_id right after 'notes'
        notes_idx = df.columns.get_loc("notes")
        df.insert(notes_idx + 1, "group_id", "")
    else:
        df["group_id"] = df["group_id"].fillna("")

# Update Batch 1 & 2 rows
# Helper to set (label, notes, group_id)
def update_row(df, frame_name, cid, label, notes, group_id):
    mask = (df["frame_name"] == frame_name) & (df["candidate_id"] == cid)
    if not mask.any():
        print(f"WARNING: row not found: {frame_name}, {cid}")
        return
    idx = df[mask].index[0]
    if label is not None:
        df.at[idx, "label"] = label
    if notes is not None:
        df.at[idx, "notes"] = notes
    if group_id is not None:
        df.at[idx, "group_id"] = group_id

# - 16881, 14253 (20221025-0028): label fp, notes not_streak, group_id blank
update_row(df1, "20221025-0028.fits", 16881, "fp", "not_streak", "")
update_row(df1, "20221025-0028.fits", 14253, "fp", "not_streak", "")

# - 3673 (20230519-0009): streak, long_streak, g_20230519
update_row(df1, "20230519-0009.fits", 3673, "streak", "long_streak", "g_20230519")

# - 3618 (20230317-0008): streak, fragment, g_20230317
update_row(df1, "20230317-0008.fits", 3618, "streak", "fragment", "g_20230317")

# - 1175 (20221026-0057): streak, fragment, g_20221026
update_row(df2, "20221026-0057.fits", 1175, "streak", "fragment", "g_20221026")

# - 14613, 12522, 17987, 9558, 10716 (20240405-0004): streak, fragment, g_20240405
for cid in [14613, 12522, 17987, 9558, 10716]:
    update_row(df2, "20240405-0004.fits", cid, "streak", "fragment", "g_20240405")

# - 770 (20221026-0057): fp, saturated_star_artifact
update_row(df2, "20221026-0057.fits", 770, "fp", "saturated_star_artifact", "")

# - 15632 (20231115-0003): fp, small_mark
update_row(df2, "20231115-0003.fits", 15632, "fp", "small_mark", "")

# - all other fp rows: group_id blank
for df in [df1, df2]:
    fp_mask = (df["label"] == "fp")
    df.loc[fp_mask, "group_id"] = ""

# Save updated batch 1 and 2
df1.to_csv(data_dir / "batch_1_v2_labels.csv", index=False)
df2.to_csv(data_dir / "batch_2_v2_labels.csv", index=False)

# -------------------------------------------------------------
# STEP 2: Create batch 3
# -------------------------------------------------------------
batch3_spec = [
    ("20221025-0028.fits", 6527, "streak", "whole_streak", "g_20221025", (1038, 0, 2957, 4096)),
    ("20230519-0009.fits", 5768, "streak", "part_of_3673", "g_20230519", (1271, 784, 3443, 3779)),
    ("20221026-0057.fits", 1183, "streak", "same_streak_as_1175", "g_20221026", (347, 1110, 427, 3328)),
    ("20221026-0057.fits", 1714, "fp", "bleed_trail", "", (607, 540, 2938, 599)),
    ("20230602-0022.fits", 11716, "streak", "whole_streak", "g_20230602", (2224, 2042, 4032, 4096)),
    ("20231115-0003.fits", 888, "streak", "collinear_frag", "g_20231115", (153, 0, 2329, 2463)),
    ("20231115-0003.fits", 12982, "streak", "collinear_frag", "g_20231115", (2334, 2473, 3769, 4096)),
    ("20240114-0028.fits", 5446, "streak", "whole_streak", "g_20240114", (966, 0, 2690, 4096)),
]

unique_b3_frames = list(dict.fromkeys([s[0] for s in batch3_spec]))
cands_by_frame = {}

for fname in unique_b3_frames:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    cands, _, _ = find_blob_candidates(
        d, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass=bypass_cfg
    )
    for c in cands:
        c["composite_score"] = compute_composite_score(c)
    cands_by_frame[fname] = {c["label"]: c for c in cands}

b3_rows = []
for fname, cid, label_val, notes_val, group_val, exp_bbox in batch3_spec:
    c = cands_by_frame[fname].get(cid)
    if c is None:
        raise ValueError(f"Candidate {cid} not found in frame {fname}")
    
    r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
    diffs = [abs(r0 - exp_bbox[0]), abs(c0 - exp_bbox[1]), abs(r1 - exp_bbox[2]), abs(c1 - exp_bbox[3])]
    max_diff = max(diffs)
    if max_diff > 5:
        print(f"DIFF > 5 px: {fname} {cid}: found [{r0}:{r1}, {c0}:{c1}], expected [{exp_bbox[0]}:{exp_bbox[2]}, {exp_bbox[1]}:{exp_bbox[3]}]")
    
    cy, cx = c["centroid"]
    dy = r1 - r0
    dx = c1 - c0
    ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    length = float(c.get("axis_major_length", np.hypot(dy, dx)))
    
    row_dict = {
        "candidate_id": int(cid),
        "frame_name": fname,
        "cutout_path": "",
        "context_path": "",
        "score": round(float(c["composite_score"]), 2),
        "area": int(c["area"]),
        "aspect_ratio": round(float(c["aspect_ratio"]), 2),
        "max_intensity": round(float(c["max_intensity"]), 2),
        "label": label_val,
        "notes": notes_val,
        "group_id": group_val,
        "centroid_row": round(float(cy), 2),
        "centroid_col": round(float(cx), 2),
        "bbox_r0": float(r0),
        "bbox_c0": float(c0),
        "bbox_r1": float(r1),
        "bbox_c1": float(c1),
        "length_px": round(length, 2),
        "angle_deg": round(ang, 2)
    }
    b3_rows.append(row_dict)

df3 = pd.DataFrame(b3_rows, columns=df1.columns)
df3.to_csv(data_dir / "batch_3_v2_labels.csv", index=False)

# -------------------------------------------------------------
# STEP 3: Print counts per CSV by label and group_id
# -------------------------------------------------------------
print("=== COUNTS PER CSV ===")
all_group_ids = set()

for name, path in [("Batch 1", data_dir / "batch_1_v2_labels.csv"),
                   ("Batch 2", data_dir / "batch_2_v2_labels.csv"),
                   ("Batch 3", data_dir / "batch_3_v2_labels.csv")]:
    df = pd.read_csv(path)
    df["group_id"] = df["group_id"].fillna("")
    n_streak = (df["label"] == "streak").sum()
    n_fp = (df["label"] == "fp").sum()
    g_ids = set(df[df["group_id"] != ""]["group_id"].unique())
    all_group_ids.update(g_ids)
    print(f"{name}: {n_streak} streak / {n_fp} fp (total {len(df)}) | group_ids: {sorted(list(g_ids))}")

print(f"\nTotal distinct group_ids across all batches: {len(all_group_ids)}")
print(f"Distinct group_ids: {sorted(list(all_group_ids))}")
