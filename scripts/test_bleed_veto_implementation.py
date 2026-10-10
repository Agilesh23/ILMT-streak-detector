import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from skimage.measure import regionprops, label
from scipy.ndimage import generate_binary_structure, binary_dilation

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background

# Load labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(base_dir / "data/labeled/batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)

streak_rows = df_all[df_all["label"] == "streak"]
print(f"Total labeled streak rows: {len(streak_rows)} (expected 15)")

frames = [
    "20221025-0028.fits",
    "20230519-0009.fits",
    "20230317-0008.fits",
    "20221026-0057.fits",
    "20221101-0030.fits",
    "20230602-0022.fits",
    "20231115-0003.fits",
    "20240114-0028.fits",
    "20240405-0004.fits"
]

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg["detection"]

def check_pca_bleed(coords, data, frame_max, max_ang_deg=10.0, sat_frac=0.9):
    if len(coords) < 2:
        return False, 0.0, 0.0
    pts_x = coords[:, 1].astype(np.float64)
    pts_y = coords[:, 0].astype(np.float64)
    dx = pts_x - np.mean(pts_x)
    dy = pts_y - np.mean(pts_y)
    cov = np.cov(np.vstack([dx, dy]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigvals)[::-1]
    vec = eigvecs[:, idx_sort[0]]
    if vec[0] < 0:
        vec = -vec
    ang_h = float(np.degrees(np.arctan2(vec[1], vec[0])))
    if ang_h > 90.0: ang_h -= 180.0
    elif ang_h < -90.0: ang_h += 180.0
    ang_v = float(abs(90.0 - abs(ang_h)))
    
    blob_pixels = data[coords[:, 0], coords[:, 1]]
    max_val = float(np.max(blob_pixels))
    has_sat = max_val >= sat_frac * frame_max
    
    is_bleed = (ang_v <= max_ang_deg) and has_sat
    return is_bleed, ang_v, max_val

for fn in frames:
    d, _ = load_fits(frame_dir / fn)
    bg, std = estimate_background(d, method="background2d")
    sub = d - bg
    thresh_map = 3.0 * std
    binary = sub > thresh_map
    struct = generate_binary_structure(2, 1)
    binary = binary_dilation(binary, structure=struct, iterations=2)
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=sub)
    frame_max = float(np.max(d))
    
    flagged = []
    for p in props:
        is_bleed, ang_v, max_val = check_pca_bleed(p.coords, d, frame_max)
        if is_bleed:
            flagged.append((p.label, ang_v, max_val))
            
    # Check which flagged candidates are in df_all
    df_f = df_all[df_all["frame_name"] == fn]
    for cid, ang_v, max_val in flagged:
        m = df_f[df_f["candidate_id"] == cid]
        lbl = m.iloc[0]["label"] if len(m) > 0 else "unlabeled"
        notes = m.iloc[0]["notes"] if len(m) > 0 else ""
        print(f"[{fn}] Bleed candidate {cid}: ang_v={ang_v:.2f} deg, max={max_val:.1f} (thresh={0.9*frame_max:.1f}), label={lbl}, notes={notes}")
