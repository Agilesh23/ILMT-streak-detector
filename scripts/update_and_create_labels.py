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
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Inspect current batch 1 and 2
df1 = pd.read_csv(data_dir / "batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "batch_2_v2_labels.csv")

print(f"Batch 1 shape: {df1.shape}, columns: {list(df1.columns)}")
print(f"Batch 2 shape: {df2.shape}, columns: {list(df2.columns)}")

# Check candidates mentioned in batch 1 & 2
items_to_check = [
    ("20221025-0028.fits", 16881),
    ("20221025-0028.fits", 14253),
    ("20230519-0009.fits", 3673),
    ("20230317-0008.fits", 3618),
    ("20221026-0057.fits", 1175),
    ("20240405-0004.fits", 14613),
    ("20240405-0004.fits", 12522),
    ("20240405-0004.fits", 17987),
    ("20240405-0004.fits", 9558),
    ("20240405-0004.fits", 10716),
    ("20221026-0057.fits", 770),
    ("20231115-0003.fits", 15632),
]

for fn, cid in items_to_check:
    in_b1 = df1[(df1["frame_name"] == fn) & (df1["candidate_id"] == cid)]
    in_b2 = df2[(df2["frame_name"] == fn) & (df2["candidate_id"] == cid)]
    print(f"{fn} {cid}: in_b1={len(in_b1)}, in_b2={len(in_b2)}")
