import sys
from pathlib import Path
import yaml

sys.path.insert(0, "src")

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

# Load config
with open("src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

edge_mask_max_x = cfg.get("detection", {}).get("edge_mask_max_x", None)
border_margin_px = cfg.get("detection", {}).get("border_margin_px", None)
border_max_length_px = cfg.get("detection", {}).get("border_max_length_px", None)
left_edge_max_col = cfg.get("detection", {}).get("left_edge_max_col", None)
left_edge_max_angle_from_vertical_deg = cfg.get("detection", {}).get("left_edge_max_angle_from_vertical_deg", None)

frames = [
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221025-0028.fits",
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230519-0009.fits",
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits",
]

print(f"{'Frame':<20} | {'Shape Filter':<12} | {'Stitched':<10} | {'Ranked/Selected':<15}")
print("-" * 65)

for path in frames:
    fname = Path(path).name
    data, header = load_fits(path)
    bg, std = estimate_background(data, method='background2d')
    candidates, count_before, count_shape = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg
    )
    stitched = stitch_collinear_fragments(candidates, data, bg, std)
    selected, cutoff = rank_and_select_candidates(stitched)
    print(f"{fname:<20} | {count_shape:<12} | {len(stitched):<10} | {len(selected):<15}")


