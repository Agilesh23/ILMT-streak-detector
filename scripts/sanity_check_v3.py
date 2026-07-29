import sys
from pathlib import Path

sys.path.insert(0, "src")

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

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
        max_area=15000.0
    )
    stitched = stitch_collinear_fragments(candidates, data, bg, std)
    selected, cutoff = rank_and_select_candidates(stitched)
    print(f"{fname:<20} | {count_shape:<12} | {len(stitched):<10} | {len(selected):<15}")
