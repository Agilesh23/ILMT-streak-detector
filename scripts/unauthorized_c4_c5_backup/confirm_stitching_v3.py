import sys
from pathlib import Path

sys.path.insert(0, "src")

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

frame_path = r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230519-0009.fits"

# 1. Load and process
data, header = load_fits(frame_path)
bg, std = estimate_background(data, method='background2d')

candidates, count_before, count_shape = find_blob_candidates(
    data, bg, std,
    n_sigma=3.0,
    min_aspect_ratio=3.0,
    min_eccentricity=0.9,
    max_area=15000.0
)

stitched = stitch_collinear_fragments(candidates, data, bg, std)
selected, cutoff_score = rank_and_select_candidates(stitched)

# 2. Find candidate with largest axis_major_length in stitched set
largest_cand = max(stitched, key=lambda c: c.get("axis_major_length", 0.0))
largest_length = largest_cand.get("axis_major_length", 0.0)
largest_score = largest_cand.get("composite_score", 0.0)
largest_label = largest_cand.get("label")
stitched_from = largest_cand.get("stitched_from", [largest_label])
n_fragments = largest_cand.get("n_fragments", 1)

# Check if largest candidate is in final selected set
is_selected = any(c.get("label") == largest_label for c in selected)

# 3. Lowest max_intensity among selected candidates
min_selected_intensity = min(c.get("max_intensity", 0.0) for c in selected) if selected else 0.0

print("=== STITCHING & CANDIDATE CONFIRMATION (20230519-0009.fits) ===")
print(f"Total candidates after shape filter: {count_shape}")
print(f"Total candidates after stitching:    {len(stitched)}")
print(f"Total candidates after ranking:      {len(selected)} (Cutoff score: {cutoff_score:.2f})")
print()
print(f"Largest Stitched Candidate Length:   {largest_length:.2f} px")
print(f"Number of Fragments Merged:         {n_fragments} (Labels: {stitched_from})")
print(f"Largest Candidate Composite Score:   {largest_score:.2f}")
print(f"Included in Final Selected Set?:    {is_selected}")
print(f"Lowest max_intensity in Selected:    {min_selected_intensity:.2f} ADU")
print()
print("Selected Candidates Detail:")
for i, c in enumerate(selected):
    print(f"  [{i+1}] Label={c.get('label')} | Length={c.get('axis_major_length', 0):.1f}px | Aspect={c.get('aspect_ratio', 0):.1f} | MaxIntensity={c.get('max_intensity', 0):.2f} | Score={c.get('composite_score', 0):.2f}")
