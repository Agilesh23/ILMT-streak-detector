"""
Diagnostic Script for Task C4d:
Investigate why 20230317-0008.fits exhibits high min_snr (33.57) compared to 20221025-0028.fits (7.86).

Extracts the top 10 highest-SNR collinear candidate pairs in 20230317-0008.fits,
reporting candidate labels, centroids, areas, max intensities, and ray sampling metrics.

Instructions for User:
Execute from PowerShell:
    python scripts/investigate_fp_frame_discrepancy.py
"""

from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d, map_coordinates

# Setup project path
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates


def analyze_pair_detailed(c1, c2, data, bg, std, snr_threshold=3.0, max_pair_distance=3500.0, angle_tolerance_deg=15.0):
    sub_img = data - bg
    std_img = std

    y1, x1 = c1["centroid"]
    y2, x2 = c2["centroid"]
    dy, dx = y2 - y1, x2 - x1
    dist = float(np.hypot(dy, dx))

    bbox1 = c1.get("bbox_full", c1.get("bbox_local", (0, 0, 1, 1)))
    bbox2 = c2.get("bbox_full", c2.get("bbox_local", (0, 0, 1, 1)))
    ang1 = float(np.degrees(np.arctan2(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])) % 180.0)
    ang2 = float(np.degrees(np.arctan2(bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])) % 180.0)
    
    vec_ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    
    d1 = abs(vec_ang - ang1) % 180.0
    diff1 = min(d1, 180.0 - d1)
    d2 = abs(vec_ang - ang2) % 180.0
    diff2 = min(d2, 180.0 - d2)

    angle_pass = (diff1 <= angle_tolerance_deg) and (diff2 <= angle_tolerance_deg)
    dist_pass = (dist <= max_pair_distance)

    if dist < 10.0 or not (dist_pass and angle_pass):
        return None

    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    kernel_length = 61
    kernel_width = 5

    n_pts = int(np.ceil(dist))
    t_steps = np.linspace(0, dist, n_pts)
    w_offsets = np.linspace(-kernel_width / 2.0, kernel_width / 2.0, kernel_width)

    sample_y = y1 + t_steps[:, None] * uy + w_offsets[None, :] * ny
    sample_x = x1 + t_steps[:, None] * ux + w_offsets[None, :] * nx

    coords = np.array([sample_y.ravel(), sample_x.ravel()])
    sampled_sub = map_coordinates(sub_img, coords, order=1, mode='constant', cval=0.0).reshape((n_pts, kernel_width))
    sampled_std = map_coordinates(std_img, coords, order=1, mode='nearest').reshape((n_pts, kernel_width))

    signal_1d = np.mean(sampled_sub, axis=1)
    std_1d = np.mean(sampled_std, axis=1)

    response_1d = uniform_filter1d(signal_1d, size=kernel_length, mode='nearest')
    gain_factor = np.sqrt(kernel_length * kernel_width)
    snr_profile = response_1d * gain_factor / np.maximum(std_1d, 1e-6)

    crop_start = int(0.10 * n_pts)
    crop_end = int(0.90 * n_pts)
    gap_snr = snr_profile[crop_start:crop_end] if crop_end > crop_start else snr_profile

    median_snr = float(np.median(gap_snr))
    min_snr = float(np.min(gap_snr))
    pct_above = float(np.mean(gap_snr >= snr_threshold) * 100.0)
    cov = float(np.std(gap_snr) / median_snr) if median_snr > 0 else 999.0

    return {
        "c1_label": c1.get("label"), "c2_label": c2.get("label"),
        "c1_centroid": (x1, y1), "c2_centroid": (x2, y2),
        "c1_area": c1.get("area"), "c2_area": c2.get("area"),
        "c1_max_i": c1.get("max_intensity"), "c2_max_i": c2.get("max_intensity"),
        "dist": dist, "diff1": diff1, "diff2": diff2,
        "pct_above": pct_above, "min_snr": min_snr, "cov": cov, "median_snr": median_snr
    }


def main():
    f_path = base_dir.parent / "ILMT_Contest_Materials" / "test_files" / "20230317-0008.fits"
    if not f_path.exists():
        f_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")

    print(f"=== TASK C4d DISCREPANCY INVESTIGATION FOR {f_path.name} ===")

    data, header = load_fits(f_path)
    bg, std = estimate_background(data, method="background2d")

    cands, _, _ = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0
    )

    print(f"Total Shape-Filtered Blobs Detected: {len(cands)}")

    # Check candidates labeled in batch 1 CSV
    labeled_ids = {66, 3984, 10204}
    found_labeled = [c for c in cands if c.get("label") in labeled_ids]
    print(f"Batch 1 Labeled FP Candidates Present in Blob List: {[c.get('label') for c in found_labeled]}")
    for c in found_labeled:
        cy, cx = c["centroid"]
        print(f"  Labeled ID {c['label']}: Centroid=(X={cx:.1f}, Y={cy:.1f}), Area={c['area']}, MaxI={c['max_intensity']:.1f}")

    pair_results = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            res = analyze_pair_detailed(cands[i], cands[j], data, bg, std)
            if res is not None:
                pair_results.append(res)

    pair_results.sort(key=lambda r: r["min_snr"], reverse=True)

    print(f"\nTested {len(pair_results)} collinear pairs (dist <= 3500px, angle <= 15°)")
    print("\n--- TOP 10 HIGHEST-SNR CANDIDATE PAIRS IN 20230317-0008.fits ---")

    for idx, p in enumerate(pair_results[:10]):
        c1_lab, c2_lab = p['c1_label'], p['c2_label']
        is_batch1 = (c1_lab in labeled_ids) or (c2_lab in labeled_ids)
        print(f"Rank {idx+1}: Pair (ID {c1_lab} <-> ID {c2_lab}) {'[BATCH 1 LABELED FP]' if is_batch1 else '[UNLABELED NOISE/STAR]'}")
        print(f"  Min Path SNR:       {p['min_snr']:.2f}  (Median SNR: {p['median_snr']:.2f}, CoV: {p['cov']:.2f})")
        print(f"  Centroid Distance:  {p['dist']:.1f} px")
        print(f"  Angle Deviations:   ({p['diff1']:.1f}°, {p['diff2']:.1f}°)")
        print(f"  Candidate 1 (ID {c1_lab}): Centroid=({p['c1_centroid'][0]:.1f}, {p['c1_centroid'][1]:.1f}), Area={p['c1_area']}, MaxI={p['c1_max_i']:.1f}")
        print(f"  Candidate 2 (ID {c2_lab}): Centroid=({p['c2_centroid'][0]:.1f}, {p['c2_centroid'][1]:.1f}), Area={p['c2_area']}, MaxI={p['c2_max_i']:.1f}\n")


if __name__ == "__main__":
    main()
