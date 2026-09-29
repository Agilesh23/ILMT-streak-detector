"""
Diagnostic Script for Task C4g:
Identify physical structure and candidate shape properties (eccentricity, aspect ratio, area, max intensity)
responsible for high-SNR candidate pairs in 20230317-0008.fits (X < 3950) and 20221026-0057.fits.

Instructions for User:
Execute from PowerShell:
    python scripts/diagnose_high_snr_offenders.py
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


def analyze_pair_detailed(c1, c2, data, bg, std, edge_mask_x=3950.0, snr_threshold=3.0, max_pair_distance=3500.0, angle_tolerance_deg=15.0):
    y1, x1 = c1["centroid"]
    y2, x2 = c2["centroid"]

    if edge_mask_x is not None and (x1 >= edge_mask_x or x2 >= edge_mask_x):
        return None

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

    if edge_mask_x is not None and np.any(sample_x >= edge_mask_x):
        return None

    sub_img = data - bg
    std_img = std

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
        "c1": c1, "c2": c2,
        "dist": dist, "diff1": diff1, "diff2": diff2,
        "pct_above": pct_above, "min_snr": min_snr, "cov": cov, "median_snr": median_snr
    }


def print_cand_details(label: str, c: dict):
    cy, cx = c["centroid"]
    area = c.get("area", 0)
    max_i = c.get("max_intensity", 0.0)
    ecc = c.get("eccentricity", 0.0)
    maj_len = c.get("axis_major_length", 0.0)
    min_len = c.get("axis_minor_length", 0.0)
    aspect_ratio = maj_len / max(min_len, 1e-5)

    print(f"    - {label} [ID {c.get('label')}]: Centroid=(X={cx:.1f}, Y={cy:.1f})")
    print(f"        Area:           {area} px²")
    print(f"        Max Intensity:  {max_i:.1f} ADU")
    print(f"        Major Length:   {maj_len:.1f} px | Minor Length: {min_len:.1f} px")
    print(f"        Aspect Ratio:   {aspect_ratio:.2f} (Threshold: >= 3.0)")
    print(f"        Eccentricity:   {ecc:.4f} (Threshold: >= 0.9000)")


def main():
    print("==================================================")
    print("TASK C4g DIAGNOSTIC: HIGH-SNR OFFENDING CANDIDATES")
    print("==================================================\n")

    # --- 1. FRAME 20230317-0008.fits (Masked X < 3950) ---
    f1_path = base_dir.parent / "ILMT_Contest_Materials" / "test_files" / "20230317-0008.fits"
    if not f1_path.exists():
        f1_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")

    print(f"--- FRAME 1: {f1_path.name} (X < 3950 Masked) ---")
    data1, _ = load_fits(f1_path)
    bg1, std1 = estimate_background(data1, method="background2d")
    cands1, _, _ = find_blob_candidates(data1, bg1, std1, n_sigma=3.0, min_aspect_ratio=3.0, min_eccentricity=0.9, max_area=15000.0)

    results1 = []
    for i in range(len(cands1)):
        for j in range(i + 1, len(cands1)):
            res = analyze_pair_detailed(cands1[i], cands1[j], data1, bg1, std1, edge_mask_x=3950.0)
            if res is not None:
                results1.append(res)

    results1.sort(key=lambda r: r["min_snr"], reverse=True)

    print(f"Found {len(results1)} valid inner collinear candidate pairs.")
    print("\nTOP 6 HIGHEST-SNR PAIRS IN 20230317-0008.fits (Inner Frame):\n")

    for idx, r in enumerate(results1[:6]):
        c1, c2 = r["c1"], r["c2"]
        print(f"Rank #{idx+1}: Pair (ID {c1['label']} <-> ID {c2['label']}) | Measured Path Min SNR: {r['min_snr']:.2f}")
        print(f"  Centroid Dist: {r['dist']:.1f} px | Angle Diffs: ({r['diff1']:.1f}°, {r['diff2']:.1f}°) | CoV: {r['cov']:.2f}")
        print_cand_details("Candidate 1", c1)
        print_cand_details("Candidate 2", c2)
        print()

    # --- 2. FRAME 20221026-0057.fits ---
    f2_path = base_dir.parent / "ILMT_Contest_Materials" / "test_files" / "20221026-0057.fits"
    if not f2_path.exists():
        f2_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221026-0057.fits")

    print("--------------------------------------------------")
    print(f"--- FRAME 2: {f2_path.name} ---")
    data2, _ = load_fits(f2_path)
    bg2, std2 = estimate_background(data2, method="background2d")
    cands2, _, _ = find_blob_candidates(data2, bg2, std2, n_sigma=3.0, min_aspect_ratio=3.0, min_eccentricity=0.9, max_area=15000.0)

    results2 = []
    for i in range(len(cands2)):
        for j in range(i + 1, len(cands2)):
            res = analyze_pair_detailed(cands2[i], cands2[j], data2, bg2, std2, edge_mask_x=None)
            if res is not None:
                results2.append(res)

    results2.sort(key=lambda r: r["min_snr"], reverse=True)

    print(f"Found {len(results2)} valid collinear candidate pairs in 20221026-0057.fits.")
    if results2:
        top2 = results2[0]
        c1, c2 = top2["c1"], top2["c2"]
        print(f"\nSingle Violating Pair (ID {c1['label']} <-> ID {c2['label']}) | Measured Path Min SNR: {top2['min_snr']:.2f}")
        print(f"  Centroid Dist: {top2['dist']:.1f} px | Angle Diffs: ({top2['diff1']:.1f}°, {top2['diff2']:.1f}°) | CoV: {top2['cov']:.2f}")
        print_cand_details("Candidate 1", c1)
        print_cand_details("Candidate 2", c2)
        print()


if __name__ == "__main__":
    main()
