# TASK G9d: audit streak-labeled singles
"""
Audit all single candidates labeled as 'streak' in data/d0b/candidates_v2.csv.
For each streak-labeled single:
- frame_name, candidate_id
- distance in px from its centroid to the nearest accepted line in the same frame (blank if none)
- length_px
- number of other candidates (any label) whose centroid lies within 150 px of it

Sorted by length_px descending. Prints total count and top 15 rows.
Saves to data/d0b/audit_streak_singles.csv.
"""

import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent


def parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' into ((r0, c0), (r1, c1))."""
    if not ep_str or pd.isna(ep_str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def point_to_segment_distance(p_r, p_c, p0_r, p0_c, p1_r, p1_c):
    """Calculate minimum Euclidean distance from point (p_r, p_c) to line segment (p0, p1)."""
    dr = p1_r - p0_r
    dc = p1_c - p0_c
    seg_len_sq = dr * dr + dc * dc
    if seg_len_sq < 1e-8:
        return math.hypot(p_r - p0_r, p_c - p0_c)

    # Clamped projection parameter t in [0, 1]
    t = ((p_r - p0_r) * dr + (p_c - p0_c) * dc) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_r = p0_r + t * dr
    proj_c = p0_c + t * dc
    return math.hypot(p_r - proj_r, p_c - proj_c)


def main():
    print("=" * 85)
    print("TASK G9d: AUDIT STREAK-LABELED SINGLES IN CANDIDATES_V2.CSV")
    print("=" * 85)

    input_path = base_dir / "data/d0b/candidates_v2.csv"
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded candidates_v2.csv: {len(df)} total candidates across {df['frame_name'].nunique()} frames.\n")

    # Group lines by frame for fast point-to-line distance lookup
    lines_by_frame = {}
    lines_df = df[df["type"].astype(str).str.lower() == "line"]
    for fname, group in lines_df.groupby("frame_name"):
        line_segments = []
        for _, lrow in group.iterrows():
            eps = parse_endpoints(lrow.get("endpoints"))
            if eps is not None:
                line_segments.append(eps)
        lines_by_frame[fname] = line_segments

    # Group all candidate centroids by frame for neighbor density within 150 px
    all_cands_by_frame = {}
    for fname, group in df.groupby("frame_name"):
        coords = []
        for idx, row in group.iterrows():
            cy = float(row["centroid_row"])
            cx = float(row["centroid_col"])
            coords.append((idx, cy, cx))
        all_cands_by_frame[fname] = coords

    # Filter: single candidates labeled as streak
    mask_streak_singles = (
        (df["type"].astype(str).str.lower() != "line") &
        (df["verdict"].astype(str).str.lower() == "streak")
    )
    df_streak_singles = df[mask_streak_singles].copy()
    total_count = len(df_streak_singles)

    records = []
    for orig_idx, row in df_streak_singles.iterrows():
        fname = str(row["frame_name"])
        cid = int(row["candidate_id"]) if pd.notna(row.get("candidate_id")) else row["cand_v2_id"]
        cy = float(row["centroid_row"])
        cx = float(row["centroid_col"])
        length_px = float(row["length_px"]) if pd.notna(row.get("length_px")) else 0.0

        # 1. Distance in px from centroid to nearest accepted line in the same frame
        line_segs = lines_by_frame.get(fname, [])
        if line_segs:
            min_dist = min(
                point_to_segment_distance(cy, cx, ep0[0], ep0[1], ep1[0], ep1[1])
                for (ep0, ep1) in line_segs
            )
            dist_str = f"{min_dist:.1f}"
            dist_val = round(min_dist, 1)
        else:
            dist_str = ""
            dist_val = np.nan

        # 2. Number of other candidates (any label) whose centroid lies within 150 px
        frame_all = all_cands_by_frame.get(fname, [])
        n_within_150 = sum(
            1 for (oth_idx, oth_cy, oth_cx) in frame_all
            if oth_idx != orig_idx and math.hypot(cy - oth_cy, cx - oth_cx) <= 150.0
        )

        records.append({
            "frame_name": fname,
            "candidate_id": cid,
            "dist_to_line_px": dist_str,
            "dist_val": dist_val,
            "length_px": round(length_px, 1),
            "n_within_150px": n_within_150,
        })

    # Sort by length_px descending
    records.sort(key=lambda r: r["length_px"], reverse=True)

    # Save CSV
    out_dir = base_dir / "data/d0b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "audit_streak_singles.csv"

    export_cols = ["frame_name", "candidate_id", "dist_to_line_px", "length_px", "n_within_150px"]
    df_out = pd.DataFrame(records)[export_cols]
    df_out.to_csv(out_csv, index=False)

    print(f"Total single candidates labeled as streak: {total_count}")
    print(f"Saved audit results to: {out_csv}\n")

    # Print Top 15
    print("TOP 15 STREAK-LABELED SINGLES (SORTED BY LENGTH_PX DESCENDING):")
    print("-" * 85)
    header_fmt = "{:<4} | {:<18} | {:<12} | {:<10} | {:<16} | {:<14}"
    row_fmt    = "{:<4} | {:<18} | {:<12} | {:<10.1f} | {:<16} | {:<14}"
    print(header_fmt.format("Rank", "Frame Name", "Candidate ID", "Length (px)", "Dist to Line(px)", "N within 150px"))
    print("-" * 85)

    for rank, r in enumerate(records[:15], 1):
        dist_display = r["dist_to_line_px"] if r["dist_to_line_px"] != "" else "none"
        print(row_fmt.format(
            rank,
            r["frame_name"][:18],
            str(r["candidate_id"]),
            r["length_px"],
            dist_display,
            r["n_within_150px"]
        ))

    print("-" * 85)
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()
