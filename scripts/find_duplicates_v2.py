# TASK G9a: find duplicate candidates in candidates_v2.csv
"""
Find duplicate candidates in data/d0b/candidates_v2.csv:
- Tests all candidate pairs within each frame for geometric overlap:
  perpendicular distance <= 10 px AND along-line extent overlap > 50% of the shorter one.
- Groups duplicates transitively into clusters.
- Checks status of known candidate pairs 5768/3673 and 888/12982.
- Saves data/d0b/duplicate_pairs_v2.csv.
- Prints diagnostic summary (pair counts, clusters, label conflicts, line pairs, first 10 pairs).
"""

from collections import defaultdict, deque
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


def parse_endpoints(ep_str):
    if not ep_str or pd.isna(ep_str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def get_candidate_geom(row):
    is_line = (str(row.get("type", "")).strip().lower() == "line")
    c_r = float(row["centroid_row"])
    c_c = float(row["centroid_col"])
    length = float(row["length_px"]) if pd.notna(row.get("length_px")) and float(row["length_px"]) > 0 else 1.0

    endpoints = parse_endpoints(row.get("endpoints"))
    if is_line and endpoints is not None:
        p0, p1 = endpoints
        dr = p1[0] - p0[0]
        dc = p1[1] - p0[1]
        norm = math.hypot(dr, dc)
        if norm > 1e-4:
            ur = dr / norm
            uc = dc / norm
        else:
            ur, uc = 1.0, 0.0
        return {
            "is_line": True,
            "center": ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0),
            "length": norm,
            "ur": ur,
            "uc": uc,
            "angle_deg": float(row["angle_deg"]) if pd.notna(row.get("angle_deg")) else math.degrees(math.atan2(ur, uc)) % 180.0,
        }
    else:
        ang_deg = float(row["angle_deg"]) if pd.notna(row.get("angle_deg")) else 0.0
        return {
            "is_line": False,
            "center": (c_r, c_c),
            "length": length,
            "ur": None,
            "uc": None,
            "angle_deg": ang_deg,
        }


def check_pair_duplicate(geom_a, geom_b, perp_tol=10.0, overlap_thresh=0.50, max_ang_diff=15.0):
    # Ensure geom_a is the primary reference (line candidate if any, or longer candidate)
    if (not geom_a["is_line"] and geom_b["is_line"]) or (geom_a["is_line"] == geom_b["is_line"] and geom_a["length"] < geom_b["length"]):
        geom_ref, geom_other = geom_b, geom_a
    else:
        geom_ref, geom_other = geom_a, geom_b

    c_ref_r, c_ref_c = geom_ref["center"]
    c_oth_r, c_oth_c = geom_other["center"]
    dr = c_oth_r - c_ref_r
    dc = c_oth_c - c_ref_c

    len_ref = geom_ref["length"]
    len_oth = geom_other["length"]
    min_len = min(len_ref, len_oth)

    # Angle difference check
    ang_diff = abs(geom_ref["angle_deg"] - geom_other["angle_deg"])
    if ang_diff > 90.0:
        ang_diff = 180.0 - ang_diff
    if min_len >= 25.0 and ang_diff > max_ang_diff:
        return False, np.nan, 0.0

    if geom_ref["is_line"]:
        # Unambiguous direction from line endpoints
        ur = geom_ref["ur"]
        uc = geom_ref["uc"]
        d_perp = abs(dr * uc - dc * ur)
        t_oth = dr * ur + dc * uc

        # Overlap along line
        int_ref_0 = -len_ref / 2.0
        int_ref_1 = len_ref / 2.0
        int_oth_0 = t_oth - len_oth / 2.0
        int_oth_1 = t_oth + len_oth / 2.0

        overlap = max(0.0, min(int_ref_1, int_oth_1) - max(int_ref_0, int_oth_0))
        overlap_frac = overlap / max(min_len, 1e-4)

        is_dup = (d_perp <= perp_tol) and (overlap_frac > overlap_thresh)
        return is_dup, d_perp, overlap_frac
    else:
        # Single vs single: direction from angle_deg (angle from vertical)
        ang_rad = math.radians(geom_ref["angle_deg"])

        # Test both possible leans (+/- column lean)
        # Lean 1 (+): ur = cos(ang), uc = sin(ang)
        ur_p = math.cos(ang_rad)
        uc_p = math.sin(ang_rad)
        dp_p = abs(dr * uc_p - dc * ur_p)
        t_p = dr * ur_p + dc * uc_p
        ov_p = max(0.0, min(len_ref / 2.0, t_p + len_oth / 2.0) - max(-len_ref / 2.0, t_p - len_oth / 2.0))
        frac_p = ov_p / max(min_len, 1e-4)

        # Lean 2 (-): ur = cos(ang), uc = -sin(ang)
        ur_m = math.cos(ang_rad)
        uc_m = -math.sin(ang_rad)
        dp_m = abs(dr * uc_m - dc * ur_m)
        t_m = dr * ur_m + dc * uc_m
        ov_m = max(0.0, min(len_ref / 2.0, t_m + len_oth / 2.0) - max(-len_ref / 2.0, t_m - len_oth / 2.0))
        frac_m = ov_m / max(min_len, 1e-4)

        # Select the orientation that aligns with the pair displacement
        if dp_p <= dp_m:
            best_dp = dp_p
            best_frac = frac_p
        else:
            best_dp = dp_m
            best_frac = frac_m

        is_dup = (best_dp <= perp_tol) and (best_frac > overlap_thresh)
        return is_dup, best_dp, best_frac


def inspect_known_id(df_v2, target_id):
    # Check if present as a single candidate
    match_cand = df_v2[df_v2["candidate_id"] == float(target_id)]
    if not match_cand.empty:
        r = match_cand.iloc[0]
        return f"Present as single candidate (id={target_id}, cand_v2_id={r['cand_v2_id']}, frame={r['frame']}, verdict={r['verdict']})"

    # Check if absorbed into any line candidate
    lines = df_v2[df_v2["type"] == "line"]
    for _, l in lines.iterrows():
        fids = [int(x) for x in str(l.get("fragment_ids", "")).split(";") if x.strip()]
        if int(target_id) in fids:
            return f"Absorbed into accepted line {l['cand_v2_id']} (frame={l['frame']}, verdict={l['verdict']})"

    return f"Not present in candidates_v2.csv (id={target_id})"


def main():
    base_dir = Path(__file__).resolve().parent.parent

    # 1. Read candidates_v2.csv only
    input_path = base_dir / "data/d0b/candidates_v2.csv"
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    df_v2 = pd.read_csv(input_path)
    print(f"Loaded candidates_v2.csv: {len(df_v2)} total candidates across {df_v2['frame'].nunique()} frames.")

    # 2. Find duplicate pairs within each frame
    duplicate_records = []
    adj_graph = defaultdict(set)
    frame_grouped = df_v2.groupby("frame")

    for frame, group in frame_grouped:
        rows = group.to_dict("records")
        n_cands = len(rows)
        if n_cands < 2:
            continue

        geoms = [get_candidate_geom(r) for r in rows]

        for i in range(n_cands):
            row_a = rows[i]
            geom_a = geoms[i]
            cid_a = str(int(row_a["candidate_id"])) if pd.notna(row_a.get("candidate_id")) else str(row_a["cand_v2_id"])

            for j in range(i + 1, n_cands):
                row_b = rows[j]
                geom_b = geoms[j]
                cid_b = str(int(row_b["candidate_id"])) if pd.notna(row_b.get("candidate_id")) else str(row_b["cand_v2_id"])

                is_dup, perp_dist, overlap_frac = check_pair_duplicate(
                    geom_a, geom_b, perp_tol=10.0, overlap_thresh=0.50, max_ang_diff=15.0
                )

                if is_dup:
                    frame_name = str(row_a.get("frame_name", f"{frame}.fits"))
                    lbl_a = str(row_a.get("verdict", "unlabeled"))
                    lbl_b = str(row_b.get("verdict", "unlabeled"))

                    duplicate_records.append({
                        "frame_name": frame_name,
                        "candidate_id_a": cid_a,
                        "candidate_id_b": cid_b,
                        "perp_dist": round(float(perp_dist), 2),
                        "overlap_frac": round(float(overlap_frac), 3),
                        "label_a": lbl_a,
                        "label_b": lbl_b,
                        "cand_v2_id_a": row_a["cand_v2_id"],
                        "cand_v2_id_b": row_b["cand_v2_id"],
                        "type_a": row_a["type"],
                        "type_b": row_b["type"],
                    })

                    # Graph edge for clustering
                    adj_graph[row_a["cand_v2_id"]].add(row_b["cand_v2_id"])
                    adj_graph[row_b["cand_v2_id"]].add(row_a["cand_v2_id"])

    # 3. Transitive clustering
    visited = set()
    clusters = []
    for node in adj_graph:
        if node not in visited:
            comp = []
            queue = deque([node])
            visited.add(node)
            while queue:
                curr = queue.popleft()
                comp.append(curr)
                for neighbor in adj_graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(comp))

    clusters.sort(key=lambda c: len(c), reverse=True)

    # 4. Save duplicate pairs CSV
    out_csv_path = base_dir / "data/d0b/duplicate_pairs_v2.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    export_cols = ["frame_name", "candidate_id_a", "candidate_id_b", "perp_dist", "overlap_frac", "label_a", "label_b"]
    df_out = pd.DataFrame(duplicate_records)
    if not df_out.empty:
        df_out[export_cols].to_csv(out_csv_path, index=False)
    else:
        pd.DataFrame(columns=export_cols).to_csv(out_csv_path, index=False)
    print(f"Saved duplicate pairs to: {out_csv_path}")

    # 5. Diagnostic Evaluations & Printouts
    n_pairs = len(duplicate_records)
    n_clusters = len(clusters)

    # Disagreeing labels (streak vs fp)
    disagreeing_pairs = [
        r for r in duplicate_records
        if (r["label_a"] == "streak" and r["label_b"] == "fp") or (r["label_a"] == "fp" and r["label_b"] == "streak")
    ]

    # Pairs involving an accepted line
    line_pairs = [
        r for r in duplicate_records
        if r["type_a"] == "line" or r["type_b"] == "line"
    ]

    # Status of known pairs
    status_5768 = inspect_known_id(df_v2, 5768)
    status_3673 = inspect_known_id(df_v2, 3673)
    status_888 = inspect_known_id(df_v2, 888)
    status_12982 = inspect_known_id(df_v2, 12982)

    # Check if known pairs are flagged in duplicate pairs
    pair_5768_3673_flagged = any(
        ("5768" in str(r["candidate_id_a"]) and "3673" in str(r["candidate_id_b"])) or
        ("3673" in str(r["candidate_id_a"]) and "5768" in str(r["candidate_id_b"]))
        for r in duplicate_records
    )
    pair_888_12982_flagged = any(
        ("888" in str(r["candidate_id_a"]) and "12982" in str(r["candidate_id_b"])) or
        ("12982" in str(r["candidate_id_a"]) and "888" in str(r["candidate_id_b"]))
        for r in duplicate_records
    )

    print("\n" + "=" * 80)
    print("TASK G9a: DUPLICATE CANDIDATES DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"1. Total duplicate pairs found: {n_pairs}")
    print(f"2. Total candidate clusters:    {n_clusters}")
    if clusters:
        cluster_sizes = [len(c) for c in clusters]
        print(f"   Cluster sizes distribution:   max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")

    print("\n3. Pairs with conflicting labels (streak vs fp):")
    print("-" * 50)
    if disagreeing_pairs:
        print(f"   Count: {len(disagreeing_pairs)}")
        for dp in disagreeing_pairs:
            print(f"   {dp['frame_name']} | {dp['candidate_id_a']} ({dp['label_a']}) vs {dp['candidate_id_b']} ({dp['label_b']}) | d_perp={dp['perp_dist']}px, overlap={dp['overlap_frac']*100:.1f}%")
    else:
        print("   None (0 pairs)")

    print("\n4. Pairs involving an accepted line candidate:")
    print("-" * 50)
    if line_pairs:
        print(f"   Count: {len(line_pairs)}")
        for lp in line_pairs:
            print(f"   {lp['frame_name']} | {lp['candidate_id_a']} ({lp['type_a']}, {lp['label_a']}) vs {lp['candidate_id_b']} ({lp['type_b']}, {lp['label_b']}) | d_perp={lp['perp_dist']}px, overlap={lp['overlap_frac']*100:.1f}%")
    else:
        print("   None (0 pairs)")

    print("\n5. Status of Known Benchmark Duplicate Pairs:")
    print("-" * 50)
    print("   Pair 1: 5768 / 3673 (in 20230519-0009):")
    print(f"     - 5768: {status_5768}")
    print(f"     - 3673: {status_3673}")
    print(f"     - Flagged as duplicate pair in candidates_v2: {pair_5768_3673_flagged}")
    print()
    print("   Pair 2: 888 / 12982 (in 20231115-0003):")
    print(f"     - 888:   {status_888}")
    print(f"     - 12982: {status_12982}")
    print(f"     - Flagged as duplicate pair in candidates_v2: {pair_888_12982_flagged}")

    print("\n6. First 10 Duplicate Pairs:")
    print("-" * 80)
    if duplicate_records:
        header_fmt = "{:<18} {:<15} {:<15} {:<10} {:<12} {:<10} {:<10}"
        row_fmt = "{:<18} {:<15} {:<15} {:<10.2f} {:<12.3f} {:<10} {:<10}"
        print(header_fmt.format("frame_name", "cand_a", "cand_b", "perp_dist", "overlap_frac", "label_a", "label_b"))
        print("-" * 80)
        for r in duplicate_records[:10]:
            print(row_fmt.format(
                r["frame_name"][:18],
                str(r["candidate_id_a"])[:15],
                str(r["candidate_id_b"])[:15],
                r["perp_dist"],
                r["overlap_frac"],
                r["label_a"],
                r["label_b"],
            ))
    else:
        print("   No duplicate pairs found.")
    print("=" * 80)


if __name__ == "__main__":
    main()
