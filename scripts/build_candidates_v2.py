# TASK D0c-fix1: fix the G1 file glob in build_candidates_v2.py
"""
Build candidates_v2 pipeline:
1. Line candidates: every ACCEPTED line in data/d0b/all_lines_v2.csv becomes one candidate:
   cand_v2_id = '<frame>_L<line_id>', type='line', fragment_ids, n_fragments,
   length_px = t_end - t_start, angle_deg, z_peak, n_segments_z_gt_3, endpoints,
   centroid = midpoint of endpoints.
2. Single candidates: every candidate in data/g1/frame_20*.csv that is NOT absorbed by an accepted line.
   Absorbed if candidate_id is in line fragment_ids OR within 12 px of line and inside its extent.
   type='single', cand_v2_id = '<frame>_C<candidate_id>'.
3. Verdicts:
   - Line: streak if any member is streak; fp if all labeled members are fp; else unlabeled.
     Visual overrides for lines of 20230615-0006 and 20240502-0026 (line_id 1 each).
   - Single: verdict from verdicts.csv if present, else unlabeled.
Outputs data/d0b/candidates_v2.csv and prints comprehensive comparative statistics.
"""

from collections import Counter, defaultdict
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent.parent

    # 1. Load accepted lines from data/d0b/all_lines_v2.csv
    lines_path = base_dir / "data/d0b/all_lines_v2.csv"
    if not lines_path.exists():
        print(f"Error: {lines_path} not found.")
        sys.exit(1)

    df_lines = pd.read_csv(lines_path)
    df_acc = df_lines[df_lines["accepted"].astype(str).str.lower() == "true"].copy()

    lines_by_frame = defaultdict(list)
    for _, row_l in df_acc.iterrows():
        f_stem = str(row_l["frame"]).strip().lower()
        lines_by_frame[f_stem].append(row_l)

    # 2. Load verdicts from data/g1/verdicts.csv
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    verdict_map = {}
    if verdicts_path.exists():
        df_v = pd.read_csv(verdicts_path)
        col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
        col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
        col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]

        for _, r in df_v.iterrows():
            f_name = Path(str(r[col_frame])).stem.lower()
            cid = int(r[col_cid])
            v = str(r[col_verdict]).strip().lower()
            verdict_map[(f_name, cid)] = v

    # 3. Locate all 86 G1 CSVs
    g1_files = sorted(list((base_dir / "data/g1").glob("frame_20*.csv")))
    if not g1_files:
        print("Error: No data/g1/frame_20*.csv files found.")
        sys.exit(1)

    # Statistics tracking
    # Old G1 stats
    old_total_cands = 0
    old_labeled_streak = 0
    old_labeled_fp = 0
    old_labeled_review = 0
    old_p_ge_05 = 0
    old_streak_frames = set()

    # Absorption counters
    absorbed_by_id_count = 0
    absorbed_by_geom_only_count = 0
    absorbed_total_count = 0

    absorbed_id_verdicts = Counter()
    absorbed_geom_only_verdicts = Counter()
    absorbed_total_verdicts = Counter()

    v2_records = []
    lines_with_conflict = []
    frames_with_accepted_line = set()

    for g1_file in g1_files:
        stem = g1_file.stem.replace("frame_", "").lower()
        df_g1 = pd.read_csv(g1_file)

        # Accrue Old (G1) stats for this frame
        for _, g1_cand in df_g1.iterrows():
            old_total_cands += 1
            cid = int(g1_cand["candidate_id"])
            p_val = float(g1_cand["p_streak"]) if "p_streak" in g1_cand and pd.notna(g1_cand["p_streak"]) else 0.0
            if p_val >= 0.5:
                old_p_ge_05 += 1

            v = verdict_map.get((stem, cid))
            if v == "streak":
                old_labeled_streak += 1
                old_labeled_review += 1
                old_streak_frames.add(stem)
            elif v == "fp":
                old_labeled_fp += 1
                old_labeled_review += 1

        frame_acc_lines = lines_by_frame.get(stem, [])
        if frame_acc_lines:
            frames_with_accepted_line.add(stem)

        # Step 1: Create line candidates for this frame
        for row_l in frame_acc_lines:
            line_id = int(row_l["line_id"])
            cand_v2_id = f"{stem}_L{line_id}"
            frag_ids_str = str(row_l["fragment_ids"]) if pd.notna(row_l["fragment_ids"]) else ""
            frag_ids_list = [int(x) for x in frag_ids_str.split(";") if x.strip()]
            n_fragments = int(row_l["n_fragments"]) if pd.notna(row_l["n_fragments"]) else len(frag_ids_list)

            t_start = float(row_l["t_start"])
            t_end = float(row_l["t_end"])
            length_px = round(t_end - t_start, 1)
            angle_deg = round(float(row_l["angle_deg"]), 2)
            z_peak = round(float(row_l["z_peak"]), 2)
            n_segments_z_gt_3 = int(row_l["n_segments_z_gt_3"])

            ep0_r = float(row_l["endpoint_0_r"])
            ep0_c = float(row_l["endpoint_0_c"])
            ep1_r = float(row_l["endpoint_1_r"])
            ep1_c = float(row_l["endpoint_1_c"])
            endpoints_str = f"({ep0_r:.1f}, {ep0_c:.1f}); ({ep1_r:.1f}, {ep1_c:.1f})"

            mid_r = round((ep0_r + ep1_r) / 2.0, 2)
            mid_c = round((ep0_c + ep1_c) / 2.0, 2)
            centroid_str = f"({mid_r:.2f}, {mid_c:.2f})"

            # Member verdict evaluations
            n_streak_m = 0
            n_fp_m = 0
            n_unlabeled_m = 0
            for m in frag_ids_list:
                vm = verdict_map.get((stem, m))
                if vm == "streak":
                    n_streak_m += 1
                elif vm == "fp":
                    n_fp_m += 1
                else:
                    n_unlabeled_m += 1

            conflict = bool(n_streak_m > 0 and n_fp_m > 0)
            if conflict:
                lines_with_conflict.append((stem, line_id, n_streak_m, n_fp_m, frag_ids_str))

            if n_streak_m > 0:
                line_verdict = "streak"
                verdict_source = "members"
            elif n_fp_m > 0:
                line_verdict = "fp"
                verdict_source = "members"
            else:
                line_verdict = "unlabeled"
                verdict_source = "none"

            # Visual overrides
            if (stem == "20230615-0006" and line_id == 1) or (stem == "20240502-0026" and line_id == 1):
                line_verdict = "streak"
                verdict_source = "visual_d0b"

            line_rec = {
                "cand_v2_id": cand_v2_id,
                "type": "line",
                "frame": stem,
                "frame_name": f"{stem}.fits",
                "line_id": line_id,
                "candidate_id": np.nan,
                "verdict": line_verdict,
                "verdict_source": verdict_source,
                "conflict": conflict,
                "n_streak": n_streak_m,
                "n_fp": n_fp_m,
                "n_unlabeled": n_unlabeled_m,
                "fragment_ids": frag_ids_str,
                "n_fragments": n_fragments,
                "length_px": length_px,
                "angle_deg": angle_deg,
                "z_peak": z_peak,
                "n_segments_z_gt_3": n_segments_z_gt_3,
                "endpoints": endpoints_str,
                "centroid": centroid_str,
                "centroid_row": mid_r,
                "centroid_col": mid_c,
                "bbox_r0": np.nan,
                "bbox_c0": np.nan,
                "bbox_r1": np.nan,
                "bbox_c1": np.nan,
                "width_ratio": np.nan,
                "snr_mean": np.nan,
                "sat_frac": np.nan,
                "straightness": np.nan,
                "angle_from_vertical": np.nan,
                "aspect": np.nan,
                "edge_dist": np.nan,
                "p_streak": np.nan,
            }
            v2_records.append(line_rec)

        # Step 2: Check absorption of G1 candidates
        for _, cand_row in df_g1.iterrows():
            cid = int(cand_row["candidate_id"])
            cy = float(cand_row["centroid_row"])
            cx = float(cand_row["centroid_col"])
            v_cand = verdict_map.get((stem, cid), "unlabeled")

            absorbed_by_id = False
            absorbed_by_geometry = False

            for row_l in frame_acc_lines:
                frag_ids = set([int(x) for x in str(row_l["fragment_ids"]).split(";") if x.strip()])
                if cid in frag_ids:
                    absorbed_by_id = True

                c_y = float(row_l["centroid_y"])
                c_x = float(row_l["centroid_x"])
                u_y = float(row_l["uy"])
                u_x = float(row_l["ux"])
                t_s = float(row_l["t_start"])
                t_e = float(row_l["t_end"])

                t_c = (cy - c_y) * u_y + (cx - c_x) * u_x
                d_p = abs((cy - c_y) * u_x - (cx - c_x) * u_y)

                if (d_p <= 12.0) and (t_s <= t_c <= t_e):
                    absorbed_by_geometry = True

            # If absorbed by either, increment counters
            if absorbed_by_id or absorbed_by_geometry:
                absorbed_total_count += 1
                absorbed_total_verdicts[v_cand] += 1

                if absorbed_by_id:
                    absorbed_by_id_count += 1
                    absorbed_id_verdicts[v_cand] += 1
                else:
                    absorbed_by_geom_only_count += 1
                    absorbed_geom_only_verdicts[v_cand] += 1
            else:
                # Retain as Single Candidate
                single_rec = cand_row.to_dict()
                single_verdict = verdict_map.get((stem, cid), "unlabeled")
                single_verdict_source = "g1_review" if (stem, cid) in verdict_map else "none"

                single_rec.update({
                    "cand_v2_id": f"{stem}_C{cid}",
                    "type": "single",
                    "frame": stem,
                    "frame_name": f"{stem}.fits",
                    "line_id": np.nan,
                    "candidate_id": cid,
                    "verdict": single_verdict,
                    "verdict_source": single_verdict_source,
                    "conflict": False,
                    "n_streak": 1 if single_verdict == "streak" else 0,
                    "n_fp": 1 if single_verdict == "fp" else 0,
                    "n_unlabeled": 1 if single_verdict == "unlabeled" else 0,
                    "fragment_ids": "",
                    "n_fragments": 1,
                    "length_px": round(float(cand_row["length_px"]), 1) if "length_px" in cand_row and pd.notna(cand_row["length_px"]) else np.nan,
                    "angle_deg": round(float(cand_row["angle_from_vertical"]), 2) if "angle_from_vertical" in cand_row and pd.notna(cand_row["angle_from_vertical"]) else np.nan,
                    "z_peak": np.nan,
                    "n_segments_z_gt_3": np.nan,
                    "endpoints": "",
                    "centroid": f"({cy:.2f}, {cx:.2f})",
                    "centroid_row": round(cy, 2),
                    "centroid_col": round(cx, 2),
                })
                v2_records.append(single_rec)

    # 4. Save to data/d0b/candidates_v2.csv
    df_v2 = pd.DataFrame(v2_records)
    out_csv_path = base_dir / "data/d0b/candidates_v2.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_v2.to_csv(out_csv_path, index=False)

    # 5. Compute comparative metrics
    v2_total_cands = len(df_v2)
    v2_lines = len(df_v2[df_v2["type"] == "line"])
    v2_singles = len(df_v2[df_v2["type"] == "single"])

    v2_line_streaks = len(df_v2[(df_v2["type"] == "line") & (df_v2["verdict"] == "streak")])
    v2_single_streaks = len(df_v2[(df_v2["type"] == "single") & (df_v2["verdict"] == "streak")])
    v2_total_streaks = v2_line_streaks + v2_single_streaks

    v2_fp = len(df_v2[df_v2["verdict"] == "fp"])
    v2_unlabeled = len(df_v2[df_v2["verdict"] == "unlabeled"])

    new_streak_frames = set(df_v2[df_v2["verdict"] == "streak"]["frame"].unique())

    # Frames where accepted line exists AND streak-labeled single remains
    coexistence_frames = []
    for f in sorted(frames_with_accepted_line):
        singles_in_f = df_v2[(df_v2["frame"] == f) & (df_v2["type"] == "single") & (df_v2["verdict"] == "streak")]
        if not singles_in_f.empty:
            coexistence_frames.append((f, singles_in_f["candidate_id"].tolist()))

    # Print Report
    print("=" * 80)
    print("CANDIDATES V2 BUILD & COMPARATIVE COUNT REPORT")
    print("=" * 80)

    print("\n1. OLD (G1) CANDIDATES SUMMARY:")
    print("-" * 50)
    print(f"  Candidates total:     {old_total_cands}")
    print(f"  Labeled streak:       {old_labeled_streak}")
    print(f"  Labeled fp:           {old_labeled_fp}")
    print(f"  Labeled by review:    {old_labeled_review}")
    print(f"  p >= 0.5 calls:       {old_p_ge_05}")

    print("\n2. NEW (V2) CANDIDATES SUMMARY:")
    print("-" * 50)
    print(f"  Candidates total:     {v2_total_cands}")
    print(f"  Lines:                {v2_lines}")
    print(f"  Singles:              {v2_singles}")
    print(f"  Labeled streak total: {v2_total_streaks}")
    print(f"    - Lines:            {v2_line_streaks}")
    print(f"    - Singles:          {v2_single_streaks}")
    print(f"  Labeled fp:           {v2_fp}")
    print(f"  Unlabeled:            {v2_unlabeled}")

    print("\n3. G1 CANDIDATES ABSORPTION BREAKDOWN:")
    print("-" * 50)
    print(f"  Absorbed by ID:              {absorbed_by_id_count:3d}  (streak: {absorbed_id_verdicts['streak']}, fp: {absorbed_id_verdicts['fp']}, unlabeled: {absorbed_id_verdicts['unlabeled']})")
    print(f"  Absorbed by geometry only:   {absorbed_by_geom_only_count:3d}  (streak: {absorbed_geom_only_verdicts['streak']}, fp: {absorbed_geom_only_verdicts['fp']}, unlabeled: {absorbed_geom_only_verdicts['unlabeled']})")
    print(f"  Absorbed in total:           {absorbed_total_count:3d}  (streak: {absorbed_total_verdicts['streak']}, fp: {absorbed_total_verdicts['fp']}, unlabeled: {absorbed_total_verdicts['unlabeled']})")

    print("\n4. LINES WITH CONFLICT (both streak and fp members):")
    print("-" * 50)
    if lines_with_conflict:
        for f, lid, ns, nfp, fids in lines_with_conflict:
            print(f"  Frame {f} Line {lid}: {ns} streak members, {nfp} fp members | fragments: {fids}")
    else:
        print("  None")

    print("\n5. FRAMES WITH AT LEAST ONE STREAK-LABELED CANDIDATE:")
    print("-" * 50)
    print(f"  Old (G1) streak frames ({len(old_streak_frames)}): {sorted(list(old_streak_frames))}")
    print(f"  New (V2) streak frames ({len(new_streak_frames)}): {sorted(list(new_streak_frames))}")
    added_frames = sorted(list(new_streak_frames - old_streak_frames))
    removed_frames = sorted(list(old_streak_frames - new_streak_frames))
    print(f"  New frames added:     {added_frames}")
    print(f"  Old frames removed:   {removed_frames}")

    print("\n6. FRAMES WITH BOTH AN ACCEPTED LINE AND A STREAK-LABELED SINGLE:")
    print("-" * 50)
    if coexistence_frames:
        for f, cids in coexistence_frames:
            print(f"  Frame {f}: accepted line present, remaining streak singles: {cids}")
    else:
        print("  None")

    print("\n" + "=" * 80)
    print(f"Output saved successfully to: {out_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
