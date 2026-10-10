# TASK G9e: reusable visualizer
"""
Reusable visualizer for ILMT streak detection candidates and frames.

Usage Examples:
  1. Draw all candidates in a frame colored by verdict:
     python scripts/viz_check.py --frame 20230519-0009 --all

  2. Inspect specific single candidates with zooms and overlays:
     python scripts/viz_check.py --frame 20221026-0059 --cand 270 10323

  3. Inspect an accepted line with 3 length zooms and stacked strips:
     python scripts/viz_check.py --line 20240502-0026_L1

  4. Zoom in on any arbitrary pixel coordinate:
     python scripts/viz_check.py --frame 20221026-0059 --center 1980 415 --size 600 --stretch asinh

  5. Generate a contact sheet from an audit or label CSV and open in Explorer:
     python scripts/viz_check.py --from-csv data/d0b/audit_streak_singles.csv --top 12 --open
"""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import AsinhNorm, LogNorm, Normalize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.io_utils import load_fits


def parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' into ((r0, c0), (r1, c1))."""
    if not ep_str or pd.isna(ep_str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def get_crop(img, cy, cx, size):
    """
    Extract a square crop of given size centered at (cy, cx) with constant 0 padding.
    Directly copied from scripts/viz_g9b.py.
    cy: row (y), cx: col (x).
    Returns (crop, (c0, c1, r0, r1), (c0_cl, c1_cl, r0_cl, r1_cl)).
    """
    half = size // 2
    r0 = int(round(cy)) - half
    r1 = r0 + size
    c0 = int(round(cx)) - half
    c1 = c0 + size

    h, w = img.shape
    r0_cl = max(0, r0)
    r1_cl = min(h, r1)
    c0_cl = max(0, c0)
    c1_cl = min(w, c1)

    sub_crop = img[r0_cl:r1_cl, c0_cl:c1_cl]
    pad_top = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left = max(0, -c0)
    pad_right = max(0, c1 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(sub_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    else:
        crop = sub_crop
    return crop, (c0, c1, r0, r1), (c0_cl, c1_cl, r0_cl, r1_cl)


def find_fits_file(stem):
    """
    Search for FITS file by stem in standard project search directories.
    Searches test_files, sample_notebook/data, ILMT_Contest_Materials root, and data/.
    """
    contest_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    search_dirs = [
        contest_root / "test_files",
        contest_root / "sample_notebook" / "data",
        contest_root,
        base_dir / "data",
    ]
    target_name = f"{stem.lower()}.fits" if not stem.lower().endswith(".fits") else stem.lower()

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.rglob("*.fits"):
            if p.name.lower() == target_name:
                return p, search_dirs
    return None, search_dirs


def get_stretch_norm(data, stretch_type="asinh", vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0):
    """Compute normalization stretch object matching scripts/viz_g9b.py."""
    vmin = float(np.percentile(data, vmin_pct))
    vmax = float(np.percentile(data, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0

    st = str(stretch_type).lower()
    if st == "log":
        pos_data = data[data > 0]
        v_min_log = float(np.percentile(pos_data, 1.0)) if len(pos_data) > 0 else 1e-2
        return LogNorm(vmin=max(1e-2, v_min_log), vmax=max(1.0, vmax)), vmin, vmax
    elif st == "linear":
        return Normalize(vmin=vmin, vmax=vmax), vmin, vmax
    else:
        # Default asinh
        try:
            return AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax), vmin, vmax
        except Exception:
            return None, vmin, vmax


def save_clean_image(out_path, img_data, norm=None, vmin=None, vmax=None):
    """Save raw image with NO overlay, NO markers, NO text, and NO axes."""
    if norm is not None:
        scaled = np.clip(norm(img_data), 0.0, 1.0)
        plt.imsave(out_path, scaled, cmap="gray", origin="lower")
    else:
        plt.imsave(out_path, img_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)


def load_label_overrides():
    """Load data/d0b/label_overrides.csv if it exists."""
    overrides = {}
    p = base_dir / "data/d0b/label_overrides.csv"
    if p.exists():
        try:
            df_ov = pd.read_csv(p)
            for _, r in df_ov.iterrows():
                fn = Path(str(r["frame_name"])).stem.lower()
                cid = str(int(r["candidate_id"])) if pd.notna(r.get("candidate_id")) else str(r["candidate_id"])
                overrides[(fn, cid)] = str(r["new_label"]).strip().lower()
        except Exception as e:
            print(f"Warning: Failed to load label overrides: {e}")
    return overrides


def get_candidate_label_info(cand_row, overrides, stem):
    """Get verdict string and display label (with asterisk if overridden)."""
    orig_verdict = str(cand_row.get("verdict", "unlabeled")).strip().lower()
    cid_str = str(int(cand_row["candidate_id"])) if pd.notna(cand_row.get("candidate_id")) else str(cand_row.get("cand_v2_id"))
    
    key = (stem.lower(), cid_str)
    if key in overrides:
        display_label = f"{overrides[key]}*"
        color_verdict = overrides[key]
    else:
        display_label = orig_verdict
        color_verdict = orig_verdict

    # Assign distinct colors
    if color_verdict == "streak":
        color = "cyan"
    elif color_verdict == "fp":
        color = "red"
    else:
        color = "yellow"

    return display_label, color


def print_crop_info(name, path, req_bounds, cl_bounds):
    """Print saved path and crop bounds after clipping at frame edges."""
    c0, c1, r0, r1 = req_bounds
    c0_cl, c1_cl, r0_cl, r1_cl = cl_bounds
    print(f"Saved: {path.relative_to(base_dir)}")
    print(f"  Crop Bounds [{name}]: requested row [{r0}, {r1}], col [{c0}, {c1}] -> in-frame row [{r0_cl}, {r1_cl}], col [{c0_cl}, {c1_cl}]")
    if (c0 != c0_cl) or (c1 != c1_cl) or (r0 != r0_cl) or (r1 != r1_cl):
        print(f"  (Note: clipped at frame border and zero-padded to preserve center)")


def generate_line_strip(sub, ep0, ep1, length_px, norm_stretch, vmin_s, vmax_s, out_path, cand_id, frame_name, angle_deg):
    """Generate 4-panel strip visualization matching scripts/viz_g9b.py."""
    ep0_r, ep0_c = ep0
    ep1_r, ep1_c = ep1
    dr = ep1_r - ep0_r
    dc = ep1_c - ep0_c
    ur = dr / max(length_px, 1e-4)
    uc = dc / max(length_px, 1e-4)

    t_vals = np.arange(0.0, length_px + 1.0, 1.0)
    w_strip = np.arange(-7.0, 8.0, 1.0)  # 15 px width (-7 to +7)

    T_g, W_g = np.meshgrid(t_vals, w_strip)
    Y_c = (ep0_r + T_g * ur) - W_g * uc
    X_c = (ep0_c + T_g * uc) + W_g * ur
    strip_center = map_coordinates(sub, [Y_c.ravel(), X_c.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_c.shape)

    W_g_m25 = W_g - 25.0
    Y_m25 = (ep0_r + T_g * ur) - W_g_m25 * uc
    X_m25 = (ep0_c + T_g * uc) + W_g_m25 * ur
    strip_m25 = map_coordinates(sub, [Y_m25.ravel(), X_m25.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_m25.shape)

    W_g_p25 = W_g + 25.0
    Y_p25 = (ep0_r + T_g * ur) - W_g_p25 * uc
    X_p25 = (ep0_c + T_g * uc) + W_g_p25 * ur
    strip_p25 = map_coordinates(sub, [Y_p25.ravel(), X_p25.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_p25.shape)

    # Bin horizontally if long
    n_t = len(t_vals)
    bin_f = max(1, int(math.ceil(n_t / 1200.0)))
    n_trim = (n_t // bin_f) * bin_f
    if bin_f > 1:
        s_m25_disp = strip_m25[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
        s_c_disp = strip_center[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
        s_p25_disp = strip_p25[:, :n_trim].reshape(15, n_trim // bin_f, bin_f).mean(axis=2)
    else:
        s_m25_disp = strip_m25
        s_c_disp = strip_center
        s_p25_disp = strip_p25

    w_prof = np.arange(-40.0, 41.0, 1.0)
    T_p_grid, W_p_grid = np.meshgrid(t_vals, w_prof)
    Y_prof = (ep0_r + T_p_grid * ur) - W_p_grid * uc
    X_prof = (ep0_c + T_p_grid * uc) + W_p_grid * ur
    prof_map = map_coordinates(sub, [Y_prof.ravel(), X_prof.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_prof.shape)
    perp_profile = np.median(prof_map, axis=1)

    fig_s = plt.figure(figsize=(13, 7), dpi=140)
    gs = fig_s.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 1.2, 2.0], hspace=0.35)

    im_kwargs = {"cmap": "gray", "origin": "lower", "aspect": "auto"}
    if norm_stretch is not None:
        im_kwargs["norm"] = norm_stretch
    else:
        im_kwargs["vmin"] = vmin_s
        im_kwargs["vmax"] = vmax_s

    ax_s1 = fig_s.add_subplot(gs[0])
    ax_s1.imshow(s_m25_disp, **im_kwargs)
    ax_s1.set_title("Parallel Control Strip: offset = -25 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold")
    ax_s1.set_ylabel("w [-32..-18]", fontsize=7.5)

    ax_s2 = fig_s.add_subplot(gs[1])
    ax_s2.imshow(s_c_disp, **im_kwargs)
    ax_s2.set_title("On-Line Streak Strip: offset = 0 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold", color="blue")
    ax_s2.set_ylabel("w [-7..+7]", fontsize=7.5)

    ax_s3 = fig_s.add_subplot(gs[2])
    ax_s3.imshow(s_p25_disp, **im_kwargs)
    ax_s3.set_title("Parallel Control Strip: offset = +25 px (width 15 px) [same stretch]", fontsize=9, fontweight="bold")
    ax_s3.set_ylabel("w [+18..+32]", fontsize=7.5)
    ax_s3.set_xlabel(f"Along-line distance t ({'binned x' + str(bin_f) if bin_f > 1 else 'px'})", fontsize=8)

    ax_s4 = fig_s.add_subplot(gs[3])
    ax_s4.plot(w_prof, perp_profile, color="black", lw=1.5, label="Median Along-Line Profile")
    ax_s4.axvline(0.0, color="blue", lw=1.2, linestyle="--", alpha=0.7, label="Line Center (0 px)")
    ax_s4.axvspan(-7.0, 7.0, color="blue", alpha=0.08, label="On-Line Width (15 px)")
    ax_s4.axvline(-25.0, color="red", lw=1.0, linestyle=":", alpha=0.7, label="Control Offsets (±25 px)")
    ax_s4.axvline(25.0, color="red", lw=1.0, linestyle=":", alpha=0.7)
    ax_s4.set_title("Transverse Profile (Median Along Line vs Perpendicular Offset w)", fontsize=9.5, fontweight="bold")
    ax_s4.set_xlabel("Perpendicular Offset w (px)")
    ax_s4.set_ylabel("Median ADU")
    ax_s4.grid(True, linestyle=":", alpha=0.6)
    ax_s4.legend(loc="upper right", fontsize=7.5, framealpha=0.7)

    fig_s.suptitle(f"{cand_id} Strip Verification ({frame_name}) | Length={length_px:.1f}px | Angle={angle_deg:.1f}°", fontsize=11, fontweight="bold")
    plt.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig_s)
    print(f"Saved: {out_path.relative_to(base_dir)}")


def run_contact_sheet(csv_path, top_n, stretch_type, should_open):
    """Build a multi-candidate contact sheet from a CSV list of (frame_name, candidate_id)."""
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found.")
        sys.exit(1)

    df_req = pd.read_csv(csv_path)
    if "frame_name" not in df_req.columns or "candidate_id" not in df_req.columns:
        print(f"Error: CSV {csv_path} must have 'frame_name' and 'candidate_id' columns.")
        sys.exit(1)

    if top_n is not None and top_n > 0:
        df_req = df_req.head(top_n)

    n_items = len(df_req)
    if n_items == 0:
        print("CSV contains 0 rows.")
        return

    # Load candidates_v2.csv and overrides
    cands_path = base_dir / "data/d0b/candidates_v2.csv"
    cands_lookup = {}
    if cands_path.exists():
        df_c2 = pd.read_csv(cands_path)
        for _, r in df_c2.iterrows():
            f_stem = Path(str(r["frame_name"])).stem.lower()
            cid = str(int(r["candidate_id"])) if pd.notna(r.get("candidate_id")) else str(r["cand_v2_id"])
            cands_lookup[(f_stem, cid)] = r

    overrides = load_label_overrides()

    n_cols = min(3, n_items)
    n_rows = int(math.ceil(n_items / float(n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows), dpi=120)
    if n_items == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])

    fits_cache = {}

    for idx, (_, r_req) in enumerate(df_req.iterrows()):
        row_i = idx // n_cols
        col_i = idx % n_cols
        ax = axes[row_i, col_i]

        fn = str(r_req["frame_name"])
        f_stem = Path(fn).stem.lower()
        cid_str = str(int(r_req["candidate_id"])) if str(r_req["candidate_id"]).replace('.','',1).isdigit() else str(r_req["candidate_id"])

        # Locate candidate coords
        cand_row = cands_lookup.get((f_stem, cid_str))
        if cand_row is not None:
            cy = float(cand_row["centroid_row"])
            cx = float(cand_row["centroid_col"])
            disp_label, color = get_candidate_label_info(cand_row, overrides, f_stem)
        else:
            cy, cx = 2048.0, 2048.0
            disp_label, color = "unknown", "gray"

        # Load FITS
        if f_stem not in fits_cache:
            p_fits, _ = find_fits_file(f_stem)
            if p_fits and p_fits.exists():
                d_raw, _ = load_fits(p_fits)
                bg_raw, _ = estimate_background(d_raw, method="background2d")
                fits_cache[f_stem] = d_raw - bg_raw
            else:
                fits_cache[f_stem] = None

        sub = fits_cache[f_stem]
        if sub is not None:
            crop, req_b, cl_b = get_crop(sub, cy, cx, 300)
            norm_c, vmin_c, vmax_c = get_stretch_norm(crop, stretch_type=stretch_type)
            if norm_c is not None:
                ax.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_c)
            else:
                ax.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_c, vmax=vmax_c)

            # Draw subtle center marker
            ax.scatter([cx], [cy], s=40, facecolors="none", edgecolors=color, linewidth=1.2, alpha=0.8)
            ax.set_xlim(req_b[0], req_b[1])
            ax.set_ylim(req_b[2], req_b[3])
            title_txt = f"{fn} | ID: {cid_str}\nLabel: {disp_label}"
            ax.set_title(title_txt, fontsize=10, fontweight="bold", color="darkblue" if color != "red" else "darkred")
        else:
            ax.text(0.5, 0.5, f"FITS missing:\n{fn}", ha="center", va="center", color="red")
            ax.axis("off")

    # Blank out any remaining unused subplots in grid
    for idx in range(n_items, n_rows * n_cols):
        row_i = idx // n_cols
        col_i = idx % n_cols
        axes[row_i, col_i].axis("off")

    plt.tight_layout()
    viz_dir = base_dir / "data/viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_sheet = viz_dir / f"contact_sheet_{csv_path.stem}.png"
    plt.savefig(out_sheet, bbox_inches="tight", dpi=120)
    plt.close(fig)

    print(f"\nSaved contact sheet to: {out_sheet.relative_to(base_dir)}")
    if should_open:
        open_folder(viz_dir)


def open_folder(folder_path):
    """Open folder in Windows Explorer."""
    try:
        os.startfile(str(folder_path))
    except Exception:
        try:
            subprocess.Popen(["explorer", str(folder_path)])
        except Exception as e:
            print(f"Warning: Could not open folder in Explorer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reusable ILMT Streak Visualizer (TASK G9e)")
    parser.add_argument("--frame", type=str, default=None, help="Frame name with or without .fits (e.g. 20230519-0009)")
    parser.add_argument("--cand", nargs="+", type=str, default=None, help="candidate_ids to draw and zoom on (singles)")
    parser.add_argument("--line", type=str, default=None, help="e.g. 20240502-0026_L1 (draws line, 3 zooms, strip)")
    parser.add_argument("--center", nargs=2, type=float, default=None, metavar=("ROW", "COL"), help="Center coordinate for custom zoom (row col)")
    parser.add_argument("--size", type=int, default=300, help="Crop size in px for custom zoom (default: 300)")
    parser.add_argument("--all", action="store_true", help="Draw every candidate in the frame, coloured by label")
    parser.add_argument("--from-csv", type=str, default=None, help="Path to CSV containing frame_name and candidate_id for contact sheet")
    parser.add_argument("--top", type=int, default=None, help="Max rows for contact sheet")
    parser.add_argument("--stretch", type=str, choices=["asinh", "log", "linear"], default="asinh", help="Image stretch (default: asinh)")
    parser.add_argument("--open", action="store_true", help="Open output folder in Explorer when done")

    args = parser.parse_args()

    print("=" * 80)
    print("TASK G9e: REUSABLE VISUALIZER")
    print("=" * 80)
    print("Row/Column Convention Used:")
    print("  - row = y (vertical axis, FITS origin='lower')")
    print("  - col = x (horizontal axis)")
    print("-" * 80)

    # 1. Contact sheet mode
    if args.from_csv:
        csv_p = Path(args.from_csv)
        if not csv_p.is_absolute():
            csv_p = base_dir / csv_p
        run_contact_sheet(csv_p, args.top, args.stretch, args.open)
        return

    # Determine frame name
    frame_arg = args.frame
    if not frame_arg and args.line:
        # Extract frame stem from line id e.g. 20240502-0026_L1
        frame_arg = args.line.split("_")[0]

    if not frame_arg:
        print("Error: Please provide --frame NAME, --line LINE_ID, or --from-csv PATH.")
        sys.exit(1)

    frame_stem = Path(frame_arg).stem
    frame_name = f"{frame_stem}.fits"

    # FITS Lookup
    fits_path, searched_dirs = find_fits_file(frame_stem)
    if not fits_path or not fits_path.exists():
        print(f"Error: FITS file for '{frame_stem}' not found.")
        print("Searched directories:")
        for sd in searched_dirs:
            print(f"  - {sd}")
        sys.exit(1)

    print(f"Found FITS file: {fits_path}")

    # Output directory: data/viz/<frame>/
    out_dir = base_dir / "data/viz" / frame_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.relative_to(base_dir)}\n")

    # Load FITS data and background
    data, header = load_fits(fits_path)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    h_img, w_img = data.shape

    # Load candidates_v2.csv and label overrides
    overrides = load_label_overrides()
    cands_path = base_dir / "data/d0b/candidates_v2.csv"
    frame_cands = []
    if cands_path.exists():
        df_c2 = pd.read_csv(cands_path)
        mask = (df_c2["frame_name"] == frame_name) | (df_c2["frame"] == frame_stem)
        frame_cands_df = df_c2[mask]
        frame_cands = frame_cands_df.to_dict("records")
        if len(frame_cands) == 0:
            print(f"no candidates in candidates_v2.csv for this frame ({frame_name})")
        else:
            print(f"Loaded {len(frame_cands)} candidates in candidates_v2.csv for {frame_name}")
    else:
        print(f"no candidates in candidates_v2.csv for this frame (file missing)")

    # 1. Clean Overview: ALWAYS saved for every --frame run
    scale_ds = max(1, int(math.ceil(max(h_img, w_img) / 1000.0)))
    h_ds = h_img // scale_ds
    w_ds = w_img // scale_ds
    frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
    vmin_ov = float(np.percentile(frame_ds, 1.0))
    vmax_ov = float(np.percentile(frame_ds, 99.5))

    clean_ov_path = out_dir / f"{frame_stem}_clean_overview.png"
    save_clean_image(clean_ov_path, frame_ds, vmin=vmin_ov, vmax=vmax_ov)
    print_crop_info("Full Frame Clean Overview", clean_ov_path, (0, w_img, 0, h_img), (0, w_img, 0, h_img))

    # 2. Overlay Overview (only when candidates are selected: --cand, --line, or --all)
    should_draw_overlay = bool(args.all or args.cand or args.line)
    if should_draw_overlay:
        fig_ov, ax_ov = plt.subplots(figsize=(9, 9), dpi=140)
        ax_ov.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin_ov, vmax=vmax_ov)

        # Draw requested candidates
        if args.all:
            cands_to_draw = frame_cands
        elif args.cand:
            selected_ids = set([str(c) for c in args.cand])
            cands_to_draw = [
                c for c in frame_cands
                if str(int(c["candidate_id"])) in selected_ids or str(c.get("cand_v2_id")) in selected_ids
            ]
        else:
            cands_to_draw = []

        for c_entry in cands_to_draw:
            cy = float(c_entry["centroid_row"])
            cx = float(c_entry["centroid_col"])
            disp_label, color = get_candidate_label_info(c_entry, overrides, frame_stem)
            cid_str = str(int(c_entry["candidate_id"])) if pd.notna(c_entry.get("candidate_id")) else str(c_entry["cand_v2_id"])

            if str(c_entry.get("type", "")).lower() == "line":
                eps = parse_endpoints(c_entry.get("endpoints"))
                if eps is not None:
                    ax_ov.plot([eps[0][1], eps[1][1]], [eps[0][0], eps[1][0]], color=color, linewidth=1.2, alpha=0.6)
            else:
                r0 = float(c_entry["bbox_r0"]) if pd.notna(c_entry.get("bbox_r0")) else cy - 10
                r1 = float(c_entry["bbox_r1"]) if pd.notna(c_entry.get("bbox_r1")) else cy + 10
                c0 = float(c_entry["bbox_c0"]) if pd.notna(c_entry.get("bbox_c0")) else cx - 10
                c1 = float(c_entry["bbox_c1"]) if pd.notna(c_entry.get("bbox_c1")) else cx + 10
                rect = patches.Rectangle((c0, r0), max(10, c1 - c0), max(10, r1 - r0), linewidth=1.0, edgecolor=color, facecolor="none", alpha=0.6)
                ax_ov.add_patch(rect)

            ax_ov.text(cx + 25, cy, f"{cid_str} ({disp_label})", color=color, fontsize=8, alpha=0.85)

        # If line requested, also draw line endpoints
        if args.line:
            match_lines = [c for c in frame_cands if str(c.get("cand_v2_id")) == args.line or str(c.get("line_id")) == args.line]
            for l_entry in match_lines:
                eps = parse_endpoints(l_entry.get("endpoints"))
                if eps is not None:
                    ax_ov.plot([eps[0][1], eps[1][1]], [eps[0][0], eps[1][0]], color="cyan", linewidth=1.4, alpha=0.7)
                    ax_ov.text(eps[0][1] + 20, eps[0][0], f"{args.line}", color="cyan", fontsize=8.5, fontweight="bold")

        ax_ov.set_xlim(0, w_img)
        ax_ov.set_ylim(0, h_img)
        ax_ov.set_title(f"Overview: {frame_name} | Overlays ({args.stretch} stretch)", fontsize=10)
        ax_ov.set_xlabel("Column (px)")
        ax_ov.set_ylabel("Row (px)")
        plt.tight_layout()

        ov_path = out_dir / f"{frame_stem}_overview.png"
        plt.savefig(ov_path, bbox_inches="tight", dpi=140)
        plt.close(fig_ov)
        print(f"Saved: {ov_path.relative_to(base_dir)}")

    # 3. Candidate Zooms (--cand)
    if args.cand and frame_cands:
        cand_size = args.size if args.size else 300
        for cand_id_arg in args.cand:
            match_cand = [
                c for c in frame_cands
                if str(c.get("cand_v2_id")) == str(cand_id_arg) or
                   (pd.notna(c.get("candidate_id")) and str(int(c["candidate_id"])) == str(cand_id_arg))
            ]
            if not match_cand:
                print(f"Warning: Candidate {cand_id_arg} not found in {frame_name}.")
                continue

            c_row = match_cand[0]
            cy = float(c_row["centroid_row"])
            cx = float(c_row["centroid_col"])
            disp_label, color = get_candidate_label_info(c_row, overrides, frame_stem)
            cid_str = str(int(c_row["candidate_id"])) if pd.notna(c_row.get("candidate_id")) else str(c_row["cand_v2_id"])

            crop, req_b, cl_b = get_crop(sub, cy, cx, cand_size)
            norm_z, vmin_z, vmax_z = get_stretch_norm(crop, stretch_type=args.stretch)

            # A) Zoom with overlay
            fig_z, ax_z = plt.subplots(figsize=(6, 6), dpi=140)
            if norm_z is not None:
                ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_z)
            else:
                ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_z, vmax=vmax_z)

            if pd.notna(c_row.get("bbox_r0")):
                r0, r1 = float(c_row["bbox_r0"]), float(c_row["bbox_r1"])
                c0, c1 = float(c_row["bbox_c0"]), float(c_row["bbox_c1"])
                rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=1.1, edgecolor=color, facecolor="none", alpha=0.7)
                ax_z.add_patch(rect)
            ax_z.scatter([cx], [cy], s=40, facecolors="none", edgecolors=color, linewidth=1.1, alpha=0.8)
            ax_z.text(req_b[0] + 10, req_b[3] - 25, f"ID: {cid_str} | {disp_label}", color=color, fontsize=9, fontweight="bold")

            ax_z.set_xlim(req_b[0], req_b[1])
            ax_z.set_ylim(req_b[2], req_b[3])
            ax_z.set_title(f"{frame_stem} Cand {cid_str} ({disp_label})", fontsize=9.5)
            ax_z.set_xlabel("Column (px)")
            ax_z.set_ylabel("Row (px)")
            plt.tight_layout()

            z_path = out_dir / f"{frame_stem}_cand_{cid_str}_zoom.png"
            plt.savefig(z_path, bbox_inches="tight", dpi=140)
            plt.close(fig_z)
            print_crop_info(f"Cand {cid_str} Zoom", z_path, req_b, cl_b)

            # B) Zoom clean (no overlays, no markers, no text)
            zc_path = out_dir / f"{frame_stem}_cand_{cid_str}_zoom_clean.png"
            save_clean_image(zc_path, crop, norm=norm_z, vmin=vmin_z, vmax=vmax_z)
            print_crop_info(f"Cand {cid_str} Clean Zoom", zc_path, req_b, cl_b)

    # 4. Line Mode (--line LINE_ID)
    if args.line:
        match_lines = [
            c for c in frame_cands
            if str(c.get("cand_v2_id")) == args.line or str(c.get("line_id")) == args.line
        ]
        if not match_lines:
            print(f"Warning: Line candidate '{args.line}' not found in candidates_v2.csv for {frame_name}.")
        else:
            l_row = match_lines[0]
            eps = parse_endpoints(l_row.get("endpoints"))
            if eps is None:
                print(f"Error: Endpoints missing for line {args.line}")
            else:
                (ep0_r, ep0_c), (ep1_r, ep1_c) = eps
                dr = ep1_r - ep0_r
                dc = ep1_c - ep0_c
                length_px = float(l_row["length_px"]) if pd.notna(l_row.get("length_px")) else math.hypot(dr, dc)
                ang_deg = float(l_row["angle_deg"]) if pd.notna(l_row.get("angle_deg")) else 0.0

                # 3 zooms at 25%, 50%, 75%
                for frac, pct_str in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
                    zf_r = ep0_r + frac * dr
                    zf_c = ep0_c + frac * dc

                    crop, req_b, cl_b = get_crop(sub, zf_r, zf_c, 400)
                    norm_z, vmin_z, vmax_z = get_stretch_norm(crop, stretch_type=args.stretch)

                    # Overlay zoom
                    fig_z, ax_z = plt.subplots(figsize=(6, 6), dpi=140)
                    if norm_z is not None:
                        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_z)
                    else:
                        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_z, vmax=vmax_z)

                    ax_z.plot([ep0_c, ep1_c], [ep0_r, ep1_r], color="cyan", linewidth=1.1, alpha=0.55, linestyle="--")
                    ax_z.set_xlim(req_b[0], req_b[1])
                    ax_z.set_ylim(req_b[2], req_b[3])
                    ax_z.set_title(f"{args.line} Zoom @ {pct_str}% Length", fontsize=9.5)
                    ax_z.set_xlabel("Column (px)")
                    ax_z.set_ylabel("Row (px)")
                    plt.tight_layout()

                    z_path = out_dir / f"{args.line}_zoom_{pct_str}pct.png"
                    plt.savefig(z_path, bbox_inches="tight", dpi=140)
                    plt.close(fig_z)
                    print_crop_info(f"Line Zoom {pct_str}%", z_path, req_b, cl_b)

                    # Clean zoom
                    zc_path = out_dir / f"{args.line}_zoom_{pct_str}pct_clean.png"
                    save_clean_image(zc_path, crop, norm=norm_z, vmin=vmin_z, vmax=vmax_z)
                    print_crop_info(f"Line Clean Zoom {pct_str}%", zc_path, req_b, cl_b)

                # Line Strip
                strip_path = out_dir / f"{args.line}_strip.png"
                norm_str, vmin_s, vmax_s = get_stretch_norm(sub, stretch_type=args.stretch, vmin_pct=3.0, vmax_pct=99.5)
                generate_line_strip(sub, (ep0_r, ep0_c), (ep1_r, ep1_c), length_px, norm_str, vmin_s, vmax_s, strip_path, args.line, frame_name, ang_deg)

    # 5. Arbitrary Location Zoom (--center ROW COL --size N)
    if args.center:
        cr_r, cr_c = float(args.center[0]), float(args.center[1])
        c_size = args.size if args.size else 300

        crop, req_b, cl_b = get_crop(sub, cr_r, cr_c, c_size)
        norm_z, vmin_z, vmax_z = get_stretch_norm(crop, stretch_type=args.stretch)

        # Overlay zoom
        fig_z, ax_z = plt.subplots(figsize=(6, 6), dpi=140)
        if norm_z is not None:
            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_z)
        else:
            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_z, vmax=vmax_z)

        # Subtle crosshair
        ax_z.axvline(cr_c, color="yellow", lw=0.8, linestyle=":", alpha=0.6)
        ax_z.axhline(cr_r, color="yellow", lw=0.8, linestyle=":", alpha=0.6)
        ax_z.set_xlim(req_b[0], req_b[1])
        ax_z.set_ylim(req_b[2], req_b[3])
        ax_z.set_title(f"{frame_stem} Zoom at ({int(cr_r)}, {int(cr_c)}) size={c_size}", fontsize=9.5)
        ax_z.set_xlabel("Column (px)")
        ax_z.set_ylabel("Row (px)")
        plt.tight_layout()

        z_path = out_dir / f"{frame_stem}_center_{int(cr_r)}_{int(cr_c)}_zoom_{c_size}.png"
        plt.savefig(z_path, bbox_inches="tight", dpi=140)
        plt.close(fig_z)
        print_crop_info("Custom Center Zoom", z_path, req_b, cl_b)

        # Clean zoom
        zc_path = out_dir / f"{frame_stem}_center_{int(cr_r)}_{int(cr_c)}_zoom_{c_size}_clean.png"
        save_clean_image(zc_path, crop, norm=norm_z, vmin=vmin_z, vmax=vmax_z)
        print_crop_info("Custom Center Clean Zoom", zc_path, req_b, cl_b)

    print("-" * 80)
    print("Visualizer run completed.")
    if args.open:
        open_folder(out_dir)


if __name__ == "__main__":
    main()
