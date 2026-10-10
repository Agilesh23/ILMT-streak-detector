# TASK G9j + G9k: build label_tool batch from candidates_v2.csv (blind)
"""
Build a label_tool.py batch CSV and PNG images from candidates_v2.csv.

Usage:
  # All streak-labeled singles (33):
  python scripts/make_review_batch_v2.py --streak-singles --name streak_singles

  # All accepted lines (type == line):
  python scripts/make_review_batch_v2.py --lines --name all_lines

  # Every candidate in one frame:
  python scripts/make_review_batch_v2.py --frame 20221026-0059 --name frame_059

  # Combine flags; --name is always required:
  python scripts/make_review_batch_v2.py --streak-singles --lines --name full

Output:
  data/labeled/review_v2_<name>_labels.csv
  outputs/review_v2/<name>/<frame_stem>_<cand_id>_cutout.png
  outputs/review_v2/<name>/<frame_stem>_<cand_id>_context.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

# ── project paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.background import estimate_background
from src.io_utils import load_fits

CANDIDATES_CSV   = BASE_DIR / "data" / "d0b" / "candidates_v2.csv"
LABELED_DIR      = BASE_DIR / "data" / "labeled"
FITS_SEARCH_ROOT = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")


# ─────────────────────────────────────────────────────────────────────────────
# FITS lookup  (copied from scripts/viz_check.py)
# ─────────────────────────────────────────────────────────────────────────────
def find_fits_file(stem: str):
    """Return Path to FITS file for given stem, or None."""
    search_dirs = [
        FITS_SEARCH_ROOT / "test_files",
        FITS_SEARCH_ROOT / "sample_notebook" / "data",
        FITS_SEARCH_ROOT,
        BASE_DIR / "data",
    ]
    target_name = stem.lower()
    if not target_name.endswith(".fits"):
        target_name += ".fits"
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.rglob("*.fits"):
            if p.name.lower() == target_name:
                return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# get_crop  (copied from scripts/g1_run.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_crop(img, cy, cx, size):
    """Extract square crop of `size` centred at (cy, cx) with zero-padding."""
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
    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(sub_crop,
                      ((pad_top, pad_bottom), (pad_left, pad_right)),
                      mode="constant", constant_values=0)
    else:
        crop = sub_crop
    return crop, (c0, c1, r0, r1)


# ─────────────────────────────────────────────────────────────────────────────
# Frame loader / sigma-image cache
# ─────────────────────────────────────────────────────────────────────────────
_frame_cache: dict = {}   # stem -> sub_sig  (background-subtracted / std)


def load_frame(frame_name: str):
    """Return sub_sig (sigma image), h, w for a frame. Results cached by stem."""
    stem = Path(frame_name).stem
    if stem in _frame_cache:
        return _frame_cache[stem]

    fits_path = find_fits_file(stem)
    if fits_path is None:
        raise FileNotFoundError(f"FITS file not found for frame: {frame_name}")

    data, _ = load_fits(fits_path)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, float(std))
    sub_sig = sub / np.clip(std_map, 1e-6, None)

    h, w = data.shape
    result = (sub_sig, h, w)

    if len(_frame_cache) > 6:
        _frame_cache.pop(next(iter(_frame_cache)))
    _frame_cache[stem] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single candidate  –  cutout + context images
# ─────────────────────────────────────────────────────────────────────────────
def render_single(row: pd.Series, out_dir: Path, frame_stem: str):
    """
    Produce cutout and context PNGs for one single candidate.
    Always renders fresh from the FITS sigma image (never reuses g1/calls).
    Blind: no score, p, label, or box overlay on cutout.

    Returns (cutout_rel, context_rel) as strings relative to BASE_DIR.
    """
    cand_id_raw = row["candidate_id"]
    try:
        cand_id_int = int(float(cand_id_raw))
    except (ValueError, TypeError):
        cand_id_int = str(cand_id_raw).strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    cutout_fname  = f"{frame_stem}_{cand_id_int}_cutout.png"
    context_fname = f"{frame_stem}_{cand_id_int}_context.png"
    cutout_path   = out_dir / cutout_fname
    context_path  = out_dir / context_fname

    # ── render from FITS ─────────────────────────────────────────────────────
    sub_sig, h_f, w_f = load_frame(row["frame_name"])

    r0 = float(row["bbox_r0"])
    c0 = float(row["bbox_c0"])
    r1 = float(row["bbox_r1"])
    c1 = float(row["bbox_c1"])
    cy = (r0 + r1) / 2.0
    cx = (c0 + c1) / 2.0

    # Cutout: 600x600 crop centered on box centre, same stretch (-1 to +5), no overlay
    crop, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub_sig, cy, cx, 600)

    fig_c, ax_c = plt.subplots(figsize=(6, 6))
    ax_c.imshow(crop, cmap="gray", origin="lower",
                extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=-1.0, vmax=5.0)
    ax_c.set_xlim(c0_cr, c1_cr)
    ax_c.set_ylim(r0_cr, r1_cr)
    ax_c.axis("off")
    fig_c.savefig(str(cutout_path), bbox_inches="tight", dpi=100)
    plt.close(fig_c)

    # Context: whole-frame thumbnail with thin box and both thin semi-transparent diagonals
    scale_ds = max(1, int(np.ceil(max(h_f, w_f) / 600.0)))
    h_ds = h_f // scale_ds
    w_ds = w_f // scale_ds
    frame_ds = (
        sub_sig[: h_ds * scale_ds, : w_ds * scale_ds]
        .reshape(h_ds, scale_ds, w_ds, scale_ds)
        .mean(axis=(1, 3))
    )
    vmin_a, vmax_a = np.percentile(frame_ds, [5, 99.5])

    fig_x, ax_x = plt.subplots(figsize=(6, 6))
    ax_x.imshow(frame_ds, cmap="gray", origin="lower",
                extent=[0, w_f, 0, h_f], vmin=vmin_a, vmax=vmax_a)
    rect_x = patches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0, linewidth=0.8, edgecolor="red", facecolor="none"
    )
    ax_x.add_patch(rect_x)
    # Both box diagonals, thin and semi-transparent
    ax_x.plot([c0, c1], [r0, r1], color="red", linewidth=0.8, alpha=0.5)
    ax_x.plot([c0, c1], [r1, r0], color="red", linewidth=0.8, alpha=0.5)
    ax_x.set_xlim(0, w_f)
    ax_x.set_ylim(0, h_f)
    ax_x.axis("off")
    fig_x.savefig(str(context_path), bbox_inches="tight", dpi=100)
    plt.close(fig_x)

    return (
        str(cutout_path.relative_to(BASE_DIR)),
        str(context_path.relative_to(BASE_DIR)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Line candidate  –  4-panel strip (cutout) + context image
# (strip code copied from scripts/make_line_refs.py)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' string to two (row, col) tuples, or None."""
    if not ep_str or (isinstance(ep_str, float) and np.isnan(ep_str)):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def _fit_line_from_endpoints(ep_a, ep_b):
    """Return (mean_y, mean_x, uy, ux, t_min, t_max) from two endpoint tuples."""
    r0, c0 = ep_a
    r1, c1 = ep_b
    mean_y = (r0 + r1) / 2.0
    mean_x = (c0 + c1) / 2.0
    dy = r1 - r0
    dx = c1 - c0
    length = np.hypot(dy, dx)
    uy = dy / length if length > 1e-6 else 1.0
    ux = dx / length if length > 1e-6 else 0.0
    t0 = (r0 - mean_y) * uy + (c0 - mean_x) * ux
    t1 = (r1 - mean_y) * uy + (c1 - mean_x) * ux
    return mean_y, mean_x, uy, ux, min(t0, t1), max(t0, t1)


def _get_boundary_intersections(mean_y, mean_x, uy, ux, h, w):
    """Intersect infinite line with image borders."""
    ts = []
    if abs(uy) > 1e-7:
        for y_b, label in [(0.0, "bottom"), (float(h), "top")]:
            t = (y_b - mean_y) / uy
            x = mean_x + t * ux
            if 0.0 <= x <= w:
                ts.append((t, y_b, x, label))
    if abs(ux) > 1e-7:
        for x_b, label in [(0.0, "left"), (float(w), "right")]:
            t = (x_b - mean_x) / ux
            y = mean_y + t * uy
            if 0.0 <= y <= h:
                ts.append((t, y, x_b, label))
    unique_ts = []
    for item in sorted(ts, key=lambda x: x[0]):
        if not unique_ts or abs(item[0] - unique_ts[-1][0]) > 1.0:
            unique_ts.append(item)
    if len(unique_ts) >= 2:
        return unique_ts[0], unique_ts[-1]
    return None, None


def render_line(row: pd.Series, out_dir: Path, frame_stem: str):
    """
    Produce 4-panel strip cutout and whole-frame context for one line candidate.
    Returns (cutout_rel, context_rel) as strings relative to BASE_DIR.
    """
    line_id_val = row.get("line_id", "")
    if pd.isna(line_id_val) or str(line_id_val).strip() == "":
        line_id_val = str(row["candidate_id"])
    line_id = str(line_id_val).strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    cutout_path  = out_dir / f"{line_id}_strip.png"
    context_path = out_dir / f"{line_id}_context.png"

    sub_sig, h_f, w_f = load_frame(row["frame_name"])

    # Geometry from endpoints column or bbox diagonal
    ep_str = row.get("endpoints", None)
    eps = _parse_endpoints(ep_str) if ep_str is not None else None
    if eps is not None:
        ep_a, ep_b = eps
    else:
        r0 = float(row["bbox_r0"]); c0_b = float(row["bbox_c0"])
        r1 = float(row["bbox_r1"]); c1_b = float(row["bbox_c1"])
        ep_a, ep_b = (r0, c0_b), (r1, c1_b)

    mean_y, mean_x, uy, ux, t_min, t_max = _fit_line_from_endpoints(ep_a, ep_b)

    pt_b1, pt_b2 = _get_boundary_intersections(mean_y, mean_x, uy, ux, h_f, w_f)
    if pt_b1 is not None:
        t_b1, t_b2 = pt_b1[0], pt_b2[0]
    else:
        t_b1 = t_min - 300.0
        t_b2 = t_max + 300.0

    t_strip_start = max(t_min - 300.0, t_b1)
    t_strip_end   = min(t_max + 300.0, t_b2)
    if t_strip_end <= t_strip_start:
        t_strip_start = t_min - 100.0
        t_strip_end   = t_max + 100.0

    t_vals = np.arange(t_strip_start, t_strip_end + 1.0, 1.0)
    w_vals = np.arange(-60.0, 61.0, 1.0)
    T_grid, W_grid = np.meshgrid(t_vals, w_vals)

    # Real strip
    Y_real = mean_y + T_grid * uy - W_grid * ux
    X_real = mean_x + T_grid * ux + W_grid * uy
    strip_real_raw  = map_coordinates(sub_sig,
                                      [Y_real.ravel(), X_real.ravel()],
                                      order=1, mode="constant", cval=0.0
                                      ).reshape(Y_real.shape)
    strip_real_blur = gaussian_filter(strip_real_raw, sigma=1.5)

    vmin_strip = float(np.percentile(strip_real_blur, 5.0))
    vmax_strip = float(np.percentile(strip_real_blur, 99.5))

    n_len = len(t_vals)
    bin_f = max(1, int(np.ceil(n_len / 2000.0)))
    n_trim = (n_len // bin_f) * bin_f
    strip_real_disp = (
        strip_real_blur[:, :n_trim].reshape(121, n_trim // bin_f, bin_f).mean(axis=2)
        if bin_f > 1 else strip_real_blur
    )
    col_inlier_start = (t_min - t_strip_start) / float(bin_f)
    col_inlier_end   = (t_max - t_strip_start) / float(bin_f)

    # Control strip (+150 px perpendicular)
    W_ctrl_grid = W_grid + 150.0
    Y_ctrl = mean_y + T_grid * uy - W_ctrl_grid * ux
    X_ctrl = mean_x + T_grid * ux + W_ctrl_grid * uy
    strip_ctrl_raw  = map_coordinates(sub_sig,
                                      [Y_ctrl.ravel(), X_ctrl.ravel()],
                                      order=1, mode="constant", cval=0.0
                                      ).reshape(Y_ctrl.shape)
    strip_ctrl_blur = gaussian_filter(strip_ctrl_raw, sigma=1.5)
    strip_ctrl_disp = (
        strip_ctrl_blur[:, :n_trim].reshape(121, n_trim // bin_f, bin_f).mean(axis=2)
        if bin_f > 1 else strip_ctrl_blur
    )

    prof_real = np.mean(strip_real_raw, axis=1)
    prof_ctrl = np.mean(strip_ctrl_raw, axis=1)

    # Overview frame
    scale_ds = max(1, int(np.ceil(max(h_f, w_f) / 1000.0)))
    h_ds = h_f // scale_ds
    w_ds = w_f // scale_ds
    frame_ds = (
        sub_sig[: h_ds * scale_ds, : w_ds * scale_ds]
        .reshape(h_ds, scale_ds, w_ds, scale_ds)
        .mean(axis=(1, 3))
    )
    vmin_frame = float(np.percentile(frame_ds, 1.0))
    vmax_frame = float(np.percentile(frame_ds, 99.5))

    # 4-panel figure
    fig = plt.figure(figsize=(15, 12), dpi=140)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[4.5, 4.5, 1.3, 1.3])
    fig.suptitle(f"{frame_stem} | Line {line_id}", fontsize=13, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.imshow(frame_ds, cmap="gray", origin="lower",
               extent=[0, w_f, 0, h_f], vmin=vmin_frame, vmax=vmax_frame)
    ax1.set_title("1. Plain Frame (line endpoints)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, w_f)
    ax1.set_ylim(0, h_f)
    ax1.set_xlabel("X (px)", fontsize=9)
    ax1.set_ylabel("Y (px)", fontsize=9)
    tick_len = 50.0
    if pt_b1 is not None and pt_b2 is not None:
        for pt in [pt_b1, pt_b2]:
            _, y_b, x_b, border = pt
            if border == "bottom":
                ax1.plot([x_b, x_b], [0, tick_len], color="red", linewidth=2.5)
            elif border == "top":
                ax1.plot([x_b, x_b], [h_f - tick_len, h_f], color="red", linewidth=2.5)
            elif border == "left":
                ax1.plot([0, tick_len], [y_b, y_b], color="red", linewidth=2.5)
            elif border == "right":
                ax1.plot([w_f - tick_len, w_f], [y_b, y_b], color="red", linewidth=2.5)
    else:
        ax1.plot([ep_a[1], ep_b[1]], [ep_a[0], ep_b[0]], color="red", linewidth=1.5)

    ax4 = fig.add_subplot(gs[0:2, 1])
    ax4.plot(w_vals, prof_real, color="cyan", linewidth=2.0, label="Real strip")
    ax4.plot(w_vals, prof_ctrl, color="orange", linewidth=1.8, linestyle="--", label="Control (+150px)")
    ax4.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax4.set_title("4. Perpendicular Profile", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Perpendicular offset (px)", fontsize=9)
    ax4.set_ylabel("Mean brightness (sigma)", fontsize=9)
    ax4.set_xlim(-60, 60)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    ax2 = fig.add_subplot(gs[2, :])
    ax2.imshow(strip_real_disp, cmap="gray", origin="lower",
               extent=[0, strip_real_disp.shape[1], -60, 60],
               vmin=vmin_strip, vmax=vmax_strip, aspect="auto")
    ax2.set_title("2. Straightened Strip (±60px, σ=1.5, binned)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Offset (px)", fontsize=8)
    ax2.set_xlabel("Length along line (px)", fontsize=8)
    ax2.plot([col_inlier_start, col_inlier_start], [45, 60], color="red", linewidth=2.5)
    ax2.plot([col_inlier_end, col_inlier_end], [45, 60], color="red", linewidth=2.5)

    ax3 = fig.add_subplot(gs[3, :])
    ax3.imshow(strip_ctrl_disp, cmap="gray", origin="lower",
               extent=[0, strip_ctrl_disp.shape[1], -60, 60],
               vmin=vmin_strip, vmax=vmax_strip, aspect="auto")
    ax3.set_title("3. Control Strip (+150px perpendicular)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Offset (px)", fontsize=8)
    ax3.set_xlabel("Length along line (px)", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(cutout_path))
    plt.close(fig)

    # Context: whole-frame thumbnail at ~600 px with line drawn
    scale_ctx = max(1, int(np.ceil(max(h_f, w_f) / 600.0)))
    h_ctx = h_f // scale_ctx
    w_ctx = w_f // scale_ctx
    frame_ctx = (
        sub_sig[: h_ctx * scale_ctx, : w_ctx * scale_ctx]
        .reshape(h_ctx, scale_ctx, w_ctx, scale_ctx)
        .mean(axis=(1, 3))
    )
    vmin_ctx, vmax_ctx = np.percentile(frame_ctx, [5, 99.5])

    fig2, ax2x = plt.subplots(figsize=(6, 6))
    ax2x.imshow(frame_ctx, cmap="gray", origin="lower",
                extent=[0, w_f, 0, h_f], vmin=vmin_ctx, vmax=vmax_ctx)
    ax2x.plot([ep_a[1], ep_b[1]], [ep_a[0], ep_b[0]], color="red", linewidth=1.5, alpha=0.9)
    ax2x.set_xlim(0, w_f)
    ax2x.set_ylim(0, h_f)
    ax2x.axis("off")
    fig2.savefig(str(context_path), bbox_inches="tight", dpi=100)
    plt.close(fig2)

    return (
        str(cutout_path.relative_to(BASE_DIR)),
        str(context_path.relative_to(BASE_DIR)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Build label_tool.py batch from candidates_v2.csv"
    )
    p.add_argument("--streak-singles", action="store_true",
                   help="Include all singles with verdict == streak")
    p.add_argument("--lines", action="store_true",
                   help="Include all rows with type == line")
    p.add_argument("--frame", metavar="NAME",
                   help="Include all candidates in this frame (with or without .fits)")
    p.add_argument("--name", required=True, metavar="TEXT",
                   help="Name used in output file names")
    return p.parse_args()


def _ensure_fits(s):
    s = str(s).strip()
    return s if s.lower().endswith(".fits") else s + ".fits"


def main():
    args = parse_args()

    if not args.streak_singles and not args.lines and not args.frame:
        print("Error: specify at least one of --streak-singles, --lines, --frame")
        sys.exit(1)

    df = pd.read_csv(CANDIDATES_CSV)
    df["frame_name"] = df["frame_name"].apply(_ensure_fits)

    masks = []
    if args.streak_singles:
        masks.append(
            (df["type"].str.lower() == "single") &
            (df["verdict"].str.lower() == "streak")
        )
    if args.lines:
        masks.append(df["type"].str.lower() == "line")
    if args.frame:
        frame_norm = _ensure_fits(args.frame).lower()
        masks.append(df["frame_name"].str.lower() == frame_norm)

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    subset = df[combined].copy().reset_index(drop=True)

    if len(subset) == 0:
        print("No rows matched. Nothing to do.")
        sys.exit(0)

    print(f"Selected {len(subset)} rows for batch '{args.name}'.")

    img_out_dir = BASE_DIR / "outputs" / "review_v2" / args.name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LABELED_DIR / f"review_v2_{args.name}_labels.csv"

    output_rows = []
    for _, row in subset.iterrows():
        frame_name = row["frame_name"]
        frame_stem = Path(frame_name).stem
        row_type   = str(row.get("type", "single")).lower()

        if row_type == "line":
            line_id_val = row.get("line_id", "")
            if pd.isna(line_id_val) or str(line_id_val).strip() == "":
                line_id_val = row["candidate_id"]
            cand_id = str(line_id_val).strip()
        else:
            cand_id_raw = row["candidate_id"]
            try:
                cand_id = str(int(float(cand_id_raw)))
            except (ValueError, TypeError):
                cand_id = str(cand_id_raw).strip()

        try:
            if row_type == "line":
                cutout_rel, context_rel = render_line(row, img_out_dir, frame_stem)
            else:
                cutout_rel, context_rel = render_single(row, img_out_dir, frame_stem)
            print(f"  OK  {frame_stem}  {cand_id}")
        except Exception as exc:
            print(f"  WARN {frame_stem} {cand_id}: {exc}")
            cutout_rel  = ""
            context_rel = ""

        output_rows.append({
            "candidate_id":  cand_id,
            "frame_name":    frame_name,
            "cutout_path":   cutout_rel,
            "context_path":  context_rel,
            "score":         "",
            "area":          "",
            "aspect_ratio":  "",
            "label":         "",
            "notes":         "",
        })

    out_df = pd.DataFrame(output_rows, columns=[
        "candidate_id", "frame_name", "cutout_path", "context_path",
        "score", "area", "aspect_ratio", "label", "notes",
    ])
    out_df.to_csv(str(out_csv), index=False, na_rep="")

    print(f"\nCSV written : {out_csv}")
    print(f"Images in   : {img_out_dir}")
    print(f"\nRun the review with:")
    print(f"  python scripts/label_tool.py {out_csv.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
