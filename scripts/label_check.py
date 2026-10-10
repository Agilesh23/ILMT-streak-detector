# TASK G9g + G9h: interactive review tool
"""
Interactive review tool for ILMT streak detection candidates and frames.

Notes are typed directly inside the plot window: press 'n' to enter note mode,
type text, Enter to save, Esc to cancel.

Usage Examples:
  1. Review all single candidates labeled as streak:
     python scripts/label_check.py --streak-singles

  2. Review all accepted lines:
     python scripts/label_check.py --lines

  3. Review all candidates in a specific frame:
     python scripts/label_check.py --frame 20221026-0059

  4. Review candidates from an external audit CSV:
     python scripts/label_check.py --from-csv data/d0b/audit_streak_singles.csv

  5. Whole-frame level inspection without candidate zooms:
     python scripts/label_check.py --frames 20221025-0028 20221026-0057 20230519-0009
"""

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import sys

import matplotlib
from matplotlib.colors import AsinhNorm, LogNorm, Normalize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.io_utils import load_fits


def clear_matplotlib_keymaps():
    """Clear default matplotlib shortcut keymaps that conflict with review keys."""
    keys_to_clear = [
        "keymap.save",
        "keymap.fullscreen",
        "keymap.quit",
        "keymap.quit_all",
        "keymap.back",
        "keymap.forward",
        "keymap.pan",
        "keymap.zoom",
        "keymap.grid",
        "keymap.yscale",
        "keymap.xscale",
        "keymap.home",
        "keymap.help",
    ]
    cleared = []
    for k in keys_to_clear:
        if k in plt.rcParams:
            plt.rcParams[k] = []
            cleared.append(k)
    return cleared


def parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' into ((r0, c0), (r1, c1)). Copied from scripts/viz_check.py."""
    if not ep_str or pd.isna(ep_str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def get_crop(img, cy, cx, size):
    """
    Extract a square crop of given size centered at (cy, cx) with constant 0 padding.
    Directly copied from scripts/viz_check.py.
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
    Directly copied from scripts/viz_check.py.
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
                return p
    return None


def get_stretch_norm(data, stretch_type="asinh", vmin_pct=1.0, vmax_pct=99.8, linear_width=3.0):
    """Compute normalization stretch object. Directly copied from scripts/viz_check.py."""
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
        try:
            return AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax), vmin, vmax
        except Exception:
            return None, vmin, vmax


def load_label_overrides():
    """Load data/d0b/label_overrides.csv if it exists. Directly copied from scripts/viz_check.py."""
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


class ReviewSession:
    def __init__(self, items, source_desc, redo=False):
        self.items = items
        self.source_desc = source_desc
        self.redo = redo
        self.out_csv_path = base_dir / "data/d0b/visual_review_v2.csv"
        self.out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.overrides = load_label_overrides()
        self.history = self.load_existing_reviews()

        # In-window note mode state
        self.note_mode = False
        self.note_buffer = ""
        self.note_text_artist = None

        # Filter items if not redo: items with a blank verdict count as not reviewed (not skipped)
        self.active_items = []
        self.skipped_count = 0
        for item in self.items:
            key = (item["frame_name"], str(item["candidate_id"]), item["kind"])
            has_verdict = (key in self.history and bool(str(self.history[key].get("verdict", "")).strip()))
            if not self.redo and has_verdict:
                self.skipped_count += 1
            else:
                # Attach earlier review / note if present
                if key in self.history:
                    item["verdict"] = self.history[key].get("verdict", "")
                    item["note"] = self.history[key].get("note", "")
                self.active_items.append(item)

        self.current_idx = 0
        self.show_overlay = True
        self.fits_cache = {}

        # Tracking counts
        self.stats = {"streak": 0, "fp": 0, "unsure": 0}

    def load_existing_reviews(self):
        history = {}
        if self.out_csv_path.exists():
            try:
                df = pd.read_csv(self.out_csv_path)
                for _, r in df.iterrows():
                    key = (str(r["frame_name"]), str(r["candidate_id"]), str(r["kind"]))
                    raw_verdict = str(r.get("verdict", "")).strip() if pd.notna(r.get("verdict")) else ""
                    if raw_verdict.lower() in ("nan", "none"):
                        raw_verdict = ""
                    raw_note = str(r.get("note", "")).strip() if pd.notna(r.get("note")) else ""
                    if raw_note.lower() in ("nan", "none"):
                        raw_note = ""
                    history[key] = {
                        "verdict": raw_verdict,
                        "note": raw_note,
                        "reviewed_at": str(r.get("reviewed_at", "")) if pd.notna(r.get("reviewed_at")) else "",
                        "source": str(r.get("source", "")) if pd.notna(r.get("source")) else "",
                    }
            except Exception as e:
                print(f"Warning: Could not read existing review CSV: {e}")
        return history

    def save_review_state(self):
        """Write the entire visual_review_v2.csv file."""
        records = []
        for (fn, cid, kind), rdata in self.history.items():
            records.append({
                "frame_name": fn,
                "candidate_id": cid,
                "kind": kind,
                "verdict": rdata["verdict"],
                "note": rdata["note"],
                "reviewed_at": rdata["reviewed_at"],
                "source": rdata["source"],
            })

        df_out = pd.DataFrame(records, columns=[
            "frame_name", "candidate_id", "kind", "verdict", "note", "reviewed_at", "source"
        ])
        df_out.to_csv(self.out_csv_path, index=False)

    def record_verdict(self, verdict):
        if not self.active_items:
            return
        item = self.active_items[self.current_idx]
        item["verdict"] = verdict
        item["reviewed_at"] = datetime.now().isoformat()
        item["source"] = self.source_desc

        key = (item["frame_name"], str(item["candidate_id"]), item["kind"])
        self.history[key] = {
            "verdict": verdict,
            "note": item.get("note", ""),
            "reviewed_at": item["reviewed_at"],
            "source": item["source"],
        }
        self.save_review_state()
        print(f"Recorded [{verdict.upper()}] for {item['frame_name']} cand {item['candidate_id']}")

        # Advance to next item
        self.advance_next(skip_judged=False)

    def advance_next(self, skip_judged=False):
        if self.current_idx < len(self.active_items) - 1:
            if skip_judged:
                # Seek next without a verdict
                found = False
                for j in range(self.current_idx + 1, len(self.active_items)):
                    if not self.active_items[j].get("verdict"):
                        self.current_idx = j
                        found = True
                        break
                if not found:
                    self.current_idx = len(self.active_items) - 1
            else:
                self.current_idx += 1
            self.render()
        else:
            print("Reached the last item. Press 'q' to save and exit.")
            self.render()

    def go_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.render()

    def enter_note_mode(self):
        if not self.active_items:
            return
        item = self.active_items[self.current_idx]
        self.note_mode = True
        curr_note = item.get("note", "")
        self.note_buffer = str(curr_note) if curr_note and not pd.isna(curr_note) else ""
        self.render()

    def update_note_text(self):
        if hasattr(self, "note_text_artist") and self.note_text_artist is not None:
            self.note_text_artist.set_text(f"Note: {self.note_buffer}_")
            self.fig.canvas.draw_idle()
        else:
            self.render()

    def load_frame_data(self, frame_name):
        f_stem = Path(frame_name).stem.lower()
        if f_stem in self.fits_cache:
            return self.fits_cache[f_stem]

        p = find_fits_file(f_stem)
        if not p or not p.exists():
            return None

        data, _ = load_fits(p)
        bg, _ = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = sub.shape

        scale_ds = max(1, int(math.ceil(max(h_img, w_img) / 1000.0)))
        h_ds = h_img // scale_ds
        w_ds = w_img // scale_ds
        frame_ds = sub[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
        vmin_ov = float(np.percentile(frame_ds, 1.0))
        vmax_ov = float(np.percentile(frame_ds, 99.5))

        cached = {
            "sub": sub,
            "frame_ds": frame_ds,
            "shape": (h_img, w_img),
            "vmin_ov": vmin_ov,
            "vmax_ov": vmax_ov,
        }
        self.fits_cache[f_stem] = cached
        return cached

    def setup_figure(self):
        clear_matplotlib_keymaps()
        self.fig = plt.figure(figsize=(15, 9), dpi=110)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def render(self):
        if not self.active_items:
            print("No items to review.")
            return

        self.fig.clear()
        item = self.active_items[self.current_idx]
        kind = item["kind"]
        frame_name = item["frame_name"]
        f_stem = Path(frame_name).stem
        cid = item["candidate_id"]

        # Label override check for display
        orig_lbl = item.get("current_label", "unlabeled")
        cid_str = str(int(cid)) if str(cid).replace('.','',1).isdigit() else str(cid)
        ov_key = (f_stem.lower(), cid_str)
        if ov_key in self.overrides:
            disp_label = f"{self.overrides[ov_key]}*"
        else:
            disp_label = str(orig_lbl)

        v_val = item.get("verdict", "")
        verdict_str = v_val.upper() if v_val else "NONE"
        note_str = item.get("note", "")
        note_disp = f" | Note: '{note_str}'" if note_str else ""

        progress_str = f"[{self.current_idx + 1}/{len(self.active_items)}]"
        title_main = (
            f"{progress_str} {frame_name} | ID: {cid} | Kind: {kind} | Label: {disp_label} | "
            f"Verdict: [{verdict_str}]{note_disp}"
        )
        if self.note_mode:
            sub_title = "NOTE MODE (Enter = save, Esc = cancel)"
            self.fig.suptitle(f"{title_main}\n{sub_title}", fontsize=11, fontweight="bold", y=0.98, color="darkred")
        else:
            sub_help = "Keys: [s] Streak  [f] False Positive  [u] Unsure  [o] Toggle Overlay  [n] Note  [←/→] Prev/Next  [q] Save & Quit"
            self.fig.suptitle(f"{title_main}\n{sub_help}", fontsize=11, fontweight="bold", y=0.98)

        # Load frame
        fdata = self.load_frame_data(frame_name)
        if fdata is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"FITS file for {frame_name} not found in search paths.", ha="center", va="center", color="red", fontsize=14)
            self.fig.canvas.draw_idle()
            return

        sub = fdata["sub"]
        frame_ds = fdata["frame_ds"]
        h_img, w_img = fdata["shape"]
        vmin_ov = fdata["vmin_ov"]
        vmax_ov = fdata["vmax_ov"]

        if kind == "frame":
            # Overview only
            ax_ov = self.fig.add_subplot(111)
            ax_ov.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin_ov, vmax=vmax_ov)
            ax_ov.set_title(f"Whole Frame Overview: {frame_name}", fontsize=10)
            ax_ov.set_xlabel("Column (px)")
            ax_ov.set_ylabel("Row (px)")
        else:
            # Gridspec: Left overview (large), Right 3 zoom panels
            gs = self.fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0], hspace=0.28, wspace=0.15)
            ax_ov = self.fig.add_subplot(gs[:, 0])
            ax_ov.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin_ov, vmax=vmax_ov)
            ax_ov.set_xlim(0, w_img)
            ax_ov.set_ylim(0, h_img)
            ax_ov.set_xlabel("Column (px)")
            ax_ov.set_ylabel("Row (px)")
            ax_ov.set_title("Full Frame Overview", fontsize=10)

            # Draw overlay on overview if enabled
            if self.show_overlay:
                if kind == "line":
                    eps = item.get("endpoints")
                    if eps is not None:
                        ax_ov.plot([eps[0][1], eps[1][1]], [eps[0][0], eps[1][0]], color="cyan", lw=1.2, alpha=0.7)
                elif kind == "single":
                    r0, r1 = item.get("r0", 0), item.get("r1", 0)
                    c0, c1 = item.get("c0", 0), item.get("c1", 0)
                    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0, lw=1.0, edgecolor="yellow", facecolor="none", alpha=0.7)
                    ax_ov.add_patch(rect)
                    # Both diagonals
                    ax_ov.plot([c0, c1], [r0, r1], color="orange", lw=0.8, alpha=0.6, linestyle=":")
                    ax_ov.plot([c0, c1], [r1, r0], color="orange", lw=0.8, alpha=0.6, linestyle=":")

            # Right 3 Zoom Panels (300x300, clean)
            zoom_axes = [self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[1, 1]), self.fig.add_subplot(gs[2, 1])]

            if kind == "line":
                eps = item.get("endpoints")
                if eps is not None:
                    (ep0_r, ep0_c), (ep1_r, ep1_c) = eps
                    dr = ep1_r - ep0_r
                    dc = ep1_c - ep0_c
                    fractions = [(0.25, "25% Length"), (0.50, "50% Length"), (0.75, "75% Length")]

                    for z_i, (frac, label_t) in enumerate(fractions):
                        zr = ep0_r + frac * dr
                        zc = ep0_c + frac * dc
                        crop, req_b, _ = get_crop(sub, zr, zc, 300)
                        norm_z, vmin_z, vmax_z = get_stretch_norm(crop, stretch_type="asinh")
                        ax_z = zoom_axes[z_i]
                        if norm_z is not None:
                            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_z)
                        else:
                            ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_z, vmax=vmax_z)
                        ax_z.set_xlim(req_b[0], req_b[1])
                        ax_z.set_ylim(req_b[2], req_b[3])
                        ax_z.set_title(f"Clean Zoom @ {label_t} (r={zr:.1f}, c={zc:.1f})", fontsize=8.5)
            elif kind == "single":
                r0, r1 = item.get("r0", 0), item.get("r1", 0)
                c0, c1 = item.get("c0", 0), item.get("c1", 0)
                cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0

                # 3 points: box center, middle of diagonal 1, middle of diagonal 2
                # If box is large, diagonal points are sampled halfway between center and corners
                dr_half = (r1 - r0) * 0.25
                dc_half = (c1 - c0) * 0.25
                targets = [
                    (cy, cx, "Box Centre"),
                    (cy - dr_half, cx - dc_half, "Middle of Diagonal 1 (main)"),
                    (cy + dr_half, cx - dc_half, "Middle of Diagonal 2 (anti)"),
                ]

                for z_i, (zr, zc, label_t) in enumerate(targets):
                    crop, req_b, _ = get_crop(sub, zr, zc, 300)
                    norm_z, vmin_z, vmax_z = get_stretch_norm(crop, stretch_type="asinh")
                    ax_z = zoom_axes[z_i]
                    if norm_z is not None:
                        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], norm=norm_z)
                    else:
                        ax_z.imshow(crop, cmap="gray", origin="lower", extent=[req_b[0], req_b[1], req_b[2], req_b[3]], vmin=vmin_z, vmax=vmax_z)
                    ax_z.set_xlim(req_b[0], req_b[1])
                    ax_z.set_ylim(req_b[2], req_b[3])
                    ax_z.set_title(f"Clean Zoom: {label_t}", fontsize=8.5)

        if self.note_mode:
            self.note_text_artist = self.fig.text(
                0.05,
                0.015,
                f"Note: {self.note_buffer}_",
                fontsize=11,
                fontweight="bold",
                color="darkblue",
                family="monospace",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="#fff9d6", edgecolor="#c49000", lw=1.0),
            )
        else:
            self.note_text_artist = None

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key
        if not k:
            return

        # Note mode keystroke handling
        if self.note_mode:
            if k in ("enter", "return"):
                if self.active_items:
                    item = self.active_items[self.current_idx]
                    item["note"] = self.note_buffer
                    key = (item["frame_name"], str(item["candidate_id"]), item["kind"])
                    current_verdict = item.get("verdict", "")
                    reviewed_at = datetime.now().isoformat()
                    if key in self.history:
                        self.history[key]["note"] = self.note_buffer
                        self.history[key]["verdict"] = current_verdict
                        self.history[key]["reviewed_at"] = reviewed_at
                    else:
                        self.history[key] = {
                            "verdict": current_verdict,
                            "note": self.note_buffer,
                            "reviewed_at": reviewed_at,
                            "source": self.source_desc,
                        }
                    self.save_review_state()
                    print(f"Saved note for {item['frame_name']} cand {item['candidate_id']}: '{self.note_buffer}'")
                self.note_mode = False
                self.render()
                return
            elif k == "escape":
                print("Note editing cancelled.")
                self.note_mode = False
                self.render()
                return
            elif k == "backspace":
                if len(self.note_buffer) > 0:
                    self.note_buffer = self.note_buffer[:-1]
                    self.update_note_text()
                return
            elif k in ("space", " "):
                self.note_buffer += " "
                self.update_note_text()
                return
            elif len(k) == 1 and k.isprintable():
                self.note_buffer += k
                self.update_note_text()
                return
            elif k.startswith("shift+") and len(k) == 7:
                char = k[6:]
                if char.isalpha():
                    self.note_buffer += char.upper()
                else:
                    self.note_buffer += char
                self.update_note_text()
                return
            else:
                # Commands and arrow keys are ignored in note mode
                return

        # Normal command mode keys
        if k == "s":
            self.record_verdict("streak")
        elif k == "f":
            self.record_verdict("fp")
        elif k == "u":
            self.record_verdict("unsure")
        elif k == "n":
            self.enter_note_mode()
        elif k == "o":
            self.show_overlay = not self.show_overlay
            self.render()
        elif k == "right":
            self.advance_next(skip_judged=True)
        elif k == "left":
            self.go_prev()
        elif k == "q":
            print("\nQuit requested. Saving and exiting review.")
            self.save_review_state()
            plt.close(self.fig)

    def on_close(self, event):
        if self.note_mode:
            print("\nWindow closed while in note mode. Discarding unfinished note.")
            self.note_mode = False
        print("\nWindow closed. Saving review session.")
        self.save_review_state()


def build_review_items(args):
    """Parse CLI modes and compile structured list of items to review."""
    items = []
    cands_path = base_dir / "data/d0b/candidates_v2.csv"
    df_cands = pd.read_csv(cands_path) if cands_path.exists() else pd.DataFrame()

    if args.lines:
        source_desc = "--lines"
        if df_cands.empty:
            print("Error: candidates_v2.csv not found.")
            return items, source_desc
        line_rows = df_cands[df_cands["type"].astype(str).str.lower() == "line"]
        for _, r in line_rows.iterrows():
            items.append({
                "frame_name": str(r["frame_name"]),
                "candidate_id": str(r["cand_v2_id"]),
                "kind": "line",
                "current_label": str(r.get("verdict", "unlabeled")),
                "endpoints": parse_endpoints(r.get("endpoints")),
                "length_px": float(r.get("length_px", 0.0)),
            })

    elif args.streak_singles:
        source_desc = "--streak-singles"
        if df_cands.empty:
            print("Error: candidates_v2.csv not found.")
            return items, source_desc
        mask = (df_cands["type"].astype(str).str.lower() != "line") & (df_cands["verdict"].astype(str).str.lower() == "streak")
        single_rows = df_cands[mask]
        for _, r in single_rows.iterrows():
            cy = float(r["centroid_row"])
            cx = float(r["centroid_col"])
            r0 = float(r["bbox_r0"]) if pd.notna(r.get("bbox_r0")) else cy - 10
            r1 = float(r["bbox_r1"]) if pd.notna(r.get("bbox_r1")) else cy + 10
            c0 = float(r["bbox_c0"]) if pd.notna(r.get("bbox_c0")) else cx - 10
            c1 = float(r["bbox_c1"]) if pd.notna(r.get("bbox_c1")) else cx + 10
            items.append({
                "frame_name": str(r["frame_name"]),
                "candidate_id": str(int(r["candidate_id"])) if pd.notna(r.get("candidate_id")) else str(r["cand_v2_id"]),
                "kind": "single",
                "current_label": str(r.get("verdict", "streak")),
                "cy": cy, "cx": cx,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "length_px": float(r.get("length_px", 0.0)),
            })

    elif args.frame:
        source_desc = f"--frame {args.frame}"
        f_stem = Path(args.frame).stem
        f_name = f"{f_stem}.fits"
        if df_cands.empty:
            print("Error: candidates_v2.csv not found.")
            return items, source_desc
        mask = (df_cands["frame_name"] == f_name) | (df_cands["frame"] == f_stem)
        frame_rows = df_cands[mask]
        for _, r in frame_rows.iterrows():
            is_line = (str(r.get("type", "")).lower() == "line")
            cy = float(r["centroid_row"])
            cx = float(r["centroid_col"])
            r0 = float(r["bbox_r0"]) if pd.notna(r.get("bbox_r0")) else cy - 10
            r1 = float(r["bbox_r1"]) if pd.notna(r.get("bbox_r1")) else cy + 10
            c0 = float(r["bbox_c0"]) if pd.notna(r.get("bbox_c0")) else cx - 10
            c1 = float(r["bbox_c1"]) if pd.notna(r.get("bbox_c1")) else cx + 10
            items.append({
                "frame_name": f_name,
                "candidate_id": str(r["cand_v2_id"]) if is_line else (str(int(r["candidate_id"])) if pd.notna(r.get("candidate_id")) else str(r["cand_v2_id"])),
                "kind": "line" if is_line else "single",
                "current_label": str(r.get("verdict", "unlabeled")),
                "endpoints": parse_endpoints(r.get("endpoints")) if is_line else None,
                "cy": cy, "cx": cx,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                "length_px": float(r.get("length_px", 0.0)),
            })

    elif args.frames:
        source_desc = f"--frames {' '.join(args.frames)}"
        for f_arg in args.frames:
            f_stem = Path(f_arg).stem
            items.append({
                "frame_name": f"{f_stem}.fits",
                "candidate_id": "frame",
                "kind": "frame",
                "current_label": "frame_view",
            })

    elif args.from_csv:
        source_desc = f"--from-csv {args.from_csv}"
        csv_p = Path(args.from_csv)
        if not csv_p.is_absolute():
            csv_p = base_dir / csv_p
        if not csv_p.exists():
            print(f"Error: {csv_p} not found.")
            return items, source_desc
        df_in = pd.read_csv(csv_p)
        for _, r_in in df_in.iterrows():
            fn = str(r_in["frame_name"])
            cid_str = str(int(r_in["candidate_id"])) if str(r_in["candidate_id"]).replace('.','',1).isdigit() else str(r_in["candidate_id"])
            # Match in candidates_v2.csv if available
            c_match = None
            if not df_cands.empty:
                m = (df_cands["frame_name"] == fn) & (
                    (df_cands["candidate_id"].astype(str) == str(r_in["candidate_id"])) |
                    (df_cands["cand_v2_id"].astype(str) == cid_str)
                )
                if m.any():
                    c_match = df_cands[m].iloc[0]

            if c_match is not None:
                is_line = (str(c_match.get("type", "")).lower() == "line")
                cy = float(c_match["centroid_row"])
                cx = float(c_match["centroid_col"])
                r0 = float(c_match["bbox_r0"]) if pd.notna(c_match.get("bbox_r0")) else cy - 10
                r1 = float(c_match["bbox_r1"]) if pd.notna(c_match.get("bbox_r1")) else cy + 10
                c0 = float(c_match["bbox_c0"]) if pd.notna(c_match.get("bbox_c0")) else cx - 10
                c1 = float(c_match["bbox_c1"]) if pd.notna(c_match.get("bbox_c1")) else cx + 10
                items.append({
                    "frame_name": fn,
                    "candidate_id": cid_str,
                    "kind": "line" if is_line else "single",
                    "current_label": str(c_match.get("verdict", "unlabeled")),
                    "endpoints": parse_endpoints(c_match.get("endpoints")) if is_line else None,
                    "cy": cy, "cx": cx,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                })
            else:
                items.append({
                    "frame_name": fn,
                    "candidate_id": cid_str,
                    "kind": "single",
                    "current_label": "unlabeled",
                    "cy": 2048.0, "cx": 2048.0,
                    "r0": 2038.0, "r1": 2058.0, "c0": 2038.0, "c1": 2058.0,
                })
    else:
        source_desc = "none"

    return items, source_desc


def main():
    parser = argparse.ArgumentParser(description="Interactive Review Tool (TASK G9g)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-csv", type=str, help="Rows with frame_name, candidate_id")
    group.add_argument("--frame", type=str, help="All candidates in that frame from candidates_v2.csv")
    group.add_argument("--lines", action="store_true", help="All accepted lines (kind == line)")
    group.add_argument("--streak-singles", action="store_true", help="All singles whose label is streak")
    group.add_argument("--frames", nargs="+", type=str, help="Frame-level review: no candidates, whole frame only")

    parser.add_argument("--redo", action="store_true", help="Also show items already reviewed (default: skip them)")
    args = parser.parse_args()

    print("=" * 80)
    print("TASK G9g: INTERACTIVE REVIEW TOOL")
    print("=" * 80)

    items, source_desc = build_review_items(args)
    if not items:
        print("No items found for the requested mode.")
        sys.exit(0)

    session = ReviewSession(items, source_desc, redo=args.redo)
    print(f"Total items in mode: {len(items)}")
    print(f"Items already reviewed (skipped): {session.skipped_count}")
    print(f"Items to review in this session:   {len(session.active_items)}")
    print("-" * 80)

    if not session.active_items:
        print("All items in this mode have already been reviewed! Use --redo to review again.")
        # Summary report
        streak_c = sum(1 for v in session.history.values() if v["verdict"] == "streak")
        fp_c = sum(1 for v in session.history.values() if v["verdict"] == "fp")
        unsure_c = sum(1 for v in session.history.values() if v["verdict"] == "unsure")
        print("\nSession Summary:")
        print(f"  Streak verdicts: {streak_c}")
        print(f"  FP verdicts:     {fp_c}")
        print(f"  Unsure verdicts: {unsure_c}")
        print(f"  Items skipped:   {session.skipped_count}")
        print(f"  Items unreached: 0")
        sys.exit(0)

    session.setup_figure()
    session.render()
    plt.show()

    # Final summary calculations
    streak_count = 0
    fp_count = 0
    unsure_count = 0
    unreached_count = 0

    for item in session.active_items:
        v = item.get("verdict")
        if v == "streak":
            streak_count += 1
        elif v == "fp":
            fp_count += 1
        elif v == "unsure":
            unsure_count += 1
        else:
            unreached_count += 1

    print("\n" + "=" * 80)
    print("REVIEW SESSION SUMMARY")
    print("=" * 80)
    print(f"Streak verdicts: {streak_count}")
    print(f"FP verdicts:     {fp_count}")
    print(f"Unsure verdicts: {unsure_count}")
    print(f"Items skipped:   {session.skipped_count}")
    print(f"Items unreached: {unreached_count}")
    print(f"Review database: {session.out_csv_path.relative_to(base_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
