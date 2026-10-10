import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation, gaussian_filter
from skimage.measure import label as sk_label
import matplotlib

if not any(arg == "review" for arg in sys.argv[1:]):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Disable matplotlib default key bindings that clash with our keys
plt.rcParams["keymap.save"] = []
plt.rcParams["keymap.quit"] = []
plt.rcParams["keymap.fullscreen"] = []
plt.rcParams["keymap.yscale"] = []
plt.rcParams["keymap.xscale"] = []

_clash_keys = {"y", "n", "u", "q", "s", "f", "0", "1", "2"}
for _k in list(plt.rcParams.keys()):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = [
            v for v in plt.rcParams[_k]
            if v.lower() not in _clash_keys and not any(v.lower() == f"ctrl+{c}" for c in _clash_keys)
        ]

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
inject_dir = data_dir / "inject"
inject_dir.mkdir(parents=True, exist_ok=True)
blind_dir = inject_dir / "blind"
blind_dir.mkdir(parents=True, exist_ok=True)
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)
models_dir = base_dir / "models"
search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import width_estimation, extract_streak_patch

FRAME_NAMES = ["20221101-0030", "20230615-0006", "20231122-0019"]
LEVELS = [0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0, 6.5]
BLIND_LEVELS = [0.4, 0.6, 1.0, 1.6, 2.5, 4.0]

FEATURE_COLS = [
    "width_ratio",
    "snr_mean",
    "sat_frac",
    "straightness",
    "angle_from_vertical",
    "aspect",
    "edge_dist",
    "length_px",
]


def load_config():
    with open(base_dir / "src/config.yaml", "r") as f:
        return yaml.safe_load(f)


def find_frame_path(stem):
    for p in search_root.rglob("*"):
        if p.is_file() and p.stem.lower() == stem.lower() and p.suffix.lower() == ".fits":
            return p
    return None


def generate_streak_params(rng, h_img, w_img):
    # Angle uniform in [15, 75] or [105, 165] degrees from horizontal
    val = rng.uniform(0.0, 120.0)
    angle_deg = 15.0 + val if val < 60.0 else 105.0 + (val - 60.0)

    # Passes through random point in central 1500x1500 px
    cx = w_img / 2.0
    cy = h_img / 2.0
    x0 = float(rng.uniform(cx - 750.0, cx + 750.0))
    y0 = float(rng.uniform(cy - 750.0, cy + 750.0))
    return angle_deg, (x0, y0)


def inject_streak(data, std_map, level, angle_deg, point, psf_fwhm_px):
    x0, y0 = point
    sigma_cross = psf_fwhm_px / 2.355

    rad = np.radians(angle_deg)
    perp_x = -np.sin(rad)
    perp_y = np.cos(rad)

    h_img, w_img = data.shape
    cols = np.arange(w_img, dtype=np.float64)[None, :]
    rows = np.arange(h_img, dtype=np.float64)[:, None]

    d_perp = (cols - x0) * perp_x + (rows - y0) * perp_y
    mask_near = np.abs(d_perp) <= (5.0 * sigma_cross)

    streak_model = np.zeros_like(data, dtype=np.float64)
    streak_model[mask_near] = level * std_map[mask_near] * np.exp(-0.5 * (d_perp[mask_near] / sigma_cross) ** 2)

    injected_data = data.copy() + streak_model
    return injected_data


def run_pipeline_on_image(data, bg, std, cfg, clf):
    det_cfg = cfg.get("detection", {})
    edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
    border_margin_px = det_cfg.get("border_margin_px", 15)
    border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
    left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
    left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
    min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
    bypass_cfg = det_cfg.get("long_thin_bypass", None)
    bleed_veto_cfg = det_cfg.get("bleed_veto", None)
    psf_fwhm_px = cfg.get("characterization", {}).get("psf_fwhm_px", 6.79)

    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
    sub_sigma = sub / np.clip(std_map, 1e-6, None)
    frame_max = float(np.max(data))
    h_img, w_img = data.shape

    thresh_map = 3.0 * std
    binary = sub > thresh_map
    struct = generate_binary_structure(2, 1)
    binary = binary_dilation(binary, structure=struct, iterations=2)
    labeled_map = sk_label(binary)

    candidates, _, _ = find_blob_candidates(
        data, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass=bypass_cfg,
        bleed_veto=bleed_veto_cfg,
    )
    stitched = stitch_collinear_fragments(candidates, data, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score, drop_bleed_trails=True)

    cands_out = []
    for cand in selected:
        cid = int(cand["label"])
        members = cand.get("stitched_from", [cid])
        cand_mask = np.isin(labeled_map, members)
        coords = np.argwhere(cand_mask)

        if len(coords) < 2:
            cy, cx = cand["centroid"]
            r0, c0, r1, c1 = cand.get("bbox_full", cand.get("bbox_local"))
            coords = np.array([[cy, cx], [cy + 1, cx + 1]], dtype=np.float64)
        else:
            r0 = int(np.min(coords[:, 0]))
            r1 = int(np.max(coords[:, 0])) + 1
            c0 = int(np.min(coords[:, 1]))
            c1 = int(np.max(coords[:, 1])) + 1

        pts_y = coords[:, 0].astype(np.float64)
        pts_x = coords[:, 1].astype(np.float64)
        cy = float(np.mean(pts_y))
        cx = float(np.mean(pts_x))
        dx = pts_x - cx
        dy = pts_y - cy

        cov = np.cov(np.vstack([dx, dy]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        vec = eigvecs[:, 0]
        u_x, u_y = float(vec[0]), float(vec[1])
        if u_x < 0 or (abs(u_x) < 1e-7 and u_y < 0):
            u_x, u_y = -u_x, -u_y
        perp_x, perp_y = -u_y, u_x

        d_perp = dx * perp_x + dy * perp_y
        straightness = float(np.sqrt(np.mean(d_perp ** 2)))

        lam0 = max(0.0, float(eigvals[0]))
        lam1 = max(0.0, float(eigvals[1]))
        aspect = float(np.sqrt(lam0 / max(lam1, 1e-6)))

        ang_h = float(np.degrees(np.arctan2(u_y, u_x)))
        if ang_h > 90.0:
            ang_h -= 180.0
        elif ang_h < -90.0:
            ang_h += 180.0
        angle_from_vertical = float(abs(90.0 - abs(ang_h)))

        t_proj = dx * u_x + dy * u_y
        t_min = float(np.min(t_proj))
        t_max = float(np.max(t_proj))
        length_px = float(t_max - t_min)
        p0 = (cx + t_min * u_x, cy + t_min * u_y)
        p1 = (cx + t_max * u_x, cy + t_max * u_y)

        edge_dist = float(min(r0, c0, h_img - r1, w_img - c1))
        pix_raw = data[coords[:, 0], coords[:, 1]]
        sat_frac = float(np.mean(pix_raw >= 0.9 * frame_max))

        box_w = 40
        patch_len = max(10, int(np.ceil(length_px)))
        try:
            patch, _ = extract_streak_patch(sub, p0, p1, box_width=box_w, output_length=patch_len)
        except Exception:
            patch = np.zeros((box_w, patch_len), dtype=np.float32)

        w_fit = width_estimation(patch, n_sigma=2.0, psf_fwhm=psf_fwhm_px)
        if w_fit is not None and w_fit.get("width_vs_psf_ratio") is not None and not np.isnan(w_fit["width_vs_psf_ratio"]):
            width_ratio = float(w_fit["width_vs_psf_ratio"])
        else:
            cross_prof = np.median(patch, axis=1)
            base = np.percentile(cross_prof, 20)
            pk = np.max(cross_prof)
            half = base + 0.5 * (pk - base)
            above = cross_prof >= half
            fwhm_est = float(np.sum(above)) if np.any(above) else 2.0
            width_ratio = float(fwhm_est / psf_fwhm_px)

        transverse_offsets = np.arange(-4, 5)
        t_steps = np.linspace(t_min, t_max, max(10, int(length_px / 2.0)))
        snrs = []
        for t in t_steps:
            xc_t = cx + t * u_x
            yc_t = cy + t * u_y
            sample_xs = xc_t + transverse_offsets * perp_x
            sample_ys = yc_t + transverse_offsets * perp_y
            vals = map_coordinates(sub_sigma, np.vstack([sample_ys, sample_xs]), order=1, mode="constant", cval=0.0)
            snrs.append(float(np.median(vals)))
        snr_mean = float(np.mean(snrs)) if snrs else 0.0

        feats = [width_ratio, snr_mean, sat_frac, straightness, angle_from_vertical, aspect, edge_dist, length_px]
        p_streak = float(clf.predict_proba([feats])[0, 1])

        cands_out.append({
            "candidate_id": cid,
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "p_streak": p_streak,
        })

    return cands_out


def check_found(cands, angle_deg, point):
    if not cands:
        return False, False, 0.0

    x0, y0 = point
    rad = np.radians(angle_deg)
    perp_x = -np.sin(rad)
    perp_y = -np.cos(rad) if False else np.cos(rad)
    # Proper 2D perpendicular vector: (-sin(theta), cos(theta))
    perp_x = -np.sin(rad)
    perp_y = np.cos(rad)

    found_sel = False
    found_flag = False
    best_p = 0.0

    for cand in cands:
        coords = cand["coords"]
        c_pts = coords[:, 1]
        r_pts = coords[:, 0]
        dists = np.abs((c_pts - x0) * perp_x + (r_pts - y0) * perp_y)
        min_dist = float(np.min(dists)) if len(dists) > 0 else 999.0

        if min_dist <= 15.0:
            found_sel = True
            if cand["p_streak"] > best_p:
                best_p = cand["p_streak"]
            if cand["p_streak"] >= 0.5:
                found_flag = True

    return found_sel, found_flag, best_p


def cmd_run():
    cfg = load_config()
    psf_fwhm_px = cfg.get("characterization", {}).get("psf_fwhm_px", 6.79)
    clf = joblib.load(models_dir / "m1_v1.joblib")

    csv_path = inject_dir / "results.csv"
    existing_keys = set()
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        for _, r in df_old.iterrows():
            existing_keys.add((str(r["frame"]), float(r["level"])))

    # Pre-generate parameters deterministically with seed 11
    rng_master = np.random.default_rng(seed=11)
    planned_params = {}
    for fn in FRAME_NAMES:
        for lvl in LEVELS:
            ang, pt = generate_streak_params(rng_master, 4096, 4096)
            planned_params[(fn, lvl)] = (ang, pt)

    # Frame processing
    results = []
    for fn in FRAME_NAMES:
        fpath = find_frame_path(fn)
        if fpath is None:
            print(f"Error: frame {fn} not found.")
            continue

        raw_data, _ = load_fits(fpath)
        bg, std = estimate_background(raw_data, method="background2d")
        std_map = std if isinstance(std, np.ndarray) else np.full_like(raw_data, std)

        # Baseline
        base_key = (fn, 0.0)
        if base_key not in existing_keys:
            cands_base = run_pipeline_on_image(raw_data, bg, std, cfg, clf)
            n_sel_base = len(cands_base)
            n_fl_base = sum(1 for c in cands_base if c["p_streak"] >= 0.5)
            row_base = {
                "frame": fn,
                "level": 0.0,
                "angle": 0.0,
                "point": "(0,0)",
                "found_selected": False,
                "found_flagged": False,
                "best_p": 0.0,
                "n_selected": n_sel_base,
                "n_flagged": n_fl_base,
            }
            results.append(row_base)
            df_add = pd.DataFrame([row_base])
            df_add.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            existing_keys.add(base_key)
            print(f"[{fn} | Baseline] n_selected={n_sel_base}, n_flagged={n_fl_base}")

        # Injected levels
        for lvl in LEVELS:
            key = (fn, lvl)
            if key in existing_keys:
                continue

            ang, pt = planned_params[(fn, lvl)]
            injected_data = inject_streak(raw_data, std_map, lvl, ang, pt, psf_fwhm_px)
            cands_inj = run_pipeline_on_image(injected_data, bg, std, cfg, clf)

            found_sel, found_fl, best_p = check_found(cands_inj, ang, pt)
            n_sel = len(cands_inj)
            n_fl = sum(1 for c in cands_inj if c["p_streak"] >= 0.5)

            pt_str = f"({pt[0]:.1f},{pt[1]:.1f})"
            row_inj = {
                "frame": fn,
                "level": lvl,
                "angle": round(ang, 2),
                "point": pt_str,
                "found_selected": found_sel,
                "found_flagged": found_fl,
                "best_p": round(best_p, 4),
                "n_selected": n_sel,
                "n_flagged": n_fl,
            }
            results.append(row_inj)
            df_add = pd.DataFrame([row_inj])
            df_add.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            existing_keys.add(key)
            print(
                f"[{fn} | L={lvl:.2f}] ang={ang:.1f}°, pt={pt_str} -> "
                f"sel={found_sel}, flag={found_fl}, best_p={best_p:.3f}, n_sel={n_sel}, n_fl={n_fl}"
            )


def cmd_blind():
    cfg = load_config()
    psf_fwhm_px = cfg.get("characterization", {}).get("psf_fwhm_px", 6.79)

    # 12 injected items (2 per level for levels in BLIND_LEVELS)
    rng_master = np.random.default_rng(seed=11)
    planned_items = []

    # Assign frames across 12 items: 4 per frame
    frame_cycle = FRAME_NAMES * 4  # 12 items

    item_idx = 0
    for lvl in BLIND_LEVELS:
        for _ in range(2):
            fn = frame_cycle[item_idx]
            item_idx += 1
            ang, pt = generate_streak_params(rng_master, 4096, 4096)
            planned_items.append({
                "injected": True,
                "frame": fn,
                "level": lvl,
                "angle": round(ang, 2),
                "point": pt,
            })

    # 6 untouched items: 2 per frame
    for fn in FRAME_NAMES:
        for _ in range(2):
            planned_items.append({
                "injected": False,
                "frame": fn,
                "level": 0.0,
                "angle": 0.0,
                "point": (0.0, 0.0),
            })

    # Shuffle with seeded RNG
    rng_shuffle = np.random.default_rng(seed=11)
    indices = np.arange(len(planned_items))
    rng_shuffle.shuffle(indices)

    shuffled_items = [planned_items[i] for i in indices]

    # Preload frames
    frame_data = {}
    for fn in FRAME_NAMES:
        fpath = find_frame_path(fn)
        raw_data, _ = load_fits(fpath)
        bg, std = estimate_background(raw_data, method="background2d")
        std_map = std if isinstance(std, np.ndarray) else np.full_like(raw_data, std)
        frame_data[fn] = (raw_data, bg, std_map)

    key_rows = []
    for idx, it in enumerate(shuffled_items):
        test_name = f"test_{idx + 1:02d}"
        fn = it["frame"]
        raw_data, bg, std_map = frame_data[fn]

        if it["injected"]:
            img_data = inject_streak(raw_data, std_map, it["level"], it["angle"], it["point"], psf_fwhm_px)
        else:
            img_data = raw_data

        sub = img_data - bg
        h_img, w_img = img_data.shape
        h_down = h_img // 4
        w_down = w_img // 4
        sub_4x4 = sub[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).mean(axis=(1, 3))
        smoothed = gaussian_filter(sub_4x4, sigma=1.5)
        vmin, vmax = np.percentile(smoothed, [20, 99.8])

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(smoothed, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(test_name, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(blind_dir / f"{test_name}.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        key_rows.append({
            "test_name": test_name,
            "frame": fn,
            "injected": it["injected"],
            "level": it["level"],
            "angle": it["angle"],
        })

    # Save key.csv silently without printing
    df_key = pd.DataFrame(key_rows)
    df_key.to_csv(inject_dir / "key.csv", index=False)


class BlindReviewTool:
    def __init__(self):
        self.csv_path = inject_dir / "human.csv"
        self.total = 18

        existing = {}
        if self.csv_path.exists():
            df_old = pd.read_csv(self.csv_path)
            for _, r in df_old.iterrows():
                existing[str(r["test_name"])] = str(r["answer"]).strip()

        self.items = []
        for i in range(1, self.total + 1):
            t_name = f"test_{i:02d}"
            self.items.append({
                "test_name": t_name,
                "answer": existing.get(t_name, ""),
            })

        unreviewed = [idx for idx, it in enumerate(self.items) if not it["answer"]]
        if unreviewed:
            self.current_idx = unreviewed[0]
            print(f"Resuming at image {self.current_idx + 1} of {self.total} (unreviewed: {len(unreviewed)}).")
        else:
            print(f"All {self.total} images are already reviewed! Review mode starting at 1.")
            self.current_idx = 0

        self.fig = None
        self.ax = None

    def save_csv(self):
        records = [it for it in self.items if it["answer"]]
        df_save = pd.DataFrame(records, columns=["test_name", "answer"])
        df_save.to_csv(self.csv_path, index=False)

    def update_view(self):
        it = self.items[self.current_idx]
        img_path = blind_dir / f"{it['test_name']}.png"

        self.ax.clear()
        if img_path.exists():
            img = Image.open(img_path)
            self.ax.imshow(img)
        else:
            self.ax.text(0.5, 0.5, f"Missing:\n{img_path.name}", ha="center", va="center", color="red", fontsize=12)
        self.ax.axis("off")

        ans_str = it["answer"].upper() if it["answer"] else "UNANSWERED"
        title = (
            f"Image {self.current_idx + 1}/{self.total}: {it['test_name']} [{ans_str}]\n"
            f"[y] I see a streak   [n] I do not   [u] Undo   [q] Save & Quit"
        )
        self.fig.suptitle(title, fontsize=12, fontweight="medium")
        self.fig.canvas.draw()

    def on_key(self, event):
        if not event.key:
            return
        k = event.key.lower()
        if k in ("y", "n"):
            it = self.items[self.current_idx]
            it["answer"] = k
            self.save_csv()
            print(f"[{self.current_idx + 1}/{self.total}] {it['test_name']} -> {k}")
            if self.current_idx < self.total - 1:
                self.current_idx += 1
                self.update_view()
            else:
                print("Reached the last image!")
                self.update_view()
        elif k == "u":
            if self.current_idx > 0:
                self.current_idx -= 1
                it = self.items[self.current_idx]
                print(f"Undo -> back to image {self.current_idx + 1}/{self.total} ({it['test_name']}, answer: '{it['answer']}')")
                self.update_view()
            else:
                print("Already at the first image.")
        elif k == "q":
            print("Saving and quitting...")
            self.save_csv()
            plt.close(self.fig)

    def run(self):
        # Configure non-conflicting keybindings
        plt.rcParams["keymap.save"] = []
        plt.rcParams["keymap.quit"] = []
        plt.rcParams["keymap.fullscreen"] = []
        plt.rcParams["keymap.yscale"] = []
        plt.rcParams["keymap.xscale"] = []

        _clash_keys = {"y", "n", "u", "q"}
        for _k in list(plt.rcParams.keys()):
            if _k.startswith("keymap."):
                plt.rcParams[_k] = [
                    v for v in plt.rcParams[_k]
                    if v.lower() not in _clash_keys and not any(v.lower() == f"ctrl+{c}" for c in _clash_keys)
                ]

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_view()
        plt.show()


def cmd_review():
    tool = BlindReviewTool()
    tool.run()


def cmd_report():
    human_csv = inject_dir / "human.csv"
    key_csv = inject_dir / "key.csv"
    results_csv = inject_dir / "results.csv"

    if not human_csv.exists():
        print("Error: data/inject/human.csv does not exist.")
        return

    df_human = pd.read_csv(human_csv)
    if len(df_human) < 18:
        print(f"Error: human.csv has only {len(df_human)}/18 rows. Complete review first.")
        return

    if not key_csv.exists() or not results_csv.exists():
        print("Error: key.csv or results.csv missing.")
        return

    df_key = pd.read_csv(key_csv)
    df_results = pd.read_csv(results_csv)
    df_merged = pd.merge(df_key, df_human, on="test_name")

    # 1. Baseline counts
    df_base = df_results[df_results["level"] == 0.0]
    print("\n=== 1. BASELINE COUNTS PER FRAME ===")
    for _, r in df_base.iterrows():
        print(f"{r['frame']}: selected={int(r['n_selected'])}, flagged={int(r['n_flagged'])}")

    # 2. Pipeline recall by level
    pipe_rows = []
    print("\n=== 2. PIPELINE RECALL BY LEVEL (OVER 3 FRAMES) ===")
    for lvl in LEVELS:
        sub_lvl = df_results[df_results["level"] == lvl]
        n_inj = len(sub_lvl)
        rec_sel = float(sub_lvl["found_selected"].mean()) if n_inj > 0 else 0.0
        rec_fl = float(sub_lvl["found_flagged"].mean()) if n_inj > 0 else 0.0
        pipe_rows.append({
            "level": lvl,
            "n_injected": n_inj,
            "recall_selected": rec_sel,
            "recall_flagged": rec_fl,
        })
        print(f"Level {lvl:4.2f} (n={n_inj}): Selected Recall = {rec_sel*100:5.1f}%, Flagged Recall = {rec_fl*100:5.1f}%")

    df_pipe_report = pd.DataFrame(pipe_rows)

    # 3. Human hit rate by level
    print("\n=== 3. HUMAN HIT RATE BY LEVEL ===")
    human_rows = []
    for lvl in BLIND_LEVELS:
        sub_h = df_merged[(df_merged["injected"] == True) & (df_merged["level"] == lvl)]
        n_h = len(sub_h)
        hit_rate = float((sub_h["answer"] == "y").mean()) if n_h > 0 else 0.0
        human_rows.append({
            "level": lvl,
            "n": n_h,
            "hit_rate": hit_rate,
        })
        print(f"Level {lvl:4.2f} (n={n_h}): Human Hit Rate = {hit_rate*100:5.1f}%")

    df_human_report = pd.DataFrame(human_rows)

    # 4. Human false-positive rate on untouched images
    sub_untouched = df_merged[df_merged["injected"] == False]
    n_unt = len(sub_untouched)
    n_fp = int((sub_untouched["answer"] == "y").sum())
    fp_rate = float(n_fp / n_unt) if n_unt > 0 else 0.0
    print("\n=== 4. HUMAN FALSE-POSITIVE RATE ON UNTOUCHED IMAGES ===")
    print(f"FP Rate: {fp_rate*100:.1f}% ({n_fp}/{n_unt})")

    # 5. Plot outputs/review/inject_report.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(
        df_pipe_report["level"],
        df_pipe_report["recall_selected"],
        "o-",
        color="royalblue",
        lw=2.2,
        label="Pipeline: Selected (blob + stitch)",
    )
    ax.plot(
        df_pipe_report["level"],
        df_pipe_report["recall_flagged"],
        "s-",
        color="forestgreen",
        lw=2.2,
        label="Pipeline: Flagged (p >= 0.5)",
    )
    ax.scatter(
        df_human_report["level"],
        df_human_report["hit_rate"],
        color="crimson",
        s=80,
        marker="D",
        zorder=5,
        label="Human Hit Rate (Blind Overview)",
    )

    ax.axhline(fp_rate, color="crimson", ls=":", lw=1.2, alpha=0.7, label=f"Human FP rate on untouched ({fp_rate*100:.1f}%)")
    ax.set_xscale("log")
    ax.set_xticks(LEVELS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Injected Streak Peak Amplitude (units of background σ)", fontsize=11)
    ax.set_ylabel("Recall / Hit Rate", fontsize=11)
    ax.set_title("Synthetic Streak Injection Benchmark: Pipeline vs Human", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(review_dir / "inject_report.png", bbox_inches="tight", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ILMT Synthetic Streak Injection Benchmark")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("blind")
    subparsers.add_parser("review")
    subparsers.add_parser("report")

    args = parser.parse_args()
    if args.subcommand == "run":
        cmd_run()
    elif args.subcommand == "blind":
        cmd_blind()
    elif args.subcommand == "review":
        cmd_review()
    elif args.subcommand == "report":
        cmd_report()


if __name__ == "__main__":
    main()
