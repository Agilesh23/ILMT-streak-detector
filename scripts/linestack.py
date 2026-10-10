import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation, gaussian_filter, distance_transform_edt
from skimage.measure import label as sk_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)
search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

with open(base_dir / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg.get("detection", {})
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
bypass_cfg = det_cfg.get("long_thin_bypass", None)
bleed_veto_cfg = det_cfg.get("bleed_veto", None)

targets_def = [
    {
        "frame": "20240209-0036",
        "cids": [12678, 13055, 13130, 13441, 14112, 14638, 14846, 15327, 15400, 15703, 16535, 16936],
        "fixed_angle": None,
        "is_neg_ctrl": False,
    },
    {
        "frame": "20241129-0009",
        "cids": [6695],
        "fixed_angle": None,
        "is_neg_ctrl": False,
    },
    {
        "frame": "20250101-0021",
        "cids": [3199],
        "fixed_angle": None,
        "is_neg_ctrl": False,
    },
    {
        "frame": "20230317-0008",
        "cids": [3618],
        "fixed_angle": 155.0,
        "is_neg_ctrl": False,
    },
    {
        "frame": "20240405-0004",
        "cids": [14613, 12522, 17987, 9558, 10716],
        "fixed_angle": None,
        "is_neg_ctrl": False,
    },
    {
        "frame": "20221101-0030",
        "cids": [],
        "fixed_angle": 30.0,
        "is_neg_ctrl": True,
    },
]


def find_frame_path(frame_stem):
    for p in search_root.rglob("*"):
        if p.is_file() and p.stem.lower() == frame_stem.lower() and p.suffix.lower() == ".fits":
            return p
    return None


def get_line_in_frame(cx, cy, angle_deg, w_img, h_img):
    rad = np.radians(angle_deg)
    u_x = float(np.cos(rad))
    u_y = float(np.sin(rad))

    t_candidates = []
    if abs(u_x) > 1e-7:
        for x_edge in [0.0, float(w_img - 1)]:
            t = (x_edge - cx) / u_x
            y = cy + t * u_y
            if 0.0 <= y <= float(h_img - 1):
                t_candidates.append(t)

    if abs(u_y) > 1e-7:
        for y_edge in [0.0, float(h_img - 1)]:
            t = (y_edge - cy) / u_y
            x = cx + t * u_x
            if 0.0 <= x <= float(w_img - 1):
                t_candidates.append(t)

    if len(t_candidates) < 2:
        return None, None, 0.0

    t_min = min(t_candidates)
    t_max = max(t_candidates)
    return t_min, t_max, (t_max - t_min)


def compute_profile(proc_img, cx, cy, angle_deg, t_min, t_max, offsets):
    rad = np.radians(angle_deg)
    u_x = float(np.cos(rad))
    u_y = float(np.sin(rad))
    perp_x = -u_y
    perp_y = u_x

    t_steps = np.arange(t_min, t_max + 0.1, 1.0)
    xs = (cx + t_steps * u_x)[:, None] + offsets[None, :] * perp_x
    ys = (cy + t_steps * u_y)[:, None] + offsets[None, :] * perp_y

    coords = np.vstack([ys.ravel(), xs.ravel()])
    vals = map_coordinates(proc_img, coords, order=1, mode="constant", cval=np.nan).reshape(len(t_steps), len(offsets))

    P = np.nanmean(vals, axis=0)
    return P


def main():
    offsets = np.arange(-60, 61)  # 121 points
    ctrl_shifts = [-450, -350, -250, -150, 150, 250, 350, 450]
    d_mask5 = (offsets >= -5) & (offsets <= 5)

    fig, axes = plt.subplots(len(targets_def), 2, figsize=(16, 24), dpi=100)

    for idx, t_info in enumerate(targets_def):
        frame_stem = t_info["frame"]
        cids = t_info["cids"]
        fixed_angle = t_info["fixed_angle"]
        is_neg = t_info["is_neg_ctrl"]

        fpath = find_frame_path(frame_stem)
        if fpath is None:
            print(f"Error: Frame {frame_stem} not found.")
            continue

        data, header = load_fits(fpath)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
        sub_sigma = sub / np.clip(std_map, 1e-6, None)
        frame_max = float(np.max(data))
        h_img, w_img = data.shape

        # Saturated-core mask (NaN within 60 px of pixels >= 0.9 * frame max)
        sat_mask = (data >= 0.9 * frame_max)
        if np.any(sat_mask):
            dist_sat = distance_transform_edt(~sat_mask)
            is_near_sat = (dist_sat <= 60.0)
        else:
            is_near_sat = np.zeros_like(data, dtype=bool)

        proc_img = sub.copy().astype(np.float64)
        proc_img[is_near_sat] = np.nan

        thresh_5std = 5.0 * std_map
        proc_img = np.clip(proc_img, -thresh_5std, thresh_5std)

        # Retrieve centroids and candidate masks
        coords_dict = {}
        pixel_sets = {}

        if not is_neg:
            g1_csv = base_dir / "data/g1" / f"frame_{frame_stem}.csv"
            if g1_csv.exists():
                df_g1 = pd.read_csv(g1_csv)
                for _, r in df_g1.iterrows():
                    cid_val = int(r["candidate_id"])
                    if cid_val in cids:
                        coords_dict[cid_val] = (float(r["centroid_row"]), float(r["centroid_col"]))

            for b in [1, 2, 3]:
                lab_csv = base_dir / "data/labeled" / f"batch_{b}_v2_labels.csv"
                if lab_csv.exists():
                    df_lab = pd.read_csv(lab_csv)
                    for _, r in df_lab.iterrows():
                        cid_val = int(r["candidate_id"])
                        fn_val = str(r["frame_name"])
                        if cid_val in cids and frame_stem in fn_val:
                            if cid_val not in coords_dict:
                                coords_dict[cid_val] = (float(r["centroid_row"]), float(r["centroid_col"]))

            # Pipeline execution if any cid missing or for 1-piece PCA
            if len(coords_dict) < len(cids) or len(cids) == 1:
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

                for cand in selected:
                    c_label = int(cand["label"])
                    if c_label in cids:
                        members = cand.get("stitched_from", [c_label])
                        cand_mask = np.isin(labeled_map, members)
                        pts = np.argwhere(cand_mask)
                        if len(pts) >= 2:
                            coords_dict[c_label] = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
                            pixel_sets[c_label] = pts

        # Line definition
        residual_px = None
        n_pieces = len(cids)

        if is_neg:
            cx = w_img / 2.0
            cy = h_img / 2.0
            angle_deg = 30.0
        elif n_pieces >= 3:
            pts_x = [coords_dict[cid][1] for cid in cids if cid in coords_dict]
            pts_y = [coords_dict[cid][0] for cid in cids if cid in coords_dict]
            pts_x = np.array(pts_x, dtype=float)
            pts_y = np.array(pts_y, dtype=float)

            cx = float(np.mean(pts_x))
            cy = float(np.mean(pts_y))
            dx = pts_x - cx
            dy = pts_y - cy

            cov = np.cov(np.vstack([dx, dy]))
            eigvals, eigvecs = np.linalg.eigh(cov)
            vec = eigvecs[:, np.argmax(eigvals)]
            angle_deg = float(np.degrees(np.arctan2(vec[1], vec[0]))) % 180.0

            rad = np.radians(angle_deg)
            perp_x = -np.sin(rad)
            perp_y = np.cos(rad)
            d_perp = dx * perp_x + dy * perp_y
            residual_px = float(np.sqrt(np.mean(d_perp ** 2)))
        elif fixed_angle is not None:
            cid = cids[0]
            cy, cx = coords_dict[cid]
            angle_deg = fixed_angle
        else:
            cid = cids[0]
            cy, cx = coords_dict[cid]
            pts = pixel_sets.get(cid)
            if pts is not None and len(pts) >= 2:
                pts_y = pts[:, 0].astype(float)
                pts_x = pts[:, 1].astype(float)
                dx = pts_x - cx
                dy = pts_y - cy
                cov = np.cov(np.vstack([dx, dy]))
                eigvals, eigvecs = np.linalg.eigh(cov)
                vec = eigvecs[:, np.argmax(eigvals)]
                pca_angle = float(np.degrees(np.arctan2(vec[1], vec[0]))) % 180.0
                rad_pca = np.radians(pca_angle)
                t_proj = dx * np.cos(rad_pca) + dy * np.sin(rad_pca)
                length_px = float(np.max(t_proj) - np.min(t_proj))
            else:
                pca_angle = 0.0
                length_px = 150.0

            if length_px < 100.0:
                # Scan +/-20 degrees around axis in 1 degree steps
                best_peak_z = -1e9
                best_scan_angle = pca_angle

                for scan_deg in np.arange(pca_angle - 20, pca_angle + 21, 1.0):
                    theta = float(scan_deg % 180.0)
                    t_min_s, t_max_s, len_s = get_line_in_frame(cx, cy, theta, w_img, h_img)
                    if t_min_s is None or len_s < 200:
                        continue

                    p0_target = compute_profile(proc_img, cx, cy, theta, t_min_s, t_max_s, np.array([0.0]))[0]
                    rad_s = np.radians(theta)
                    p_x = -np.sin(rad_s)
                    p_y = np.cos(rad_s)

                    ctrl_p0s = []
                    for cs in ctrl_shifts:
                        cx_c = cx + cs * p_x
                        cy_c = cy + cs * p_y
                        tm_c, tx_c, l_c = get_line_in_frame(cx_c, cy_c, theta, w_img, h_img)
                        if tm_c is not None and l_c >= 500:
                            p0_c = compute_profile(proc_img, cx_c, cy_c, theta, tm_c, tx_c, np.array([0.0]))[0]
                            if not np.isnan(p0_c):
                                ctrl_p0s.append(p0_c)

                    if len(ctrl_p0s) >= 2:
                        med_c = float(np.median(ctrl_p0s))
                        mad_c = float(np.median(np.abs(ctrl_p0s - med_c)))
                        sc = max(1.4826 * mad_c, 0.05)
                        z0 = (p0_target - med_c) / sc
                        if z0 > best_peak_z:
                            best_peak_z = z0
                            best_scan_angle = theta

                angle_deg = best_scan_angle
            else:
                angle_deg = pca_angle

        # Line in frame
        t_min, t_max, line_len = get_line_in_frame(cx, cy, angle_deg, w_img, h_img)
        P_line = compute_profile(proc_img, cx, cy, angle_deg, t_min, t_max, offsets)

        # Controls
        rad = np.radians(angle_deg)
        perp_x = -np.sin(rad)
        perp_y = np.cos(rad)

        valid_ctrl_profiles = []
        for cs in ctrl_shifts:
            cx_c = cx + cs * perp_x
            cy_c = cy + cs * perp_y
            tm_c, tx_c, l_c = get_line_in_frame(cx_c, cy_c, angle_deg, w_img, h_img)
            if tm_c is not None and l_c >= 500:
                P_c = compute_profile(proc_img, cx_c, cy_c, angle_deg, tm_c, tx_c, offsets)
                if np.sum(~np.isnan(P_c)) >= 0.5 * len(offsets):
                    valid_ctrl_profiles.append(P_c)

        # Robust z scaling from all control profile values
        if len(valid_ctrl_profiles) > 0:
            all_c_vals = np.concatenate([pc[~np.isnan(pc)] for pc in valid_ctrl_profiles])
            ctrl_med = float(np.median(all_c_vals))
            ctrl_mad = float(np.median(np.abs(all_c_vals - ctrl_med)))
            scale = float(max(1.4826 * ctrl_mad, 0.05))
        else:
            ctrl_med = 0.0
            scale = 0.05

        z_line = (P_line - ctrl_med) / scale
        z_controls = [(pc - ctrl_med) / scale for pc in valid_ctrl_profiles]

        line_peak_z = float(np.nanmax(z_line[d_mask5]))
        ctrl_peaks = [float(np.nanmax(zc[d_mask5])) for zc in z_controls]
        max_ctrl_peak_z = max(ctrl_peaks) if ctrl_peaks else 0.0
        n_ctrl_gt_3 = sum(1 for p in ctrl_peaks if p > 3.0)

        res_str = f"{residual_px:6.2f} px" if residual_px is not None else "   n/a"
        print(
            f"{frame_stem}: n_pieces={n_pieces:2d}, angle={angle_deg:5.1f} deg, "
            f"residual={res_str}, line_len={line_len:4.0f} px, "
            f"peak_z={line_peak_z:5.2f}, max_ctrl_peak={max_ctrl_peak_z:5.2f}, "
            f"ctrl_gt_3={n_ctrl_gt_3}/{len(valid_ctrl_profiles)}"
        )

        # Plot Column (a): P(d) in z units
        ax_a = axes[idx, 0]
        for zc in z_controls:
            ax_a.plot(offsets, zc, color="gray", lw=0.8, alpha=0.6)
        ax_a.plot(offsets, z_line, color="royalblue", lw=2.5, label="Line")

        ax_a.axvline(0, color="black", lw=0.8, ls=":")
        ax_a.axhline(0, color="black", lw=0.8, ls=":")
        ax_a.axhline(3.0, color="red", lw=0.8, ls="--", alpha=0.7)
        ax_a.set_xlim(-60, 60)
        ax_a.set_xlabel("Perpendicular offset d (px)", fontsize=8.5)
        ax_a.set_ylabel("P(d) (z units)", fontsize=8.5)

        title_res = f"res={residual_px:.2f}px" if residual_px is not None else "res=n/a"
        title_a = f"{frame_stem} | n={n_pieces} | {angle_deg:.1f}° | {title_res} | peak_z={line_peak_z:.2f}"
        ax_a.set_title(title_a, fontsize=9.5, fontweight="bold")
        ax_a.grid(True, linestyle=":", alpha=0.4)

        # Plot Column (b): Straightened strip (+/-80 px, 4x along-line downsampled, gaussian sigma 2)
        u_x = float(np.cos(rad))
        u_y = float(np.sin(rad))
        t_down = np.arange(t_min, t_max + 0.1, 4.0)
        w_steps = np.arange(-80, 81, 1.0)  # 161 points

        xs_strip = (cx + t_down * u_x)[:, None] + w_steps[None, :] * perp_x
        ys_strip = (cy + t_down * u_y)[:, None] + w_steps[None, :] * perp_y

        coords_strip = np.vstack([ys_strip.ravel(), xs_strip.ravel()])
        strip = map_coordinates(sub_sigma, coords_strip, order=1, mode="constant", cval=0.0).reshape(len(t_down), len(w_steps)).T

        strip_smooth = gaussian_filter(strip, sigma=2.0)

        ax_b = axes[idx, 1]
        ax_b.imshow(strip_smooth, cmap="gray", origin="lower", vmin=-0.5, vmax=2.0, aspect="auto")
        ax_b.axhline(80, color="orange", lw=1.2, ls="--")
        ax_b.set_yticks([0, 40, 80, 120, 160])
        ax_b.set_yticklabels([-80, -40, 0, 40, 80])
        ax_b.set_ylabel("Offset w (px)", fontsize=8.5)
        ax_b.set_xlabel("Along-line pos (4x binned)", fontsize=8.5)
        ax_b.set_title("Straightened strip (±80 px, 4x binned, σ=2)", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(review_dir / "linestack_sheet.png", bbox_inches="tight", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
