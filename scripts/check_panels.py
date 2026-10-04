import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage.measure import label as sk_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = Path(__file__).resolve().parent.parent
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
data_dir = base_dir / "data"
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)

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

# Load stored label CSVs
df1 = pd.read_csv(data_dir / "labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "labeled/batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)
df_all["frame_clean"] = df_all["frame_name"].astype(str).str.replace(".fits", "", regex=False)

target_candidates = [
    ("20221025-0028", 10277),
    ("20230519-0009", 10540),
    ("20221101-0030", 9983),
    ("20230317-0008", 3618),
    ("20221026-0057", 11455),
]


def get_crop(img, cy, cx, size):
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
    return crop, (c0, c1, r0, r1)


fig, axes = plt.subplots(len(target_candidates), 3, figsize=(15, 25), dpi=100)

for idx, (frame_clean, cid) in enumerate(target_candidates):
    # Lookup stored location in CSV
    m = df_all[(df_all["frame_clean"] == frame_clean) & (df_all["candidate_id"] == cid)]
    if len(m) > 0:
        row = m.iloc[0]
        label_val = str(row.get("label", "unknown"))
        stored_cy = float(row.get("centroid_row", np.nan))
        stored_cx = float(row.get("centroid_col", np.nan))
        stored_r0 = int(row.get("bbox_r0", 0))
        stored_c0 = int(row.get("bbox_c0", 0))
        stored_r1 = int(row.get("bbox_r1", 0))
        stored_c1 = int(row.get("bbox_c1", 0))
        stored_bbox = (stored_r0, stored_c0, stored_r1, stored_c1)
    else:
        label_val = "unknown"
        stored_cy, stored_cx = np.nan, np.nan
        stored_bbox = (0, 0, 0, 0)

    # Reproduce candidate detection
    fpath = frame_dir / f"{frame_clean}.fits"
    data, header = load_fits(fpath)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
    sub_sigma = sub / np.clip(std_map, 1e-6, None)

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

    cand = next((c for c in selected if int(c["label"]) == cid), None)
    if cand is None:
        cand = next((c for c in stitched if int(c["label"]) == cid), None)
    if cand is None:
        cand = next((c for c in candidates if int(c["label"]) == cid), None)

    if cand is not None:
        members = cand.get("stitched_from", [cid])
        n_members = len(members)
        cand_mask = np.isin(labeled_map, members)
        coords = np.argwhere(cand_mask)
    else:
        members = [cid]
        n_members = 0
        cand_mask = np.zeros_like(data, dtype=bool)
        coords = np.empty((0, 2), dtype=int)

    if len(coords) >= 2:
        rep_r0 = int(np.min(coords[:, 0]))
        rep_r1 = int(np.max(coords[:, 0])) + 1
        rep_c0 = int(np.min(coords[:, 1]))
        rep_c1 = int(np.max(coords[:, 1])) + 1
        rep_bbox = (rep_r0, rep_c0, rep_r1, rep_c1)
        rep_cy = float(np.mean(coords[:, 0]))
        rep_cx = float(np.mean(coords[:, 1]))
        mask_pix = len(coords)

        vals = sub_sigma[coords[:, 0], coords[:, 1]]
        peak_sigma = float(np.max(vals))
        mean_sigma = float(np.mean(vals))

        pts_y = coords[:, 0].astype(np.float64)
        pts_x = coords[:, 1].astype(np.float64)
        dx = pts_x - rep_cx
        dy = pts_y - rep_cy
        cov = np.cov(np.vstack([dx, dy]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        vec = eigvecs[:, idx_sort][:, 0]
        u_x, u_y = float(vec[0]), float(vec[1])
        if u_x < 0 or (abs(u_x) < 1e-7 and u_y < 0):
            u_x, u_y = -u_x, -u_y
        t_proj = dx * u_x + dy * u_y
        t_min = float(np.min(t_proj))
        t_max = float(np.max(t_proj))
        p0 = (rep_cx + t_min * u_x, rep_cy + t_min * u_y)
        p1 = (rep_cx + t_max * u_x, rep_cy + t_max * u_y)
    else:
        rep_cy, rep_cx = stored_cy, stored_cx
        rep_bbox = stored_bbox
        mask_pix = 0
        peak_sigma = 0.0
        mean_sigma = 0.0
        p0 = (rep_cx - 5, rep_cy - 5)
        p1 = (rep_cx + 5, rep_cy + 5)

    if np.isnan(stored_cy) or np.isnan(stored_cx):
        stored_cy, stored_cx = rep_cy, rep_cx

    # Print candidate summary line
    print(
        f"Frame: {frame_clean}, ID: {cid} | "
        f"Stored centroid: ({stored_cy:.1f}, {stored_cx:.1f}), bbox: {stored_bbox} | "
        f"Rep centroid: ({rep_cy:.1f}, {rep_cx:.1f}), bbox: {rep_bbox} | "
        f"Mask pixels: {mask_pix} | "
        f"Peak sigma: {peak_sigma:.2f}, Mean sigma: {mean_sigma:.2f} | "
        f"n_members: {n_members}"
    )

    # Column 1: 120x120 crop centred on stored centroid, sigma units, fixed stretch -1 to +5 sigma
    crop120, (c0_120, c1_120, r0_120, r1_120) = get_crop(sub_sigma, stored_cy, stored_cx, 120)
    ax1 = axes[idx, 0]
    ax1.imshow(crop120, cmap="gray", origin="lower", extent=[c0_120, c1_120, r0_120, r1_120], vmin=-1.0, vmax=5.0)
    ax1.set_title(f"{frame_clean}, {cid}, {label_val}", fontsize=10)
    ax1.set_xlim(c0_120, c1_120)
    ax1.set_ylim(r0_120, r1_120)

    # Column 2: same crop with mask contour in cyan and PCA line in orange
    mask_crop, _ = get_crop(cand_mask.astype(np.float32), stored_cy, stored_cx, 120)
    ax2 = axes[idx, 1]
    ax2.imshow(crop120, cmap="gray", origin="lower", extent=[c0_120, c1_120, r0_120, r1_120], vmin=-1.0, vmax=5.0)
    if np.any(mask_crop > 0.5):
        xs = np.arange(c0_120, c1_120) + 0.5
        ys = np.arange(r0_120, r1_120) + 0.5
        X, Y = np.meshgrid(xs, ys)
        ax2.contour(X, Y, mask_crop, levels=[0.5], colors=["cyan"], linewidths=1.2)
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], color="orange", lw=1.5)
    ax2.set_title(f"{frame_clean}, {cid}, {label_val}", fontsize=10)
    ax2.set_xlim(c0_120, c1_120)
    ax2.set_ylim(r0_120, r1_120)

    # Column 3: 600x600 px crop with box at stored bbox
    crop600, (c0_600, c1_600, r0_600, r1_600) = get_crop(sub_sigma, stored_cy, stored_cx, 600)
    vmin3, vmax3 = np.percentile(crop600, [5, 99.5])
    ax3 = axes[idx, 2]
    ax3.imshow(crop600, cmap="gray", origin="lower", extent=[c0_600, c1_600, r0_600, r1_600], vmin=vmin3, vmax=vmax3)
    rect = patches.Rectangle(
        (stored_bbox[1], stored_bbox[0]),
        stored_bbox[3] - stored_bbox[1],
        stored_bbox[2] - stored_bbox[0],
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax3.add_patch(rect)
    ax3.set_title(f"{frame_clean}, {cid}, {label_val}", fontsize=10)
    ax3.set_xlim(c0_600, c1_600)
    ax3.set_ylim(r0_600, r1_600)

plt.tight_layout()
out_png = review_dir / "panel_check.png"
plt.savefig(out_png, bbox_inches="tight", dpi=100)
plt.close()
