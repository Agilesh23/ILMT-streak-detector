import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")

items = [
    ("20221026-0057.fits", 770, "FP (770)"),
    ("20221026-0057.fits", 11455, "FP (11455)"),
    ("20221026-0057.fits", 18383, "FP (18383)"),
    ("20240114-0028.fits", 18665, "FP (18665)"),
    ("20221026-0057.fits", 1175, "Streak (1175, comparison)")
]

fig, axes = plt.subplots(5, 2, figsize=(8, 16))

print(f"{'Candidate':<35} | {'Angle (deg)':<12} | {'Length (px)':<12} | {'Score':<8}")
print("-" * 75)

for i, (fname, cid, title_desc) in enumerate(items):
    row = df2[(df2["frame_name"] == fname) & (df2["candidate_id"] == cid)].iloc[0]
    cutout_p = base_dir / row["cutout_path"]
    context_p = base_dir / row["context_path"]
    
    img_cut = Image.open(cutout_p)
    img_ctx = Image.open(context_p)
    
    axes[i, 0].imshow(img_cut)
    axes[i, 0].set_title(f"{fname} {title_desc}\nCutout (L={row['length_px']:.1f}px, θ={row['angle_deg']:.1f}°)", fontsize=9)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(img_ctx)
    axes[i, 1].set_title(f"Context (Score={row['score']:.1f})", fontsize=9)
    axes[i, 1].axis('off')
    
    print(f"{fname} Cand {cid:<6} ({row['label']}) | {row['angle_deg']:<12.2f} | {row['length_px']:<12.2f} | {row['score']:<8.2f}")

plt.tight_layout()
out_png = base_dir / "outputs/fp_audit.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_png}")
