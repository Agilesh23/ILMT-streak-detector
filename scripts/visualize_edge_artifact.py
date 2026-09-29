"""
Visualization Script for Task C4e-visual:
Extracts and saves PNG crops of the rightmost frame edge (X=3900-4096) and a middle-frame control region
(X=2000-2196) from 20230317-0008.fits to visually inspect the suspected CCD edge column artifact.

Instructions for User:
Execute from PowerShell:
    python scripts/visualize_edge_artifact.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup project path
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits


def main():
    frame_path = base_dir.parent / "ILMT_Contest_Materials" / "test_files" / "20230317-0008.fits"
    if not frame_path.exists():
        frame_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")

    print(f"=== TASK C4e VISUAL CROP EXTRACTION ===")
    print(f"Loading Frame: {frame_path}")

    data, header = load_fits(frame_path)
    h, w = data.shape
    print(f"Frame Dimensions: {w}x{h} px")

    # Output directory
    out_dir = base_dir / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Edge crop: X = 3900..4096 (width = 196 px), Y = 1000..3000 (representative vertical slice)
    # Also create full height version
    y_start, y_end = 0, h
    edge_crop = data[y_start:y_end, 3900:w]
    middle_crop = data[y_start:y_end, 2000:2196]

    # Percentile-based display scaling (1st to 99th percentile)
    def save_scaled_png(crop: np.ndarray, out_path: Path):
        p_vmin, p_vmax = float(np.percentile(crop, 1)), float(np.percentile(crop, 99))
        if p_vmax <= p_vmin:
            p_vmin, p_vmax = float(np.min(crop)), float(np.max(crop))
        if p_vmax <= p_vmin:
            p_vmin, p_vmax = 0.0, 1.0

        plt.imsave(str(out_path), crop, cmap="gray", origin="lower", vmin=p_vmin, vmax=p_vmax)
        print(f"  Saved Image: {out_path} (Display Range: [{p_vmin:.1f}, {p_vmax:.1f}] ADU)")

    edge_out_path = out_dir / "20230317-0008_edge_crop_X3900_4096.png"
    middle_out_path = out_dir / "20230317-0008_middle_crop_X2000_2196.png"

    print("\nSaving crops using 1st-99th percentile scaling:")
    save_scaled_png(edge_crop, edge_out_path)
    save_scaled_png(middle_crop, middle_out_path)

    # Save a zoomed-in slice Y=1500..2500 for high-detail inspection
    edge_zoom = data[1500:2500, 3900:w]
    middle_zoom = data[1500:2500, 2000:2196]

    edge_zoom_path = out_dir / "20230317-0008_edge_zoom_Y1500_2500.png"
    middle_zoom_path = out_dir / "20230317-0008_middle_zoom_Y1500_2500.png"

    print("\nSaving 1000px zoomed vertical slices (Y=1500-2500):")
    save_scaled_png(edge_zoom, edge_zoom_path)
    save_scaled_png(middle_zoom, middle_zoom_path)

    print("\n=== VISUAL CROP GENERATION COMPLETE ===")
    print(f"Edge Crop File Path:   {edge_out_path}")
    print(f"Middle Crop File Path: {middle_out_path}")
    print(f"Edge Zoom File Path:   {edge_zoom_path}")
    print(f"Middle Zoom File Path: {middle_zoom_path}")


if __name__ == "__main__":
    main()
