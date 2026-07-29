"""
I/O and Tiling Utilities for ILMT Streak Detector.

Provides FITS loading with memmap and float32 casting, row-tiling splitting,
and candidate deduplication across tile overlaps.
"""

from pathlib import Path
import numpy as np
from astropy.io import fits


def load_fits(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a FITS file using memmap=True and cast data to float32 at load time.

    NOTE ON MEMMAP & FLOAT32 CASTING:
    Opening via astropy.io.fits.open(..., memmap=True) creates a memory-mapped array.
    However, calling .astype(np.float32) (or endianness conversions like >f4 to native float32)
    immediately materializes the array in system RAM. On 67MB ILMT frames (4096x4096 float32 = 67MB),
    this RAM footprint (~67MB per frame) is trivial for modern hardware, so immediate float32
    casting is safe, predictable, and prevents repeated casting downstream in estimation/detection.

    Parameters
    ----------
    path : str | Path
        Path to FITS file.

    Returns
    -------
    data, header : tuple[np.ndarray, dict]
        2D image data cast to float32, and FITS header dictionary.
    """
    with fits.open(path, memmap=True) as hdul:
        raw_data = hdul[0].data
        header = dict(hdul[0].header)
        data = np.ascontiguousarray(raw_data, dtype=np.float32)
    return data, header


def split_row_tiles(data: np.ndarray, tile_height: int = 1024, overlap: int = 100) -> list[tuple[np.ndarray, int, int]]:
    """
    Split a 2D image array into row tiles with a specified pixel overlap.

    Parameters
    ----------
    data : np.ndarray
        2D image array of shape (height, width).
    tile_height : int, default 1024
        Height of each row tile in pixels.
    overlap : int, default 100
        Overlap in pixels between consecutive row tiles.

    Returns
    -------
    tiles : list[tuple[np.ndarray, int, int]]
        List of tuples (tile_data, y_start, y_end) where y_start:y_end define the tile crop range.
    """
    height, width = data.shape
    tiles = []
    y_start = 0
    while y_start < height:
        y_end = min(y_start + tile_height, height)
        tile_data = data[y_start:y_end, :]
        tiles.append((tile_data, y_start, y_end))
        if y_end == height:
            break
        y_start += (tile_height - overlap)
    return tiles


def deduplicate_overlap_candidates(candidates: list[dict], distance_threshold: float = 20.0) -> list[dict]:
    """
    Deduplicate streak candidates detected in overlap zones between row tiles.

    Candidates identified in neighboring tiles with centroids within distance_threshold pixels
    are merged, keeping the candidate with higher Radon SNR or higher score.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dictionaries, each containing 'centroid' (y, x) in full-frame coordinates.
    distance_threshold : float, default 20.0
        Centroid distance threshold in pixels for considering candidates identical.

    Returns
    -------
    deduped : list[dict]
        Deduplicated list of candidate dictionaries.
    """
    if not candidates:
        return []

    keep = []
    used = set()

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get('radon_snr', c.get('flux_snr', 0.0)),
        reverse=True
    )

    for i, c1 in enumerate(sorted_candidates):
        if i in used:
            continue
        keep.append(c1)
        used.add(i)
        cy1, cx1 = c1['centroid']
        for j in range(i + 1, len(sorted_candidates)):
            if j in used:
                continue
            c2 = sorted_candidates[j]
            cy2, cx2 = c2['centroid']
            dist = np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2)
            if dist < distance_threshold:
                used.add(j)

    return keep
