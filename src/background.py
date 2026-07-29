"""
Unified Background Subtraction Module for ILMT Streak Detector.

Provides single-entry-point background estimation supporting fast scalar
sigma-clipping or 2D spatial background mapping via photutils.Background2D.
"""

import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground, StdBackgroundRMS


def estimate_background(data: np.ndarray, method: str = "sigma_clip", sigma: float = 3.0, box_size: int = 64, filter_size: int = 3) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Estimate background level and background RMS (std dev) for astronomical image data.

    Parameters
    ----------
    data : np.ndarray
        Input 2D image data (expected float32 cast at load time).
    method : str, default "sigma_clip"
        "sigma_clip" returns global scalar values (bg, std).
        "background2d" returns 2D background and RMS maps matching data.shape.
    sigma : float, default 3.0
        Sigma threshold for clipping.
    box_size : int, default 64
        Box size in pixels for photutils Background2D.
    filter_size : int, default 3
        Median filter size for photutils Background2D.

    Returns
    -------
    bg, std : tuple[float | np.ndarray, float | np.ndarray]
        - If method == "sigma_clip": (float, float) scalars representing median sky background and RMS.
        - If method == "background2d": (np.ndarray, np.ndarray) 2D float32 maps matching data.shape.
    """
    if method == "sigma_clip":
        mean, median, std = sigma_clipped_stats(data, sigma=sigma)
        return float(median), float(std)
    elif method == "background2d":
        sigma_clip = SigmaClip(sigma=sigma)
        bkg_estimator = MedianBackground()
        bkgrms_estimator = StdBackgroundRMS()
        bkg = Background2D(
            data,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            bkg_rms_estimator=bkgrms_estimator,
        )
        return bkg.background.astype(np.float32), bkg.background_rms.astype(np.float32)
    else:
        raise ValueError(f"Unknown background estimation method '{method}'. Supported methods: 'sigma_clip', 'background2d'.")
