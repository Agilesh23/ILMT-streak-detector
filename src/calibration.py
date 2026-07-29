"""
Radon Angle-Convention Calibration Module for ILMT Streak Detector.

Provides synthetic streak injection and empirical measurement of the offset
between injected spatial streak angles and skimage.transform.radon peak angles.
"""

from pathlib import Path
import numpy as np
import yaml
from skimage.transform import radon


def inject_synthetic_streak(
    shape: tuple[int, int] = (200, 200),
    length: float = 100.0,
    angle_deg: float = 45.0,
    intensity: float = 100.0,
    width: float = 3.0,
    noise_std: float = 5.0,
    center: tuple[float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Inject a synthetic linear streak with a Gaussian cross-section into a 2D image array.

    Parameters
    ----------
    shape : tuple[int, int], default (200, 200)
        Image array shape (height, width).
    length : float, default 100.0
        Streak length in pixels.
    angle_deg : float, default 45.0
        Spatial angle of the streak relative to horizontal (x-axis) in degrees.
    intensity : float, default 100.0
        Peak intensity in ADU above background.
    width : float, default 3.0
        FWHM width of the streak profile in pixels.
    noise_std : float, default 5.0
        Gaussian background noise standard deviation.
    center : tuple[float, float] | None, default None
        (cy, cx) center coordinates of streak. Defaults to center of image.

    Returns
    -------
    img, start_point, end_point : tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        Synthetic image array (float32), start_point (x, y), end_point (x, y).
    """
    h, w = shape
    if center is None:
        cy, cx = h / 2.0, w / 2.0
    else:
        cy, cx = center

    y_grid, x_grid = np.ogrid[:h, :w]
    angle_rad = np.radians(angle_deg)

    ux, uy = np.cos(angle_rad), np.sin(angle_rad)
    dx = x_grid - cx
    dy = y_grid - cy

    proj_along = dx * ux + dy * uy
    dist_perp = np.abs(-dx * uy + dy * ux)

    sigma = width / 2.355 if width > 0 else 1.0
    length_mask = (np.abs(proj_along) <= (length / 2.0)).astype(np.float32)
    streak = intensity * np.exp(-0.5 * (dist_perp / sigma) ** 2) * length_mask

    noise = np.random.normal(0.0, noise_std, size=shape).astype(np.float32)
    img = (streak + noise).astype(np.float32)

    start_point = (float(cx - (length / 2.0) * ux), float(cy - (length / 2.0) * uy))
    end_point = (float(cx + (length / 2.0) * ux), float(cy + (length / 2.0) * uy))

    return img, start_point, end_point


def calibrate_radon_angle_offset(
    test_angles: list[float] | None = None,
    config_path: str | Path | None = None
) -> float:
    """
    Calibrate empirical angle offset between spatial streak orientation and Radon sinogram peak.

    Parameters
    ----------
    test_angles : list[float] | None, default None
        List of test angles in degrees to inject and measure. Defaults to [15, 30, 45, 60, 75, 120, 150].
    config_path : str | Path | None, default None
        Path to config.yaml to update with calibrated offset.

    Returns
    -------
    offset_deg : float
        Mean angle conversion offset in degrees to convert Radon peak theta to spatial orientation.
    """
    if test_angles is None:
        test_angles = [15.0, 30.0, 45.0, 60.0, 75.0, 120.0, 150.0]

    offsets = []
    theta = np.linspace(0.0, 180.0, 180, endpoint=False)

    for injected_angle in test_angles:
        img, _, _ = inject_synthetic_streak(
            shape=(300, 300),
            length=150.0,
            angle_deg=injected_angle,
            intensity=200.0,
            noise_std=2.0
        )
        sino = radon(img, theta=theta, circle=False)
        peak_theta_idx = np.unravel_index(np.argmax(sino), sino.shape)[1]
        recovered_angle = theta[peak_theta_idx]

        # Calculate offset mapping Radon theta to spatial angle (modulo 180)
        # Spatial angle = (90 - Radon peak angle) % 180
        spatial_from_radon = (90.0 - recovered_angle) % 180.0
        diff = (injected_angle - spatial_from_radon) % 180.0
        if diff > 90.0:
            diff -= 180.0
        offsets.append(diff)

    mean_offset = float(np.mean(offsets))

    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("calibration", {})["radon_angle_offset_deg"] = mean_offset
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)

    return mean_offset
