"""
Physical Characterization Module for ILMT Streak Detector.

Ports and fixes physical streak characterization functions from the Hough transform notebook:
- extract_streak_patch: rectifies streak cutout along bounding boxes
- fit_weighted_tls_line: refits streak orientation across multi-fragment centroids via weighted TLS/PCA
- compute_max_inbounds_ray_length: calculates max in-bounds ray distance to image boundary
- extract_candidate_cutout: extracts rectified streak cutouts (with refit line, trim margin & exact 0-margin boundary clipping)
- width_estimation: fits Gaussian cross-section profile, calculates width_vs_psf_ratio
- calculate_error: computes per-pixel ADU noise (shot + read noise)
- compute_signal_and_snr: calculates integrated signal, noise, flux, and SNR (BUG FIXED)
- mask_star: masks bright stellar sources ahead of flux integration
"""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit


def estimate_start_end_points(img_shape: tuple[int, int], angle: float, dist: float) -> list[tuple[int, int]] | None:
    """
    Estimate intersection points of detected Hough/Radon line with image borders.
    """
    height, width = img_shape
    points = []

    for x in [0, width - 1]:
        if np.sin(angle) != 0:
            y = (dist - x * np.cos(angle)) / np.sin(angle)
            if 0 <= y < height:
                points.append((int(x), int(y)))

    for y in [0, height - 1]:
        if np.cos(angle) != 0:
            x = (dist - y * np.sin(angle)) / np.cos(angle)
            if 0 <= x < width:
                points.append((int(x), int(y)))

    if len(points) >= 2:
        return points[:2]
    return None


def extract_streak_patch(
    image: np.ndarray,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    box_width: int = 100,
    output_length: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a rectified rectangular cutout patch along a streak defined by start and end points.

    Ported unchanged from Hough notebook.
    """
    start = np.array(start_point, dtype=np.float32)
    end = np.array(end_point, dtype=np.float32)
    direction = end - start
    length = np.linalg.norm(direction)

    if length == 0:
        raise ValueError("Start and end points are the same. Cannot define streak direction.")
    unit_direction = direction / length

    perp = np.array([-unit_direction[1], unit_direction[0]])
    half_width = box_width / 2.0
    offset = perp * half_width

    corner1 = start + offset
    corner2 = start - offset
    corner3 = end - offset
    corner4 = end + offset
    bbox_points = np.array([corner1, corner2, corner3, corner4], dtype=np.float32)

    patch_width = int(box_width)
    if output_length is None:
        output_length = int(length)

    dst_x, dst_y = np.meshgrid(
        np.linspace(0, 1, output_length),
        np.linspace(0, 1, patch_width)
    )
    dst_x_flat = dst_x.flatten()
    dst_y_flat = dst_y.flatten()

    top = bbox_points[0] + (bbox_points[3] - bbox_points[0]) * dst_x_flat[:, None]
    bottom = bbox_points[1] + (bbox_points[2] - bbox_points[1]) * dst_x_flat[:, None]
    src_pts = top + (bottom - top) * dst_y_flat[:, None]

    src_x = np.clip(src_pts[:, 0], 0, image.shape[1] - 1)
    src_y = np.clip(src_pts[:, 1], 0, image.shape[0] - 1)

    patch = map_coordinates(image, [src_y, src_x], order=1, mode='reflect')
    patch = patch.reshape((patch_width, output_length))

    return patch, bbox_points


def fit_weighted_tls_line(centroids: list[tuple[float, float]], weights: list[float]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Fit a Total Least Squares (TLS) / Weighted Principal Component line to points.

    JUSTIFICATION FOR WEIGHTED PCA / TLS OVER POLYNOMIAL OLS:
    Standard ordinary least squares (e.g. np.polyfit y = m*x + c) fails on vertical or
    near-vertical streaks because the slope m = dy/dx approaches infinity (singularity).
    Weighted PCA / TLS fits a line by finding the principal eigenvector of the 2D weighted
    covariance matrix of (x, y) coordinates. This is rotationally invariant, handles
    arbitrary streak orientations seamlessly, and minimizes orthogonal distance from
    centroids to the fitted line. Weighting by fragment area ensures larger, well-defined
    fragments dominate the line orientation over tiny noise blobs.

    Parameters
    ----------
    centroids : list[tuple[float, float]]
        List of (x, y) centroid coordinates.
    weights : list[float]
        List of non-negative scalar weights (e.g. fragment area or major axis length).

    Returns
    -------
    center, direction, angle_deg : tuple
        - center: (center_x, center_y) weighted centroid
        - direction: (vx, vy) unit direction vector along streak
        - angle_deg: angle of line in degrees in [0, 180)
    """
    pts = np.array(centroids, dtype=np.float64)  # shape (N, 2): column 0 is X, column 1 is Y
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)

    if w_sum <= 0:
        w = np.ones(len(pts), dtype=np.float64)
        w_sum = np.sum(w)

    # Weighted mean centroid
    center_x = float(np.sum(w * pts[:, 0]) / w_sum)
    center_y = float(np.sum(w * pts[:, 1]) / w_sum)
    center = (center_x, center_y)

    # Center coordinates
    dx = pts[:, 0] - center_x
    dy = pts[:, 1] - center_y

    # Weighted 2x2 covariance matrix
    cov_xx = np.sum(w * dx * dx) / w_sum
    cov_xy = np.sum(w * dx * dy) / w_sum
    cov_yy = np.sum(w * dy * dy) / w_sum

    cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    # Principal Component Analysis (eigen-decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Principal direction is eigenvector for largest eigenvalue
    primary_vector = eigenvectors[:, np.argmax(eigenvalues)]

    vx, vy = float(primary_vector[0]), float(primary_vector[1])

    # Standardize direction so angle is in [0, 180)
    angle_rad = np.arctan2(vy, vx)
    angle_deg = float(np.degrees(angle_rad) % 180.0)

    # Unit direction vector aligned with angle_deg
    rad_norm = np.radians(angle_deg)
    direction = (float(np.cos(rad_norm)), float(np.sin(rad_norm)))

    return center, direction, angle_deg


def compute_max_inbounds_ray_length(
    img_shape: tuple[int, int],
    center: tuple[float, float],
    direction: tuple[float, float],
    desired_half_length: float,
    margin_px: float = 0.0
) -> tuple[float, float]:
    """
    Compute actual forward and backward half-lengths along direction vector (vx, vy)
    from center point (cx, cy) clipped so endpoints do not extend past image boundaries
    plus margin_px (default 0.0 for exact zero-overshoot clipping).

    Parameters
    ----------
    img_shape : tuple[int, int]
        Image shape (height, width).
    center : tuple[float, float]
        (cx, cy) center coordinates.
    direction : tuple[float, float]
        (vx, vy) unit direction vector.
    desired_half_length : float
        Requested half length (e.g. (2.0 * length + 50) / 2.0).
    margin_px : float, default 0.0
        Maximum allowed pixels past image boundaries (0.0 clips strictly to 0..width-1, 0..height-1).

    Returns
    -------
    back_len, fwd_len : tuple[float, float]
        Clipped half-lengths in backward (-vx, -vy) and forward (+vx, +vy) directions.
    """
    h, w = img_shape
    cx, cy = center
    vx, vy = direction

    x_min = -margin_px
    x_max = (w - 1) + margin_px
    y_min = -margin_px
    y_max = (h - 1) + margin_px

    def _max_t_along_ray(dir_x: float, dir_y: float) -> float:
        t_max = desired_half_length

        if dir_x > 1e-7:
            t_max = min(t_max, (x_max - cx) / dir_x)
        elif dir_x < -1e-7:
            t_max = min(t_max, (x_min - cx) / dir_x)

        if dir_y > 1e-7:
            t_max = min(t_max, (y_max - cy) / dir_y)
        elif dir_y < -1e-7:
            t_max = min(t_max, (y_min - cy) / dir_y)

        return max(0.0, float(t_max))

    fwd_len = _max_t_along_ray(vx, vy)
    back_len = _max_t_along_ray(-vx, -vy)

    return back_len, fwd_len


def extract_candidate_cutout(
    data: np.ndarray,
    candidate: dict,
    box_width: int = 100,
    unstitched_lookup: dict[int, dict] | None = None,
    trim_margin_px: float = 15.0,
    max_boundary_margin_px: float = 0.0
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float, float]:
    """
    Extract a rectified streak cutout patch using weighted TLS/PCA line refitting & exact 0-margin boundary ray clipping.

    Returns
    -------
    patch, start_pt, end_pt, angle_deg, final_clipped_length : tuple
        - patch: Rectified cutout patch array.
        - start_pt: (start_x, start_y) extracted segment start point.
        - end_pt: (end_x, end_y) extracted segment end point.
        - angle_deg: Refit line angle in degrees.
        - final_clipped_length: Actual extracted cutout length (back_len + fwd_len).
    """
    cy, cx = candidate["centroid"]
    min_row, min_col, max_row, max_col = candidate.get("bbox_full", candidate.get("bbox_local"))
    
    stitched_from = candidate.get("stitched_from", [])
    n_fragments = candidate.get("n_fragments", len(stitched_from) if stitched_from else 1)
    
    detected_length = float(candidate.get("axis_major_length", float(np.hypot(max_row - min_row, max_col - min_col))))

    if n_fragments > 1 and stitched_from:
        fragment_centroids = []
        fragment_weights = []
        
        for lbl in stitched_from:
            if unstitched_lookup and lbl in unstitched_lookup:
                frag = unstitched_lookup[lbl]
                f_cy, f_cx = frag["centroid"]
                f_area = frag.get("area", 1.0)
                fragment_centroids.append((f_cx, f_cy))
                fragment_weights.append(f_area)
            else:
                fragment_centroids.append((cx, cy))
                fragment_weights.append(candidate.get("area", 1.0))
                
        if len(fragment_centroids) > 1:
            center, (vx, vy), angle_deg = fit_weighted_tls_line(fragment_centroids, fragment_weights)
            center_x, center_y = center
        else:
            dy = max_row - min_row
            dx = max_col - min_col
            angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            rad = np.radians(angle_deg)
            vx, vy = float(np.cos(rad)), float(np.sin(rad))
            center_x, center_y = cx, cy

        desired_target_length = 2.0 * detected_length + (80.0 - 2.0 * trim_margin_px)
    else:
        dy = max_row - min_row
        dx = max_col - min_col
        angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        rad = np.radians(angle_deg)
        vx, vy = float(np.cos(rad)), float(np.sin(rad))
        center_x, center_y = cx, cy
        desired_target_length = 2.0 * detected_length + 80.0

    desired_half_len = desired_target_length / 2.0

    # Clip endpoints so they do not extend past image boundaries (margin_px=0.0)
    back_len, fwd_len = compute_max_inbounds_ray_length(
        data.shape, (center_x, center_y), (vx, vy), desired_half_len, margin_px=max_boundary_margin_px
    )

    start_pt = (center_x - back_len * vx, center_y - back_len * vy)
    end_pt = (center_x + fwd_len * vx, center_y + fwd_len * vy)
    final_clipped_length = back_len + fwd_len

    h_img, w_img = data.shape
    try:
        patch, _ = extract_streak_patch(data, start_pt, end_pt, box_width=box_width, output_length=int(final_clipped_length))
    except Exception:
        r_s = max(0, int(min_row - 5))
        r_e = min(h_img, int(max_row + 5))
        c_s = max(0, int(min_col - 5))
        c_e = min(w_img, int(max_col + 5))
        patch = data[r_s:r_e, c_s:c_e]

    return patch, start_pt, end_pt, angle_deg, final_clipped_length


def normalize_data(profile: np.ndarray) -> np.ndarray:
    """Normalize profile array to range [0, 1]."""
    p_min = np.min(profile)
    p_range = np.max(profile) - p_min
    if p_range == 0:
        return np.zeros_like(profile)
    return (profile - p_min) / p_range


def gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """1D Gaussian model."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_peak(median_profile: np.ndarray, window_half_width: int = 13) -> tuple[float | None, ...]:
    """Fit Gaussian to profile peak."""
    x = np.arange(len(median_profile))
    peak_idx = np.argmax(median_profile)

    start = max(0, peak_idx - window_half_width)
    end = min(len(median_profile), peak_idx + window_half_width)
    x_fit = x[start:end]
    y_fit = median_profile[start:end]

    if len(x_fit) < 3:
        return None, None, None, None, None

    a0 = np.max(y_fit)
    mu0 = float(x_fit[np.argmax(y_fit)])
    sigma0 = 5.0

    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[a0, mu0, sigma0], maxfev=1000)
        a, mu, sigma = popt
        fit_curve = gaussian(x_fit, *popt)
        return float(a), float(mu), float(abs(sigma)), fit_curve, x_fit
    except RuntimeError:
        return None, None, None, None, None


def width_estimation(streak_region: np.ndarray, n_sigma: float = 2.0, psf_fwhm: float = 6.79) -> dict | None:
    """
    Estimate streak cross-section width via Gaussian fit and compute width_vs_psf_ratio.

    Parameters
    ----------
    streak_region : np.ndarray
        Cutout image of streak region.
    n_sigma : float, default 2.0
        Sigma multiplier for width integration bounds.
    psf_fwhm : float, default 6.79
        Known point spread function FWHM in pixels (ILMT reference PSF FWHM = 6.79 px).

    Returns
    -------
    dict | None
        Fit results dictionary containing 'width', 'fitted_width_px', 'width_vs_psf_ratio',
        'lower_bound', 'upper_bound', 'sigma', or None if fit fails/invalid.
    """
    if streak_region.size == 0:
        return None

    profile = np.median(streak_region, axis=1)
    norm_prof = normalize_data(profile)

    a, mu, sigma, fit_curve, x_fit = fit_gaussian_to_peak(norm_prof, window_half_width=13)

    if sigma is not None and 0 < sigma < 23:
        lower_bound = int(mu - n_sigma * sigma)
        upper_bound = int(mu + n_sigma * sigma)

        lower_bound = max(lower_bound, 0)
        upper_bound = min(upper_bound, streak_region.shape[0] - 1)

        width = upper_bound - lower_bound
        fitted_fwhm_px = float(2.355 * sigma)
        width_vs_psf_ratio = float(fitted_fwhm_px / psf_fwhm) if psf_fwhm > 0 else 0.0

        return {
            "profile": norm_prof,
            "amplitude": a,
            "mean": mu,
            "sigma": sigma,
            "fitted_width_px": fitted_fwhm_px,
            "width_vs_psf_ratio": width_vs_psf_ratio,
            "fit_curve": fit_curve,
            "x_fit": x_fit,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "width": width
        }
    else:
        return None


def mask_star(img: np.ndarray) -> np.ndarray:
    """
    Mask bright stars/pixels exceeding 95th percentile per row with row median.

    Ported unchanged from notebook.
    """
    corrected_img = np.array(img, copy=True)
    for i in range(corrected_img.shape[0]):
        row = corrected_img[i, :]
        percentile_95 = np.percentile(row, 95)
        mask = row > percentile_95
        corrected_img[i, mask] = percentile_95
    return corrected_img


def calculate_error(image: np.ndarray, gain: float = 4.0, rdnoise: float = 6.0) -> np.ndarray:
    """
    Calculate per-pixel noise (std dev) in ADU including shot noise and read noise.

    Parameters
    ----------
    image : np.ndarray
        Input image patch (ADU).
    gain : float, default 4.0
        Gain in e-/ADU (default from notebook calculate_error).
    rdnoise : float, default 6.0
        Read noise in e- (default from notebook calculate_error).

    Returns
    -------
    std_adu : np.ndarray
        Per-pixel error standard deviation map in ADU.
    """
    signal_adu = np.clip(image, 0, None)
    signal_e = gain * signal_adu
    variance_e = signal_e + (rdnoise ** 2)
    std_adu = np.sqrt(variance_e) / gain
    return std_adu


def compute_signal_and_snr(masked_image: np.ndarray, sky: float | np.ndarray, gain: float = 4.0, rdnoise: float = 6.0) -> dict:
    """
    Compute total integrated signal, noise, total flux (ADU), and SNR from a star-masked image.

    Parameters
    ----------
    masked_image : np.ndarray
        Cutout image with stars masked.
    sky : float | np.ndarray
        Sky background level. Accepts a scalar float (from sigma_clip) or a 2D map slice matching masked_image.shape.
    gain : float, default 4.0
        Gain in e-/ADU.
    rdnoise : float, default 6.0
        Read noise in e-.

    Returns
    -------
    dict
        Dictionary of photometric estimates: signal_e, variance_e, noise_e, noise_adu, flux_adu, snr.
    """
    img = np.array(masked_image, dtype=np.float32)
    sky_val = np.array(sky, dtype=np.float32)

    signal_adu = img - sky_val
    signal_e = np.sum(signal_adu * gain)

    error_map = calculate_error(masked_image, gain=gain, rdnoise=rdnoise)
    variance_e = np.sum((error_map ** 2) * (gain ** 2))
    noise_e = np.sqrt(variance_e)
    noise_adu = noise_e / gain

    flux_adu = float(np.sum(signal_adu))
    snr = float(flux_adu / noise_adu) if noise_adu != 0 else 0.0

    return {
        "signal_e": float(signal_e),
        "variance_e": float(variance_e),
        "noise_e": float(noise_e),
        "noise_adu": float(noise_adu),
        "flux_adu": flux_adu,
        "snr": snr
    }


def extract_candidate_features(
    data: np.ndarray,
    candidate: dict,
    bg: float | np.ndarray,
    std: float | np.ndarray,
    angle_offset_deg: float = 0.0,
    box_width: int = 100,
    gain: float = 4.0,
    rdnoise: float = 6.0,
    psf_fwhm: float = 6.79,
    unstitched_lookup: dict[int, dict] | None = None
) -> dict:
    """
    Assemble the complete Section 6 feature vector for a candidate blob.

    Parameters
    ----------
    data : np.ndarray
        Full image or tile array.
    candidate : dict
        Candidate dictionary from find_blob_candidates.
    bg : float | np.ndarray
        Background estimate (scalar or 2D map).
    std : float | np.ndarray
        Background RMS estimate (scalar or 2D map).
    angle_offset_deg : float, default 0.0
        Radon angle calibration offset in degrees.
    box_width : int, default 100
        Cutout box width in pixels.
    gain : float, default 4.0
        Detector gain e-/ADU.
    rdnoise : float, default 6.0
        Read noise e-.
    psf_fwhm : float, default 6.79
        Known PSF FWHM in pixels.
    unstitched_lookup : dict[int, dict] | None, default None
        Optional lookup dictionary for unstitched candidate fragments.

    Returns
    -------
    features : dict
        Section 6 feature vector dictionary for the candidate.
    """
    from detect import radon_features

    cy, cx = candidate["centroid"]
    min_row, min_col, max_row, max_col = candidate.get("bbox_local", candidate.get("bbox_full"))

    # Extract rectangular crop around candidate bounding box for Radon features
    h, w = data.shape
    r_start = max(0, min_row - 5)
    r_end = min(h, max_row + 5)
    c_start = max(0, min_col - 5)
    c_end = min(w, max_col + 5)
    cutout = data[r_start:r_end, c_start:c_end]

    # Compute Radon features
    rf = radon_features(cutout, angle_offset_deg=angle_offset_deg)

    # Extract rectified streak cutout using integrated refit & exact 0-margin boundary clipping
    patch, start_pt, end_pt, refit_angle_deg, target_len = extract_candidate_cutout(
        data, candidate, box_width=box_width, unstitched_lookup=unstitched_lookup, trim_margin_px=15.0, max_boundary_margin_px=0.0
    )

    # Fit width
    w_fit = width_estimation(patch, n_sigma=2.0, psf_fwhm=psf_fwhm)
    if w_fit is not None:
        fitted_width_px = w_fit["fitted_width_px"]
        width_vs_psf_ratio = w_fit["width_vs_psf_ratio"]
    else:
        fitted_width_px = np.nan
        width_vs_psf_ratio = np.nan

    # Mask stars and calculate flux & SNR
    masked_patch = mask_star(patch)

    if isinstance(bg, np.ndarray):
        try:
            sky_patch, _ = extract_streak_patch(bg, start_pt, end_pt, box_width=box_width, output_length=int(target_len))
        except Exception:
            sky_patch = bg[r_start:r_end, c_start:c_end]
        sky_val = sky_patch
    else:
        sky_val = bg

    photo = compute_signal_and_snr(masked_patch, sky=sky_val, gain=gain, rdnoise=rdnoise)

    feat = {
        "candidate_id": candidate.get("label", 0),
        "centroid_y": cy,
        "centroid_x": cx,
        "radon_snr": rf["radon_snr"],
        "fwhm_rho": rf["fwhm_rho"],
        "peak_angle_deg": rf["peak_angle_deg"],
        "eccentricity": candidate["eccentricity"],
        "axis_major_length": candidate["axis_major_length"],
        "axis_minor_length": candidate["axis_minor_length"],
        "mean_intensity": candidate["mean_intensity"],
        "max_intensity": candidate["max_intensity"],
        "fitted_width_px": fitted_width_px,
        "width_vs_psf_ratio": width_vs_psf_ratio,
        "flux_snr": photo["snr"],
        "total_flux_adu": photo["flux_adu"],
    }
    return feat
