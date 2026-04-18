import numpy as np
from scipy.fft import fft2, fftshift


# ---------------------------------------------------------------------------
# Fill Density Metric
# ---------------------------------------------------------------------------

def compute_fill_density(edges, roi_mask):
    """Compute the ratio of edge pixels to total ROI area.

    Args:
        edges: 2D binary edge map (0 or 255) from Canny
        roi_mask: 2D binary mask (0 or 1) defining the print region

    Returns:
        fill_density: float ratio (edge pixels inside ROI / total ROI pixels)
    """
    roi_area = np.sum(roi_mask)
    if roi_area == 0:
        return 0.0

    edge_binary = (edges > 0).astype(np.uint8)
    edge_in_roi = np.sum(edge_binary * roi_mask)

    return edge_in_roi / roi_area


# ---------------------------------------------------------------------------
# Gradient Direction Analysis
# ---------------------------------------------------------------------------

def compute_gradient_histogram(magnitude, direction, roi_mask, num_bins=36):
    """Compute histogram of gradient directions within the ROI.

    Under-extruded layers tend to have strong edges perpendicular to
    extrusion direction (gaps between lines). Over-extruded layers show
    edges parallel to extrusion with compressed spacing.

    Args:
        magnitude: 2D gradient magnitude array
        direction: 2D gradient direction array (radians)
        roi_mask: 2D binary ROI mask
        num_bins: number of angle bins (default 36 = 10 degrees each)

    Returns:
        hist: 1D array of weighted gradient direction counts
        bin_edges: angle bin edges in degrees
    """
    # Only consider pixels inside the ROI with significant gradient
    valid = (roi_mask > 0) & (magnitude > np.percentile(magnitude[roi_mask > 0], 25))

    angles_deg = np.degrees(direction[valid]) % 180  # map to 0-180 range
    weights = magnitude[valid]

    hist, bin_edges = np.histogram(
        angles_deg, bins=num_bins, range=(0, 180), weights=weights
    )

    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist, bin_edges


def gradient_direction_entropy(magnitude, direction, roi_mask):
    """Compute entropy of gradient directions as a uniformity measure.

    High entropy = edges in all directions (more uniform / well-fused).
    Low entropy = edges concentrated in few directions (line gaps or ridges).

    Args:
        magnitude: 2D gradient magnitude
        direction: 2D gradient direction (radians)
        roi_mask: 2D binary ROI mask

    Returns:
        entropy: float, Shannon entropy of the gradient direction histogram
    """
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask)

    # Shannon entropy (ignore zero bins)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return entropy


def gradient_magnitude_stats(magnitude, roi_mask):
    """Compute statistics of gradient magnitudes within the ROI.

    Args:
        magnitude: 2D gradient magnitude
        roi_mask: 2D binary ROI mask

    Returns:
        dict with mean, std, median, and high_gradient_ratio
    """
    vals = magnitude[roi_mask > 0]
    if len(vals) == 0:
        return {'mean': 0, 'std': 0, 'median': 0, 'high_gradient_ratio': 0}

    threshold = np.percentile(vals, 75)
    return {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'median': np.median(vals),
        'high_gradient_ratio': np.sum(vals > threshold) / len(vals)
    }


def gradient_direction_kurtosis(magnitude, direction, roi_mask):
    """Compute kurtosis of the gradient direction histogram.

    Over-extruded layers tend to have more uniform gradient directions
    (lower kurtosis) while under-extruded layers show sharper directional
    peaks from line gaps (higher kurtosis).

    Args:
        magnitude: 2D gradient magnitude
        direction: 2D gradient direction (radians)
        roi_mask: 2D binary ROI mask

    Returns:
        kurtosis: float, excess kurtosis of the direction histogram
    """
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask)
    mean = np.mean(hist)
    std = np.std(hist) + 1e-8
    return float(np.mean(((hist - mean) / std) ** 4))


# ---------------------------------------------------------------------------
# Edge Spatial Analysis
# ---------------------------------------------------------------------------

def compute_edge_density_cv(edges, roi_mask, num_patches=8):
    """Compute coefficient of variation of edge density across patches.

    Measures texture regularity: well-extruded layers have uniform edge
    density, while defects create uneven patches of high/low density.

    Args:
        edges: 2D binary edge map (0 or 255)
        roi_mask: 2D binary ROI mask
        num_patches: number of patches along each axis

    Returns:
        cv: float, coefficient of variation (std/mean) of patch densities
    """
    edge_in_roi = ((edges > 0).astype(np.uint8)) * roi_mask
    h, w = edges.shape
    patch_size = max(h, w) // num_patches

    if patch_size == 0:
        return 0.0

    densities = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch_roi = roi_mask[i:i + patch_size, j:j + patch_size]
            patch_edge = edge_in_roi[i:i + patch_size, j:j + patch_size]
            area = np.sum(patch_roi)
            if area > patch_size * patch_size * 0.1:
                densities.append(np.sum(patch_edge) / area)

    if not densities or np.mean(densities) == 0:
        return 0.0
    return float(np.std(densities) / np.mean(densities))


def compute_edge_to_gradient_ratio(edges, magnitude, roi_mask):
    """Compute ratio of edge pixels to high-gradient pixels inside ROI.

    Over-extruded layers tend to have a higher fraction of strong
    gradients that survive as edges, while under-extruded layers have
    weaker, more fragmented edges relative to their gradient field.

    Args:
        edges: 2D binary edge map (0 or 255)
        magnitude: 2D gradient magnitude
        roi_mask: 2D binary ROI mask

    Returns:
        ratio: float, edge pixels / high-gradient pixels
    """
    edge_in_roi = ((edges > 0).astype(np.uint8)) * roi_mask
    total_edge = np.sum(edge_in_roi)

    roi_mag = magnitude[roi_mask > 0]
    if len(roi_mag) == 0:
        return 0.0

    high_grad_count = np.sum(roi_mag > np.percentile(roi_mag, 75))
    return float(total_edge / (high_grad_count + 1))


def compute_roi_coverage(roi_mask):
    """Compute fraction of image covered by the ROI.

    Args:
        roi_mask: 2D binary ROI mask

    Returns:
        coverage: float in [0, 1]
    """
    return float(np.sum(roi_mask) / roi_mask.size)


# ---------------------------------------------------------------------------
# Fourier-Domain Analysis
# ---------------------------------------------------------------------------

def compute_frequency_profile(grayscale, roi_mask):
    """Analyze spatial frequencies of extrusion lines using FFT.

    Regular line spacing produces strong peaks in the frequency domain.
    Missing or irregular spacing indicates defects.

    Args:
        grayscale: 2D grayscale image (float64)
        roi_mask: 2D binary ROI mask

    Returns:
        radial_profile: 1D array of radially-averaged power spectrum
        dominant_frequency: the frequency with highest power (excluding DC)
        spectral_energy_ratio: energy in dominant band vs total energy
    """
    # Apply mask (zero out background)
    masked = grayscale * roi_mask

    # 2D FFT
    f_transform = fft2(masked)
    f_shifted = fftshift(f_transform)
    power_spectrum = np.abs(f_shifted) ** 2

    # Radial average of power spectrum
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    for ri in range(max_r):
        ring = power_spectrum[r == ri]
        if len(ring) > 0:
            radial_profile[ri] = np.mean(ring)

    # Skip DC component (index 0)
    if len(radial_profile) > 1:
        spectrum_no_dc = radial_profile[1:]
        dominant_frequency = np.argmax(spectrum_no_dc) + 1
        total_energy = np.sum(spectrum_no_dc)
        if total_energy > 0:
            # Energy in a band around the dominant frequency
            band_start = max(0, dominant_frequency - 3)
            band_end = min(len(spectrum_no_dc), dominant_frequency + 3)
            band_energy = np.sum(spectrum_no_dc[band_start:band_end])
            spectral_energy_ratio = band_energy / total_energy
        else:
            spectral_energy_ratio = 0.0
    else:
        dominant_frequency = 0
        spectral_energy_ratio = 0.0

    return radial_profile, dominant_frequency, spectral_energy_ratio


# ---------------------------------------------------------------------------
# Line-Spacing Uniformity
# ---------------------------------------------------------------------------

def detect_line_direction(magnitude, direction, roi_mask):
    """Detect the dominant extrusion line direction using gradient histogram.

    The gradient direction is perpendicular to the line direction, so we
    rotate by 90 degrees to get the actual line orientation.

    Returns:
        (line_angle_deg, confidence): line angle in degrees [0, 180),
            confidence is peak-to-mean ratio of the histogram (higher = clearer direction)
    """
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask, num_bins=36)
    if hist is None or np.sum(hist) == 0:
        return 0.0, 0.0

    # Find peak bin
    peak_bin = np.argmax(hist)
    peak_val = hist[peak_bin]
    mean_val = np.mean(hist)

    # Confidence: how much the peak stands out
    confidence = peak_val / mean_val if mean_val > 0 else 0.0

    # Parabolic interpolation for sub-bin precision
    n = len(hist)
    left = hist[(peak_bin - 1) % n]
    right = hist[(peak_bin + 1) % n]
    denom = 2.0 * (2.0 * peak_val - left - right)
    if abs(denom) > 1e-10:
        offset = (left - right) / denom
    else:
        offset = 0.0

    # Gradient angle in degrees (each bin = 180/36 = 5 degrees)
    grad_angle = (peak_bin + offset) * (180.0 / n)
    # Line direction is perpendicular to gradient direction
    line_angle = (grad_angle + 90.0) % 180.0

    return line_angle, confidence


def sample_perpendicular_profiles(grayscale, roi_mask, line_angle_deg, num_profiles=20):
    """Sample intensity profiles perpendicular to the extrusion line direction.

    Generates scan lines across the ROI oriented perpendicular to the detected
    line direction, sampling grayscale intensity along each.

    Returns:
        list of 1D numpy arrays (intensity profiles), each with >= 20 samples
    """
    h, w = grayscale.shape
    # Perpendicular direction (same as gradient direction)
    perp_rad = np.deg2rad(line_angle_deg)  # line dir; perp to lines = line_dir itself rotated
    # Actually: line_angle is the line direction. We want to scan PERPENDICULAR to lines.
    # Perpendicular to the line direction = the gradient direction = line_angle - 90
    scan_rad = np.deg2rad(line_angle_deg - 90.0)
    dx = np.cos(scan_rad)
    dy = np.sin(scan_rad)

    # Line direction for distributing scan line starting positions
    line_dx = np.cos(np.deg2rad(line_angle_deg))
    line_dy = np.sin(np.deg2rad(line_angle_deg))

    # Find ROI bounding box center
    rows, cols = np.where(roi_mask > 0)
    if len(rows) == 0:
        return []
    cy, cx = np.mean(rows), np.mean(cols)
    roi_h = rows.max() - rows.min()
    roi_w = cols.max() - cols.min()
    max_extent = max(roi_h, roi_w)

    # Distribute scan line origins along the line direction
    offsets = np.linspace(-max_extent / 2, max_extent / 2, num_profiles)
    profiles = []

    for off in offsets:
        # Starting point: center + offset along line direction
        start_y = cy + off * line_dy
        start_x = cx + off * line_dx

        # Sample along the scan (perpendicular) direction
        t_range = np.arange(-max_extent, max_extent + 1)
        ys = (start_y + t_range * dy).astype(int)
        xs = (start_x + t_range * dx).astype(int)

        # Keep only valid, in-ROI pixels
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys, xs = ys[valid], xs[valid]
        in_roi = roi_mask[ys, xs] > 0
        ys, xs = ys[in_roi], xs[in_roi]

        if len(ys) >= 20:
            profiles.append(grayscale[ys, xs].astype(np.float64))

    return profiles


def find_1d_peaks(profile, min_prominence_ratio=0.1):
    """Find peaks in a 1D intensity profile.

    Smooths the profile with a small Gaussian, then detects local maxima
    with sufficient prominence.

    Returns:
        numpy array of peak indices
    """
    # Smooth with 1D Gaussian (sigma=2)
    sigma = 2.0
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(profile, kernel, mode='same')

    # Find local maxima
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Filter by prominence
    profile_range = smoothed.max() - smoothed.min()
    if profile_range < 1e-10:
        return np.array([], dtype=int)

    min_prominence = min_prominence_ratio * profile_range
    strong_peaks = []
    for p in peaks:
        # Find nearest valleys on each side
        left_valley = smoothed[p]
        for j in range(p - 1, -1, -1):
            if smoothed[j] < left_valley:
                left_valley = smoothed[j]
            if j < len(smoothed) - 1 and smoothed[j] > smoothed[j + 1] and j != p:
                break
        right_valley = smoothed[p]
        for j in range(p + 1, len(smoothed)):
            if smoothed[j] < right_valley:
                right_valley = smoothed[j]
            if j > 0 and smoothed[j] > smoothed[j - 1] and j != p:
                break
        prominence = smoothed[p] - (left_valley + right_valley) / 2.0
        if prominence >= min_prominence:
            strong_peaks.append(p)

    return np.array(strong_peaks, dtype=int)


def compute_line_spacing_uniformity(grayscale, magnitude, direction, roi_mask):
    """Compute line-spacing uniformity metric.

    Measures how regular the extrusion line spacing is by:
    1. Detecting the dominant line direction
    2. Sampling intensity profiles perpendicular to lines
    3. Finding peaks (line centers) in each profile
    4. Computing the coefficient of variation of peak spacing

    Returns:
        float in (0, 1]: 1.0 = perfectly uniform spacing, lower = more irregular.
        Returns 0.5 (neutral) if insufficient data.
    """
    roi_area = np.sum(roi_mask)
    if roi_area == 0:
        return 0.5

    # Step 1: Detect line direction
    line_angle, confidence = detect_line_direction(magnitude, direction, roi_mask)
    if confidence < 1.5:
        return 0.5  # No clear line direction

    # Step 2: Sample perpendicular profiles
    profiles = sample_perpendicular_profiles(grayscale, roi_mask, line_angle)
    if len(profiles) == 0:
        return 0.5

    # Step 3: Find peaks and collect spacings
    all_spacings = []
    for prof in profiles:
        peaks = find_1d_peaks(prof)
        if len(peaks) >= 2:
            spacings = np.diff(peaks)
            all_spacings.extend(spacings.tolist())

    # Step 4: Compute uniformity
    if len(all_spacings) < 5:
        return 0.5  # Insufficient data

    spacings_arr = np.array(all_spacings, dtype=np.float64)
    mean_spacing = np.mean(spacings_arr)
    if mean_spacing < 1e-10:
        return 0.5

    cv = np.std(spacings_arr) / mean_spacing
    uniformity = 1.0 / (1.0 + cv)

    return float(uniformity)


# ---------------------------------------------------------------------------
# Hough Transform Line Detection
# ---------------------------------------------------------------------------

def hough_line_transform(edges, roi_mask, rho_res=1.0, theta_res_deg=1.0):
    """Compute the Hough Transform accumulator for line detection.

    Parameterizes lines as rho = x*cos(theta) + y*sin(theta) and
    accumulates votes from edge pixels inside the ROI.

    Args:
        edges: 2D binary edge map (0 or 255) from Canny
        roi_mask: 2D binary ROI mask
        rho_res: rho bin resolution in pixels
        theta_res_deg: theta bin resolution in degrees

    Returns:
        accumulator: 2D vote array (num_rho_bins, num_theta_bins)
        rho_values: 1D array mapping row index to rho in pixels
        theta_values: 1D array of theta values in radians
    """
    h, w = edges.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))

    # Theta bins: 0 to 180 degrees
    thetas = np.deg2rad(np.arange(0, 180, theta_res_deg))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Rho bins: -diag to +diag
    num_rhos = int(2 * diag / rho_res) + 1
    rho_values = np.linspace(-diag, diag, num_rhos)

    # Find edge pixels inside ROI
    ys, xs = np.where((edges > 0) & (roi_mask > 0))
    if len(ys) == 0:
        return np.zeros((num_rhos, len(thetas)), dtype=np.int32), rho_values, thetas

    # Vectorized voting: iterate over theta, vectorize over pixels
    accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int32)
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    for t_idx in range(len(thetas)):
        rhos = xs_f * cos_thetas[t_idx] + ys_f * sin_thetas[t_idx]
        rho_idx = np.round((rhos + diag) / rho_res).astype(np.intp)
        # Clip to valid range
        valid = (rho_idx >= 0) & (rho_idx < num_rhos)
        np.add.at(accumulator, (rho_idx[valid], t_idx), 1)

    return accumulator, rho_values, thetas


def extract_hough_peaks(accumulator, rho_values, theta_values,
                        threshold_ratio=0.15, nms_rho_dist=10,
                        nms_theta_dist=10, max_peaks=50):
    """Extract line peaks from Hough accumulator with non-maximum suppression.

    Args:
        accumulator: 2D vote array from hough_line_transform
        rho_values: 1D array of rho values
        theta_values: 1D array of theta values in radians
        threshold_ratio: minimum vote count as fraction of max
        nms_rho_dist: minimum rho separation between peaks (bins)
        nms_theta_dist: minimum theta separation between peaks (bins)
        max_peaks: maximum number of peaks to return

    Returns:
        list of (rho, theta_rad, votes) tuples, sorted by votes descending
    """
    if accumulator.max() == 0:
        return []

    threshold = threshold_ratio * accumulator.max()

    # Find all cells above threshold
    candidates = []
    above = np.where(accumulator >= threshold)
    for ri, ti in zip(above[0], above[1]):
        candidates.append((ri, ti, accumulator[ri, ti]))

    # Sort by votes descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy NMS
    accepted = []
    for ri, ti, votes in candidates:
        if len(accepted) >= max_peaks:
            break
        # Check if too close to any accepted peak
        too_close = False
        for ari, ati, _ in accepted:
            # Circular theta distance
            dt = abs(ti - ati)
            dt = min(dt, len(theta_values) - dt)
            if abs(ri - ari) <= nms_rho_dist and dt <= nms_theta_dist:
                too_close = True
                break
        if not too_close:
            accepted.append((ri, ti, votes))

    # Convert to (rho, theta, votes)
    peaks = []
    for ri, ti, votes in accepted:
        peaks.append((rho_values[ri], theta_values[ti], int(votes)))

    return peaks


def compute_hough_features(edges, magnitude, direction, roi_mask):
    """Compute Hough Transform-based line detection features.

    Detects lines in the edge map using the Hough Transform, then
    computes 3 scalar features measuring line count, spacing regularity,
    and pattern strength.

    Args:
        edges: 2D binary edge map (0 or 255)
        magnitude: gradient magnitude from Sobel
        direction: gradient direction from Sobel (radians)
        roi_mask: 2D binary ROI mask

    Returns:
        dict with keys: hough_line_count, hough_spacing_regularity,
                        hough_peak_ratio
    """
    neutral = {
        'hough_line_count': 0.0,
        'hough_spacing_regularity': 0.5,
        'hough_peak_ratio': 0.0,
    }

    # Check for sufficient edge pixels
    roi_area = np.sum(roi_mask)
    if roi_area < 100:
        return neutral

    edge_count = np.sum((edges > 0) & (roi_mask > 0))
    if edge_count < 10:
        return neutral

    # Run Hough Transform
    accumulator, rho_values, theta_values = hough_line_transform(edges, roi_mask)

    if accumulator.max() == 0:
        return neutral

    # Extract peaks
    peaks = extract_hough_peaks(accumulator, rho_values, theta_values)
    if len(peaks) == 0:
        return neutral

    # Get dominant extrusion direction from existing function
    line_angle, dir_confidence = detect_line_direction(magnitude, direction, roi_mask)
    # Convert line angle to Hough theta (perpendicular to line direction)
    dominant_theta_rad = np.deg2rad(line_angle - 90.0) % np.pi

    # --- Feature 1: hough_line_count ---
    # Count lines within ±15 degrees of dominant direction
    parallel_tolerance = np.deg2rad(15.0)
    parallel_peaks = []
    for rho, theta, votes in peaks:
        # Circular angular distance
        dt = abs(theta - dominant_theta_rad)
        dt = min(dt, np.pi - dt)
        if dt <= parallel_tolerance:
            parallel_peaks.append((rho, theta, votes))

    # Normalize by ROI scale
    h, w = edges.shape
    roi_diag = np.sqrt(h**2 + w**2)
    hough_line_count = len(parallel_peaks) / (roi_diag / 100.0)

    # --- Feature 2: hough_spacing_regularity ---
    if len(parallel_peaks) >= 3:
        # Sort parallel lines by rho and compute consecutive spacings
        sorted_rhos = sorted([p[0] for p in parallel_peaks])
        spacings = np.diff(sorted_rhos)
        spacings = spacings[spacings > 0]  # remove duplicates

        if len(spacings) >= 2:
            mean_sp = np.mean(spacings)
            if mean_sp > 1e-8:
                cv = np.std(spacings) / mean_sp
                hough_spacing_regularity = 1.0 / (1.0 + cv)
            else:
                hough_spacing_regularity = 0.5
        else:
            hough_spacing_regularity = 0.5
    else:
        hough_spacing_regularity = 0.5

    # --- Feature 3: hough_peak_ratio ---
    # Sum accumulator votes per theta to get a 1D theta profile
    theta_profile = accumulator.sum(axis=0).astype(np.float64)
    total_energy = theta_profile.sum()

    if total_energy > 0:
        # Find peak theta and sum energy in ±5 degree window
        peak_theta_idx = np.argmax(theta_profile)
        window = 5  # degrees (since theta_res = 1 degree)
        n_thetas = len(theta_values)
        peak_energy = 0.0
        for di in range(-window, window + 1):
            idx = (peak_theta_idx + di) % n_thetas
            peak_energy += theta_profile[idx]
        hough_peak_ratio = peak_energy / total_energy
    else:
        hough_peak_ratio = 0.0

    return {
        'hough_line_count': float(hough_line_count),
        'hough_spacing_regularity': float(hough_spacing_regularity),
        'hough_peak_ratio': float(hough_peak_ratio),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def extract_features(grayscale, edges, magnitude, direction, roi_mask):
    """Extract all features for classification.

    Args:
        grayscale: 2D grayscale image
        edges: binary edge map from Canny
        magnitude: gradient magnitude from Sobel
        direction: gradient direction from Sobel
        roi_mask: binary ROI mask

    Returns:
        dict of all feature values
    """
    fill_density = compute_fill_density(edges, roi_mask)
    grad_entropy = gradient_direction_entropy(magnitude, direction, roi_mask)
    grad_stats = gradient_magnitude_stats(magnitude, roi_mask)
    _, dominant_freq, spectral_ratio = compute_frequency_profile(grayscale, roi_mask)
    grad_dir_kurtosis = gradient_direction_kurtosis(magnitude, direction, roi_mask)
    edge_density_cv = compute_edge_density_cv(edges, roi_mask)
    edge_to_grad = compute_edge_to_gradient_ratio(edges, magnitude, roi_mask)
    roi_cov = compute_roi_coverage(roi_mask)
    line_uniformity = compute_line_spacing_uniformity(
        grayscale, magnitude, direction, roi_mask)
    hough_feats = compute_hough_features(edges, magnitude, direction, roi_mask)

    return {
        'fill_density': fill_density,
        'gradient_entropy': grad_entropy,
        'gradient_mean': grad_stats['mean'],
        'gradient_std': grad_stats['std'],
        'gradient_median': grad_stats['median'],
        'high_gradient_ratio': grad_stats['high_gradient_ratio'],
        'dominant_frequency': dominant_freq,
        'spectral_energy_ratio': spectral_ratio,
        'gradient_dir_kurtosis': grad_dir_kurtosis,
        'edge_density_cv': edge_density_cv,
        'edge_to_gradient_ratio': edge_to_grad,
        'roi_coverage': roi_cov,
        'line_spacing_uniformity': line_uniformity,
        'hough_line_count': hough_feats['hough_line_count'],
        'hough_spacing_regularity': hough_feats['hough_spacing_regularity'],
        'hough_peak_ratio': hough_feats['hough_peak_ratio'],
    }


def classify_layer(features, class_centroids=None, shared_inv_cov=None):
    """Classify a first layer based on extracted features.

    Uses LDA-style Mahalanobis distance with a shared (pooled) within-class
    covariance matrix.  This is more robust to distribution shift than
    per-class covariances, which can overfit when training data is limited.

    Feature vector (8 features chosen for cross-dataset generalization):
        [fill_density, gradient_entropy, gradient_dir_kurtosis,
         edge_to_gradient_ratio, spectral_energy_ratio,
         edge_density_cv, roi_coverage, line_spacing_uniformity]

    Args:
        features: dict from extract_features()
        class_centroids: dict mapping label -> mean_vector,
                         or None for defaults trained on data+data2
        shared_inv_cov: shared inverse covariance matrix,
                        or None for default

    Returns:
        label: string — 'optimal', 'under_extruded', or 'over_extruded'
        confidence: float 0-1
    """
    feat_keys = [
        'fill_density', 'gradient_entropy', 'gradient_dir_kurtosis',
        'edge_to_gradient_ratio', 'spectral_energy_ratio',
        'edge_density_cv', 'roi_coverage', 'line_spacing_uniformity',
    ]

    if class_centroids is None:
        # Per-class centroids computed from labeled data with auto-masking
        class_centroids = {
            'optimal': np.array([
                0.0409, 4.5164, 8.9199, 0.1634, 0.9406, 0.8548,
                0.2560, 0.5618]),
            'under_extruded': np.array([
                0.0482, 4.2708, 10.4468, 0.1927, 0.9120, 0.6681,
                0.3155, 0.6059]),
            'over_extruded': np.array([
                0.0761, 4.5512, 7.3627, 0.3044, 0.9341, 0.6382,
                0.2928, 0.6366]),
        }

    if shared_inv_cov is None:
        # Pooled within-class inverse covariance (LDA-style),
        # computed from labeled data with regularization
        shared_inv_cov = np.array([
            [ 9.63480620e+01,  1.58707299e-01,  4.10842189e-02,
             -1.46050075e+01,  4.16966067e-01,  2.47085427e+00,
             -8.15433888e-01,  2.37805746e-01],
            [ 1.58707299e-01,  4.76683157e+00,  2.35736001e-01,
              6.34726300e-01,  1.04021284e+00,  7.44802720e-01,
             -2.76440629e+00,  7.48232249e-01],
            [ 4.10842189e-02,  2.35736001e-01,  6.09027322e-02,
              1.64335862e-01,  4.76309669e-01,  9.05561459e-02,
             -1.10472298e-02, -6.29213421e-02],
            [-1.46050075e+01,  6.34726300e-01,  1.64335862e-01,
              4.15909364e+01,  1.66453948e+00,  9.88180244e+00,
             -3.26890416e+00,  9.51919668e-01],
            [ 4.16966067e-01,  1.04021284e+00,  4.76309669e-01,
              1.66453948e+00,  6.99571245e+01, -6.02505609e-01,
              4.98408593e+00,  4.37715308e-01],
            [ 2.47085427e+00,  7.44802720e-01,  9.05561459e-02,
              9.88180244e+00, -6.02505609e-01,  9.71218615e+00,
             -1.21747756e+00,  4.89702496e+00],
            [-8.15433888e-01, -2.76440629e+00, -1.10472298e-02,
             -3.26890416e+00,  4.98408593e+00, -1.21747756e+00,
              2.86262490e+01,  4.66282462e+00],
            [ 2.37805746e-01,  7.48232249e-01, -6.29213421e-02,
              9.51919668e-01,  4.37715308e-01,  4.89702496e+00,
              4.66282462e+00,  6.78464958e+01],
        ])

    # Build sample vector
    sample = np.array([features[k] for k in feat_keys])

    # Mahalanobis distance to each class centroid (no class weights —
    # removed to avoid bias that hurt generalization on new data)
    distances = {}
    for label, mean_vec in class_centroids.items():
        diff = sample - mean_vec
        distances[label] = np.sqrt(np.dot(diff, np.dot(shared_inv_cov, diff)))

    # Pick the nearest class
    best_label = min(distances, key=distances.get)
    best_dist = distances[best_label]

    # Confidence: inverse of Mahalanobis distance
    confidence = np.exp(-best_dist / 5.0)

    return best_label, float(confidence)
