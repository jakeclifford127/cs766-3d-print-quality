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

    return {
        'fill_density': fill_density,
        'gradient_entropy': grad_entropy,
        'gradient_mean': grad_stats['mean'],
        'gradient_std': grad_stats['std'],
        'gradient_median': grad_stats['median'],
        'high_gradient_ratio': grad_stats['high_gradient_ratio'],
        'dominant_frequency': dominant_freq,
        'spectral_energy_ratio': spectral_ratio,
    }


def classify_layer(features, centroids=None):
    """Classify a first layer based on extracted features.

    Uses a nearest-centroid approach: computes the Euclidean distance (in
    normalized feature space) from the sample to each class centroid and
    picks the closest one.  Centroids are derived from the labeled training
    data collected for this project.

    Feature vector used:
        [fill_density, gradient_entropy, gradient_mean,
         gradient_std, spectral_energy_ratio]

    Args:
        features: dict from extract_features()
        centroids: dict mapping label -> feature vector, or None for defaults

    Returns:
        label: string — 'optimal', 'under_extruded', or 'over_extruded'
        confidence: float 0-1
    """
    # Feature keys used (order matters — must match centroids)
    feat_keys = [
        'fill_density', 'gradient_entropy', 'gradient_mean',
        'gradient_std', 'spectral_energy_ratio',
    ]

    # Default centroids (mean feature values from our labeled dataset)
    # and per-feature std for normalization
    if centroids is None:
        centroids = {
            'optimal':        np.array([0.0444, 4.6278, 27.034, 58.531, 0.9514]),
            'under_extruded': np.array([0.0567, 4.3267, 41.494, 63.027, 0.9066]),
            'over_extruded':  np.array([0.0852, 4.6700, 36.302, 68.405, 0.9054]),
        }

    # Pooled std for normalization (so each feature contributes equally)
    feat_std = np.array([0.0475, 0.4750, 17.60, 27.50, 0.0930])

    # Build sample vector
    sample = np.array([features[k] for k in feat_keys])

    # Normalized distances to each centroid
    distances = {}
    for label, centroid in centroids.items():
        diff = (sample - centroid) / (feat_std + 1e-9)
        distances[label] = np.sqrt(np.sum(diff ** 2))

    # Pick the nearest centroid
    best_label = min(distances, key=distances.get)
    best_dist = distances[best_label]

    # Confidence: inverse of distance (closer = more confident)
    # Scale so distance=0 → confidence=1, distance=5 → confidence≈0.37
    confidence = np.exp(-best_dist / 3.0)

    return best_label, float(confidence)
