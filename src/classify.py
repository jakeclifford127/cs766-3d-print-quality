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


def classify_layer(features, class_params=None):
    """Classify a first layer based on extracted features.

    Uses Mahalanobis distance to each class centroid.  Unlike simple
    Euclidean distance, Mahalanobis accounts for the covariance structure
    within each class — features that vary a lot within a class contribute
    less to the distance, while tightly clustered features contribute more.

    Feature vector used:
        [fill_density, gradient_entropy, gradient_mean,
         gradient_std, spectral_energy_ratio]

    Args:
        features: dict from extract_features()
        class_params: dict mapping label -> (mean_vector, inv_covariance),
                      or None for defaults derived from our training set

    Returns:
        label: string — 'optimal', 'under_extruded', or 'over_extruded'
        confidence: float 0-1
    """
    feat_keys = [
        'fill_density', 'gradient_entropy', 'gradient_mean',
        'gradient_std', 'spectral_energy_ratio',
    ]

    if class_params is None:
        # Per-class means and inverse covariance matrices
        # computed from labeled training data with Otsu auto-masking
        class_params = {
            'optimal': (
                np.array([0.0409, 4.5164, 29.073, 54.837, 0.9406]),
                np.array([[ 4.43242729e+03,  9.03564392e+01, -1.12049314e+01,
                             4.46490972e+00, -1.14005490e+03],
                           [ 9.03564392e+01,  7.84682507e+00, -3.56522409e-01,
                             1.77720192e-01, -2.93445625e+01],
                           [-1.12049314e+01, -3.56522409e-01,  4.42495417e-02,
                            -1.70554516e-02,  3.67157850e+00],
                           [ 4.46490972e+00,  1.77720192e-01, -1.70554516e-02,
                             9.33815009e-03, -1.57208702e+00],
                           [-1.14005490e+03, -2.93445625e+01,  3.67157850e+00,
                            -1.57208702e+00,  7.42984342e+02]])
            ),
            'under_extruded': (
                np.array([0.0482, 4.2708, 38.728, 67.093, 0.9120]),
                np.array([[ 3.01226269e+03, -5.04672081e+01, -4.31582686e+00,
                             1.82398943e-03,  2.06656507e+01],
                           [-5.04672081e+01,  4.43067037e+00,  6.97222621e-02,
                             2.46198169e-02, -1.78832631e+00],
                           [-4.31582686e+00,  6.97222621e-02,  9.77690212e-03,
                            -1.46630636e-03,  1.24631940e-02],
                           [ 1.82398943e-03,  2.46198169e-02, -1.46630636e-03,
                             3.74828928e-03,  1.85517873e-01],
                           [ 2.06656507e+01, -1.78832631e+00,  1.24631940e-02,
                             1.85517873e-01,  1.07862519e+02]])
            ),
            'over_extruded': (
                np.array([0.0761, 4.5512, 38.395, 67.046, 0.9341]),
                np.array([[ 3.61214737e+02,  5.58444414e+00, -5.17198242e-01,
                             2.09810438e-01,  5.36813320e+01],
                           [ 5.58444414e+00,  7.92013161e+00, -9.68245107e-02,
                             1.00333557e-01,  8.54309499e+00],
                           [-5.17198242e-01, -9.68245107e-02,  8.17645208e-03,
                            -3.52094553e-03,  2.64935981e-01],
                           [ 2.09810438e-01,  1.00333557e-01, -3.52094553e-03,
                             2.55725079e-03,  2.25721580e-02],
                           [ 5.36813320e+01,  8.54309499e+00,  2.64935981e-01,
                             2.25721580e-02,  3.35210551e+02]])
            ),
        }

    # Per-class weights (prior bias).  A lower weight makes a class
    # "easier" to match by scaling its distance down.  Determined via
    # grid search over the training set — should be validated on
    # held-out data to confirm generalization.
    class_weights = {
        'optimal':        0.80,
        'under_extruded': 1.40,
        'over_extruded':  1.20,
    }

    # Build sample vector
    sample = np.array([features[k] for k in feat_keys])

    # Mahalanobis distance to each class, scaled by class weight
    distances = {}
    for label, (mean_vec, inv_cov) in class_params.items():
        diff = sample - mean_vec
        # Mahalanobis: sqrt( (x-mu)^T * Sigma^{-1} * (x-mu) )
        raw_dist = np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))
        distances[label] = raw_dist * class_weights[label]

    # Pick the nearest class
    best_label = min(distances, key=distances.get)
    best_dist = distances[best_label]

    # Confidence: inverse of Mahalanobis distance
    confidence = np.exp(-best_dist / 5.0)

    return best_label, float(confidence)
