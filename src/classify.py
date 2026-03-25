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
    }


def classify_layer(features, class_centroids=None, shared_inv_cov=None):
    """Classify a first layer based on extracted features.

    Uses LDA-style Mahalanobis distance with a shared (pooled) within-class
    covariance matrix.  This is more robust to distribution shift than
    per-class covariances, which can overfit when training data is limited.

    Feature vector (7 features chosen for cross-dataset generalization):
        [fill_density, gradient_entropy, gradient_dir_kurtosis,
         edge_to_gradient_ratio, spectral_energy_ratio,
         edge_density_cv, roi_coverage]

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
        'edge_density_cv', 'roi_coverage',
    ]

    if class_centroids is None:
        # Per-class centroids computed from combined labeled data
        # (data + data2) with auto-masking
        class_centroids = {
            'optimal': np.array([
                0.0404, 4.5285, 10.0473, 0.1615, 0.9468, 0.8981, 0.2699]),
            'under_extruded': np.array([
                0.0619, 4.4120,  9.4922, 0.2475, 0.9245, 0.6712, 0.2968]),
            'over_extruded': np.array([
                0.0791, 4.6789,  7.4641, 0.3165, 0.9406, 0.6132, 0.2842]),
        }

    if shared_inv_cov is None:
        # Pooled within-class inverse covariance (LDA-style),
        # computed from combined labeled data with regularization
        shared_inv_cov = np.array([
            [ 9.41484403e+03, -9.10312621e-02,  3.12987826e-02,
             -2.34028216e+03, -1.93211219e+00,  4.05941386e+00,
             -1.99148419e+00],
            [-9.10312621e-02,  5.48355770e+00,  2.08271252e-01,
             -4.06745626e-01, -6.05945727e-02,  6.82930999e-01,
             -3.13900711e+00],
            [ 3.12987826e-02,  2.08271252e-01,  4.91479287e-02,
              1.26093229e-01,  1.11349857e+00,  2.61236621e-02,
              1.37410256e-01],
            [-2.34028216e+03, -4.06745626e-01,  1.26093229e-01,
              6.40160087e+02, -8.96707068e+00,  1.62214188e+01,
             -9.01182050e+00],
            [-1.93211219e+00, -6.05945727e-02,  1.11349857e+00,
             -8.96707068e+00,  2.84350939e+02, -8.75428957e+00,
              2.52217428e+01],
            [ 4.05941386e+00,  6.82930999e-01,  2.61236621e-02,
              1.62214188e+01, -8.75428957e+00,  1.24424613e+01,
             -3.75886416e+00],
            [-1.99148419e+00, -3.13900711e+00,  1.37410256e-01,
             -9.01182050e+00,  2.52217428e+01, -3.75886416e+00,
              5.02935104e+01],
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
