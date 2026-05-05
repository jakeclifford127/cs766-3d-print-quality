import numpy as np
from scipy.fft import fft2, fftshift


def compute_fill_density(edges, roi_mask):
    """edge/ROI ratio."""
    roi_area = np.sum(roi_mask)
    if roi_area == 0:
        return 0.0

    edge_binary = (edges > 0).astype(np.uint8)
    edge_in_roi = np.sum(edge_binary * roi_mask)

    return edge_in_roi / roi_area


def compute_gradient_histogram(magnitude, direction, roi_mask, num_bins=36):
    """Weighted gradient direction histogram."""
    valid = (roi_mask > 0) & (magnitude > np.percentile(magnitude[roi_mask > 0], 25))

    angles_deg = np.degrees(direction[valid]) % 180
    weights = magnitude[valid]

    hist, bin_edges = np.histogram(
        angles_deg, bins=num_bins, range=(0, 180), weights=weights
    )

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist, bin_edges


def gradient_direction_entropy(magnitude, direction, roi_mask):
    """Shannon entropy of gradient dirs."""
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask)

    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return entropy


def gradient_magnitude_stats(magnitude, roi_mask):
    """Mean/std/median of gradient mags."""
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
    """Kurtosis of direction histogram."""
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask)
    mean = np.mean(hist)
    std = np.std(hist) + 1e-8
    return float(np.mean(((hist - mean) / std) ** 4))


def compute_edge_density_cv(edges, roi_mask, num_patches=8):
    """CV of edge density across patches."""
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
    """Edge px / high-gradient px."""
    edge_in_roi = ((edges > 0).astype(np.uint8)) * roi_mask
    total_edge = np.sum(edge_in_roi)

    roi_mag = magnitude[roi_mask > 0]
    if len(roi_mag) == 0:
        return 0.0

    high_grad_count = np.sum(roi_mag > np.percentile(roi_mag, 75))
    return float(total_edge / (high_grad_count + 1))


def compute_roi_coverage(roi_mask):
    """ROI fraction of image."""
    return float(np.sum(roi_mask) / roi_mask.size)


def compute_frequency_profile(grayscale, roi_mask):
    """FFT radial profile + dominant freq."""
    masked = grayscale * roi_mask

    f_transform = fft2(masked)
    f_shifted = fftshift(f_transform)
    power_spectrum = np.abs(f_shifted) ** 2

    # radial average
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

    # skip DC
    if len(radial_profile) > 1:
        spectrum_no_dc = radial_profile[1:]
        dominant_frequency = np.argmax(spectrum_no_dc) + 1
        total_energy = np.sum(spectrum_no_dc)
        if total_energy > 0:
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


# line spacing

def detect_line_direction(magnitude, direction, roi_mask):
    """Find dominant extrusion line angle."""
    hist, _ = compute_gradient_histogram(magnitude, direction, roi_mask, num_bins=36)
    if hist is None or np.sum(hist) == 0:
        return 0.0, 0.0

    peak_bin = np.argmax(hist)
    peak_val = hist[peak_bin]
    mean_val = np.mean(hist)

    confidence = peak_val / mean_val if mean_val > 0 else 0.0

    # parabolic interpolation
    n = len(hist)
    left = hist[(peak_bin - 1) % n]
    right = hist[(peak_bin + 1) % n]
    denom = 2.0 * (2.0 * peak_val - left - right)
    if abs(denom) > 1e-10:
        offset = (left - right) / denom
    else:
        offset = 0.0

    grad_angle = (peak_bin + offset) * (180.0 / n)
    # perpendicular to gradient = line dir
    line_angle = (grad_angle + 90.0) % 180.0

    return line_angle, confidence


def sample_perpendicular_profiles(grayscale, roi_mask, line_angle_deg, num_profiles=20):
    """Sample intensity across lines."""
    h, w = grayscale.shape
    scan_rad = np.deg2rad(line_angle_deg - 90.0)
    dx = np.cos(scan_rad)
    dy = np.sin(scan_rad)

    line_dx = np.cos(np.deg2rad(line_angle_deg))
    line_dy = np.sin(np.deg2rad(line_angle_deg))

    rows, cols = np.where(roi_mask > 0)
    if len(rows) == 0:
        return []
    cy, cx = np.mean(rows), np.mean(cols)
    roi_h = rows.max() - rows.min()
    roi_w = cols.max() - cols.min()
    max_extent = max(roi_h, roi_w)

    offsets = np.linspace(-max_extent / 2, max_extent / 2, num_profiles)
    profiles = []

    for off in offsets:
        start_y = cy + off * line_dy
        start_x = cx + off * line_dx

        t_range = np.arange(-max_extent, max_extent + 1)
        ys = (start_y + t_range * dy).astype(int)
        xs = (start_x + t_range * dx).astype(int)

        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        ys, xs = ys[valid], xs[valid]
        in_roi = roi_mask[ys, xs] > 0
        ys, xs = ys[in_roi], xs[in_roi]

        if len(ys) >= 20:
            profiles.append(grayscale[ys, xs].astype(np.float64))

    return profiles


def find_1d_peaks(profile, min_prominence_ratio=0.1):
    """Find peaks in 1D profile."""
    # smooth w/ gaussian
    sigma = 2.0
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(profile, kernel, mode='same')

    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    profile_range = smoothed.max() - smoothed.min()
    if profile_range < 1e-10:
        return np.array([], dtype=int)

    min_prominence = min_prominence_ratio * profile_range
    strong_peaks = []
    for p in peaks:
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
    """How regular the line spacing is. Returns 0-1."""
    roi_area = np.sum(roi_mask)
    if roi_area == 0:
        return 0.5

    line_angle, confidence = detect_line_direction(magnitude, direction, roi_mask)
    if confidence < 1.5:
        return 0.5

    profiles = sample_perpendicular_profiles(grayscale, roi_mask, line_angle)
    if len(profiles) == 0:
        return 0.5

    all_spacings = []
    for prof in profiles:
        peaks = find_1d_peaks(prof)
        if len(peaks) >= 2:
            spacings = np.diff(peaks)
            all_spacings.extend(spacings.tolist())

    if len(all_spacings) < 5:
        return 0.5

    spacings_arr = np.array(all_spacings, dtype=np.float64)
    mean_spacing = np.mean(spacings_arr)
    if mean_spacing < 1e-10:
        return 0.5

    cv = np.std(spacings_arr) / mean_spacing
    uniformity = 1.0 / (1.0 + cv)

    return float(uniformity)


# hough transform

def hough_line_transform(edges, roi_mask, rho_res=1.0, theta_res_deg=1.0):
    """Hough accumulator for lines."""
    h, w = edges.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))

    thetas = np.deg2rad(np.arange(0, 180, theta_res_deg))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    num_rhos = int(2 * diag / rho_res) + 1
    rho_values = np.linspace(-diag, diag, num_rhos)

    ys, xs = np.where((edges > 0) & (roi_mask > 0))
    if len(ys) == 0:
        return np.zeros((num_rhos, len(thetas)), dtype=np.int32), rho_values, thetas

    accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int32)
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)

    for t_idx in range(len(thetas)):
        rhos = xs_f * cos_thetas[t_idx] + ys_f * sin_thetas[t_idx]
        rho_idx = np.round((rhos + diag) / rho_res).astype(np.intp)
        valid = (rho_idx >= 0) & (rho_idx < num_rhos)
        np.add.at(accumulator, (rho_idx[valid], t_idx), 1)

    return accumulator, rho_values, thetas


def extract_hough_peaks(accumulator, rho_values, theta_values,
                        threshold_ratio=0.15, nms_rho_dist=10,
                        nms_theta_dist=10, max_peaks=50):
    """Extract peaks with NMS."""
    if accumulator.max() == 0:
        return []

    threshold = threshold_ratio * accumulator.max()

    candidates = []
    above = np.where(accumulator >= threshold)
    for ri, ti in zip(above[0], above[1]):
        candidates.append((ri, ti, accumulator[ri, ti]))

    candidates.sort(key=lambda x: x[2], reverse=True)

    # greedy NMS
    accepted = []
    for ri, ti, votes in candidates:
        if len(accepted) >= max_peaks:
            break
        too_close = False
        for ari, ati, _ in accepted:
            dt = abs(ti - ati)
            dt = min(dt, len(theta_values) - dt)
            if abs(ri - ari) <= nms_rho_dist and dt <= nms_theta_dist:
                too_close = True
                break
        if not too_close:
            accepted.append((ri, ti, votes))

    peaks = []
    for ri, ti, votes in accepted:
        peaks.append((rho_values[ri], theta_values[ti], int(votes)))

    return peaks


def compute_hough_features(edges, magnitude, direction, roi_mask):
    """Line count, spacing, peak ratio from hough."""
    neutral = {
        'hough_line_count': 0.0,
        'hough_spacing_regularity': 0.5,
        'hough_peak_ratio': 0.0,
    }

    roi_area = np.sum(roi_mask)
    if roi_area < 100:
        return neutral

    edge_count = np.sum((edges > 0) & (roi_mask > 0))
    if edge_count < 10:
        return neutral

    accumulator, rho_values, theta_values = hough_line_transform(edges, roi_mask)

    if accumulator.max() == 0:
        return neutral

    peaks = extract_hough_peaks(accumulator, rho_values, theta_values)
    if len(peaks) == 0:
        return neutral

    line_angle, dir_confidence = detect_line_direction(magnitude, direction, roi_mask)
    dominant_theta_rad = np.deg2rad(line_angle - 90.0) % np.pi

    # count parallel lines
    parallel_tolerance = np.deg2rad(15.0)
    parallel_peaks = []
    for rho, theta, votes in peaks:
        dt = abs(theta - dominant_theta_rad)
        dt = min(dt, np.pi - dt)
        if dt <= parallel_tolerance:
            parallel_peaks.append((rho, theta, votes))

    h, w = edges.shape
    roi_diag = np.sqrt(h**2 + w**2)
    hough_line_count = len(parallel_peaks) / (roi_diag / 100.0)

    # spacing regularity
    if len(parallel_peaks) >= 3:
        sorted_rhos = sorted([p[0] for p in parallel_peaks])
        spacings = np.diff(sorted_rhos)
        spacings = spacings[spacings > 0]

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

    # peak ratio
    theta_profile = accumulator.sum(axis=0).astype(np.float64)
    total_energy = theta_profile.sum()

    if total_energy > 0:
        peak_theta_idx = np.argmax(theta_profile)
        window = 5
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


# k-means texture segmentation

def compute_local_texture_stats(grayscale, edges, roi_mask, patch_size=7):
    """Per-pixel texture features via integral images."""
    half = patch_size // 2
    H, W = grayscale.shape

    gray_pad = np.pad(grayscale, half, mode='reflect')
    edge_binary = (edges > 0).astype(np.float64)
    edge_pad = np.pad(edge_binary, half, mode='constant', constant_values=0)

    def make_integral(img):
        sat = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
        np.cumsum(img, axis=0, out=sat[1:, 1:])
        np.cumsum(sat[1:, 1:], axis=1, out=sat[1:, 1:])
        return sat

    integral = make_integral(gray_pad)
    integral_sq = make_integral(gray_pad ** 2)
    integral_edge = make_integral(edge_pad)

    ps = patch_size
    count = ps * ps

    local_sum = (integral[ps:H+ps, ps:W+ps]
                 - integral[:H, ps:W+ps]
                 - integral[ps:H+ps, :W]
                 + integral[:H, :W])

    local_sum_sq = (integral_sq[ps:H+ps, ps:W+ps]
                    - integral_sq[:H, ps:W+ps]
                    - integral_sq[ps:H+ps, :W]
                    + integral_sq[:H, :W])

    local_edge_sum = (integral_edge[ps:H+ps, ps:W+ps]
                      - integral_edge[:H, ps:W+ps]
                      - integral_edge[ps:H+ps, :W]
                      + integral_edge[:H, :W])

    local_mean = local_sum / count
    local_var = np.maximum((local_sum_sq / count) - (local_mean ** 2), 0.0)
    local_edge_density = local_edge_sum / count

    roi_rows, roi_cols = np.where(roi_mask > 0)
    feature_vectors = np.column_stack([
        local_mean[roi_rows, roi_cols],
        local_var[roi_rows, roi_cols],
        local_edge_density[roi_rows, roi_cols],
    ])

    return feature_vectors, (roi_rows, roi_cols)


def kmeans_cluster(X, k=3, max_iter=20, seed=42):
    """k-means with k-means++ init."""
    N, D = X.shape
    rng = np.random.RandomState(seed)

    # normalize
    feat_min = X.min(axis=0)
    feat_range = X.max(axis=0) - feat_min
    feat_range[feat_range < 1e-10] = 1.0
    X_norm = (X - feat_min) / feat_range

    # kmeans++ init
    centers = np.empty((k, D), dtype=np.float64)
    centers[0] = X_norm[rng.randint(N)]

    for c in range(1, k):
        dists = np.min(
            [np.sum((X_norm - centers[j]) ** 2, axis=1) for j in range(c)],
            axis=0)
        probs = dists / (dists.sum() + 1e-10)
        cumulative = np.cumsum(probs)
        idx = min(np.searchsorted(cumulative, rng.rand()), N - 1)
        centers[c] = X_norm[idx]

    labels = np.zeros(N, dtype=np.intp)
    for iteration in range(max_iter):
        X_sq = np.sum(X_norm ** 2, axis=1, keepdims=True)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
        XC = X_norm @ centers.T
        dist_sq = X_sq - 2 * XC + C_sq

        new_labels = np.argmin(dist_sq, axis=1)

        if np.array_equal(new_labels, labels) and iteration > 0:
            break
        labels = new_labels

        for j in range(k):
            members = X_norm[labels == j]
            if len(members) == 0:
                centers[j] = X_norm[rng.randint(N)]
            else:
                centers[j] = members.mean(axis=0)

    # un-normalize centers
    centers_orig = centers * feat_range + feat_min
    return labels, centers_orig


def compute_texture_segment_features(grayscale, edges, roi_mask):
    """Smooth/edgy ratios from k-means texture segmentation."""
    roi_area = np.sum(roi_mask)
    if roi_area < 500:
        return {'texture_smooth_ratio': 0.33, 'texture_edgy_ratio': 0.33}

    feature_vectors, _ = compute_local_texture_stats(
        grayscale, edges, roi_mask, patch_size=7)

    N = len(feature_vectors)
    if N < 10:
        return {'texture_smooth_ratio': 0.33, 'texture_edgy_ratio': 0.33}

    # subsample if huge
    max_points = 200000
    if N > max_points:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, max_points, replace=False)
        X_sample = feature_vectors[sample_idx]
    else:
        X_sample = feature_vectors
        sample_idx = None

    labels_sample, centers = kmeans_cluster(X_sample, k=3, max_iter=20, seed=42)

    if sample_idx is not None:
        feat_min = X_sample.min(axis=0)
        feat_range = X_sample.max(axis=0) - feat_min
        feat_range[feat_range < 1e-10] = 1.0
        X_all_norm = (feature_vectors - feat_min) / feat_range
        centers_norm = (centers - feat_min) / feat_range
        dists = np.sum(
            (X_all_norm[:, None, :] - centers_norm[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
    else:
        labels = labels_sample

    # sort by edge density
    sort_order = np.argsort(centers[:, 2])
    remap = np.zeros(3, dtype=np.intp)
    for new_idx, old_idx in enumerate(sort_order):
        remap[old_idx] = new_idx
    sorted_labels = remap[labels]

    total = len(sorted_labels)
    smooth_ratio = np.sum(sorted_labels == 0) / total
    edgy_ratio = np.sum(sorted_labels == 2) / total

    return {
        'texture_smooth_ratio': float(smooth_ratio),
        'texture_edgy_ratio': float(edgy_ratio),
    }


# feature extraction + classification

def extract_features(grayscale, edges, magnitude, direction, roi_mask):
    """Extract all features for classification."""
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
    texture_feats = compute_texture_segment_features(grayscale, edges, roi_mask)

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
        'texture_smooth_ratio': texture_feats['texture_smooth_ratio'],
        'texture_edgy_ratio': texture_feats['texture_edgy_ratio'],
    }


def classify_layer(features, class_centroids=None, shared_inv_cov=None):
    """LDA Mahalanobis classifier over the 6-feature signature.

    The 6 features describing each first-layer image are:
        1. Fill density (with ROI coverage as the normalization term)
        2. Gradient entropy
        3. Gradient direction kurtosis
        4. Edge-to-gradient ratio
        5. Edge density CV
        6. Line-spacing uniformity (with FFT spectral energy as a
           frequency-domain helper for the same line-pattern signal)

    Two of the six features are split into a primary column and a
    helper column for numerical stability:
      * Fill density / ROI coverage - in-ROI density vs. how much of
        the frame the ROI occupies.
      * Line-spacing uniformity / spectral energy - spatial line-pitch
        regularity vs. frequency-domain energy at that pitch.

    Uses pooled within-class covariance (LDA) for robustness on a
    small dataset.
    """
    # 6 features expanded into 8 columns: fill_density + roi_coverage
    # are a paired unit, and line_spacing_uniformity + spectral_energy
    # are a paired unit.  See the docstring.  Order matters - it has
    # to match the column order used when class_centroids and
    # shared_inv_cov were trained.
    feat_keys = [
        'fill_density', 'gradient_entropy', 'gradient_dir_kurtosis',
        'edge_to_gradient_ratio', 'spectral_energy_ratio',
        'edge_density_cv', 'roi_coverage', 'line_spacing_uniformity',
    ]

    if class_centroids is None:
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
        # pooled within-class inv cov
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

    sample = np.array([features[k] for k in feat_keys])

    distances = {}
    for label, mean_vec in class_centroids.items():
        diff = sample - mean_vec
        distances[label] = np.sqrt(np.dot(diff, np.dot(shared_inv_cov, diff)))

    best_label = min(distances, key=distances.get)
    best_dist = distances[best_label]

    confidence = np.exp(-best_dist / 5.0)

    return best_label, float(confidence)
