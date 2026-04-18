import numpy as np


def rgb_to_grayscale(image):
    """Luminance-weighted grayscale."""
    return (0.2989 * image[:, :, 0] +
            0.5870 * image[:, :, 1] +
            0.1140 * image[:, :, 2]).astype(np.float64)


def rgb_to_hsv(image):
    """RGB to HSV, H in [0,360]."""
    img = image.astype(np.float64) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    diff = cmax - cmin

    # hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    # saturation
    s = np.zeros_like(cmax)
    s[cmax > 0] = diff[cmax > 0] / cmax[cmax > 0]

    v = cmax

    return np.stack([h, s, v], axis=2)


def threshold_background(hsv_image, h_range=None, s_range=(0.0, 0.3), v_range=(0.0, 0.5)):
    """Separate print from bed via HSV thresholds."""
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    bg_mask = np.ones(h.shape, dtype=bool)

    if h_range is not None:
        bg_mask &= (h >= h_range[0]) & (h <= h_range[1])
    bg_mask &= (s >= s_range[0]) & (s <= s_range[1])
    bg_mask &= (v >= v_range[0]) & (v <= v_range[1])

    return (~bg_mask).astype(np.uint8)


def otsu_threshold(grayscale):
    """Otsu's method for auto thresholding."""
    pixel_vals = np.clip(grayscale, 0, 255).astype(np.uint8).ravel()
    total_pixels = len(pixel_vals)

    # histogram
    hist = np.zeros(256, dtype=np.float64)
    for v in pixel_vals:
        hist[v] += 1
    hist /= total_pixels

    best_threshold = 0
    best_variance = 0.0

    w0 = 0.0
    sum0 = 0.0

    total_mean = np.sum(np.arange(256) * hist)

    for t in range(256):
        w0 += hist[t]
        w1 = 1.0 - w0

        if w0 == 0 or w1 == 0:
            continue

        sum0 += t * hist[t]

        mean0 = sum0 / w0
        mean1 = (total_mean - sum0) / w1

        variance = w0 * w1 * (mean0 - mean1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    mask = (grayscale > best_threshold).astype(np.uint8)
    return mask, best_threshold


def adaptive_threshold(grayscale, block_size=31, c=10):
    """Local mean adaptive threshold."""
    from convolution import convolve2d_fast

    kernel = np.ones((block_size, block_size), dtype=np.float64)
    kernel /= kernel.size
    local_mean = convolve2d_fast(grayscale, kernel, mode='reflect')

    mask = (grayscale > (local_mean - c)).astype(np.uint8)
    return mask


# morphological ops

def dilate(mask, kernel_size=3, iterations=1):
    """Binary dilation via stride tricks."""
    result = mask.copy().astype(np.uint8)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=0)
        h, w = result.shape
        shape = (h, w, kernel_size, kernel_size)
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        result = windows.max(axis=(2, 3)).astype(np.uint8)

    return result


def erode(mask, kernel_size=3, iterations=1):
    """Binary erosion via stride tricks."""
    result = mask.copy().astype(np.uint8)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=1)
        h, w = result.shape
        shape = (h, w, kernel_size, kernel_size)
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        result = windows.min(axis=(2, 3)).astype(np.uint8)

    return result


def morph_open(mask, kernel_size=3, iterations=1):
    """Erode then dilate."""
    temp = erode(mask, kernel_size, iterations)
    return dilate(temp, kernel_size, iterations)


def morph_close(mask, kernel_size=3, iterations=1):
    """Dilate then erode."""
    temp = dilate(mask, kernel_size, iterations)
    return erode(temp, kernel_size, iterations)


# connected components

def flood_fill(mask, seed_row, seed_col):
    """BFS flood fill from seed."""
    from collections import deque

    h, w = mask.shape
    filled = np.zeros_like(mask, dtype=np.uint8)

    if mask[seed_row, seed_col] == 0:
        return filled

    visited = np.zeros_like(mask, dtype=np.uint8)
    queue = deque([(seed_row, seed_col)])
    visited[seed_row, seed_col] = 1
    filled[seed_row, seed_col] = 1

    while queue:
        r, c = queue.popleft()

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                visited[nr, nc] = 1
                filled[nr, nc] = 1
                queue.append((nr, nc))

    return filled


def connected_components(mask):
    """Label connected components."""
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    current_label = 0

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and labels[i, j] == 0:
                current_label += 1
                component = flood_fill(mask, i, j)
                labels[component == 1] = current_label

    return labels, current_label


def largest_component_mask(mask):
    """Keep only biggest component."""
    labels, num = connected_components(mask)

    if num == 0:
        return mask

    best_label = 0
    best_count = 0
    for label_id in range(1, num + 1):
        count = np.sum(labels == label_id)
        if count > best_count:
            best_count = count
            best_label = label_id

    return (labels == best_label).astype(np.uint8)


def _mask_quality(mask):
    """Score mask plausibility."""
    total = mask.size
    fg_ratio = np.sum(mask) / total

    if fg_ratio < 0.03 or fg_ratio > 0.90:
        return 0.0

    # peak at ~30% fg
    ratio_score = 1.0 - abs(fg_ratio - 0.35) / 0.55
    ratio_score = max(0.0, ratio_score)

    return ratio_score


def generate_roi_mask(image_rgb, method='auto', cleanup=True):
    """Generate ROI mask for print area.

    Tries otsu, HSV, and inverted otsu, picks best.
    """
    gray = rgb_to_grayscale(image_rgb)

    if method == 'auto':
        candidates = []

        # otsu
        otsu_mask, _ = otsu_threshold(gray)
        candidates.append(otsu_mask)

        # hsv
        hsv = rgb_to_hsv(image_rgb)
        hsv_mask = threshold_background(hsv)
        candidates.append(hsv_mask)

        # inverted otsu
        inv_mask, _ = otsu_threshold(255.0 - gray)
        candidates.append(inv_mask)

        best_mask = None
        best_score = -1.0

        for candidate in candidates:
            if cleanup:
                cleaned = morph_open(candidate, kernel_size=5, iterations=1)
                cleaned = morph_close(cleaned, kernel_size=5, iterations=1)
                cleaned = largest_component_mask(cleaned)
            else:
                cleaned = candidate

            score = _mask_quality(cleaned)
            if score > best_score:
                best_score = score
                best_mask = cleaned

        return best_mask

    elif method == 'otsu':
        mask, _ = otsu_threshold(gray)
    elif method == 'hsv':
        hsv = rgb_to_hsv(image_rgb)
        mask = threshold_background(hsv)
    elif method == 'adaptive':
        mask = adaptive_threshold(gray)
    else:
        raise ValueError(f"Unknown method: {method}")

    if cleanup:
        mask = morph_open(mask, kernel_size=5, iterations=1)
        mask = morph_close(mask, kernel_size=5, iterations=1)
        mask = largest_component_mask(mask)

    return mask
