import numpy as np


# ---------------------------------------------------------------------------
# Color Space Conversion
# ---------------------------------------------------------------------------

def rgb_to_grayscale(image):
    """Convert RGB image to grayscale using luminance weights.

    Args:
        image: 3D numpy array (H, W, 3), uint8 or float

    Returns:
        2D numpy array (H, W), float64
    """
    return (0.2989 * image[:, :, 0] +
            0.5870 * image[:, :, 1] +
            0.1140 * image[:, :, 2]).astype(np.float64)


def rgb_to_hsv(image):
    """Convert RGB image to HSV color space.

    Args:
        image: 3D numpy array (H, W, 3), values 0-255

    Returns:
        3D numpy array (H, W, 3) with H in [0,360], S in [0,1], V in [0,1]
    """
    img = image.astype(np.float64) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    diff = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    # Saturation
    s = np.zeros_like(cmax)
    s[cmax > 0] = diff[cmax > 0] / cmax[cmax > 0]

    # Value
    v = cmax

    return np.stack([h, s, v], axis=2)


# ---------------------------------------------------------------------------
# Background Subtraction via Color Thresholding
# ---------------------------------------------------------------------------

def threshold_background(hsv_image, h_range=None, s_range=(0.0, 0.3), v_range=(0.0, 0.5)):
    """Create a binary mask separating the print from the build plate.

    Default ranges target dark/neutral build plates (black, dark gray).
    Pixels OUTSIDE these ranges are considered foreground (the print).

    Args:
        hsv_image: 3D array (H, W, 3) in HSV space
        h_range: tuple (min, max) for hue, or None to ignore hue
        s_range: tuple (min, max) for saturation
        v_range: tuple (min, max) for value/brightness

    Returns:
        Binary mask: 1 = foreground (print), 0 = background (bed)
    """
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    bg_mask = np.ones(h.shape, dtype=bool)

    if h_range is not None:
        bg_mask &= (h >= h_range[0]) & (h <= h_range[1])
    bg_mask &= (s >= s_range[0]) & (s <= s_range[1])
    bg_mask &= (v >= v_range[0]) & (v <= v_range[1])

    # Foreground is the inverse of background
    return (~bg_mask).astype(np.uint8)


def adaptive_threshold(grayscale, block_size=31, c=10):
    """Simple adaptive thresholding for foreground extraction.

    Each pixel is thresholded against the mean of its local neighborhood.

    Args:
        grayscale: 2D numpy array (float64, 0-255 range)
        block_size: size of the local neighborhood (must be odd)
        c: constant subtracted from local mean

    Returns:
        Binary mask (0 or 1)
    """
    from convolution import convolve2d_fast

    # Compute local mean using a box filter
    kernel = np.ones((block_size, block_size), dtype=np.float64)
    kernel /= kernel.size
    local_mean = convolve2d_fast(grayscale, kernel, mode='reflect')

    # Pixel is foreground if it's brighter than local mean - c
    mask = (grayscale > (local_mean - c)).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Morphological Operations (for mask cleanup)
# ---------------------------------------------------------------------------

def dilate(mask, kernel_size=3, iterations=1):
    """Binary dilation — expand foreground regions.

    Uses sliding-window max via stride tricks instead of Python loops
    for much better performance on larger images.

    Args:
        mask: 2D binary array (0/1)
        kernel_size: size of square structuring element (odd)
        iterations: number of times to apply dilation

    Returns:
        Dilated binary mask
    """
    result = mask.copy().astype(np.uint8)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=0)
        h, w = result.shape
        # Build windowed view and take max over the kernel
        shape = (h, w, kernel_size, kernel_size)
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        result = windows.max(axis=(2, 3)).astype(np.uint8)

    return result


def erode(mask, kernel_size=3, iterations=1):
    """Binary erosion — shrink foreground regions.

    Uses sliding-window min via stride tricks instead of Python loops
    for much better performance on larger images.

    Args:
        mask: 2D binary array (0/1)
        kernel_size: size of square structuring element (odd)
        iterations: number of times to apply erosion

    Returns:
        Eroded binary mask
    """
    result = mask.copy().astype(np.uint8)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(result, pad, mode='constant', constant_values=1)
        h, w = result.shape
        # Build windowed view and take min over the kernel
        shape = (h, w, kernel_size, kernel_size)
        strides = padded.strides * 2
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        result = windows.min(axis=(2, 3)).astype(np.uint8)

    return result


def morph_open(mask, kernel_size=3, iterations=1):
    """Morphological opening (erode then dilate) — removes small noise."""
    temp = erode(mask, kernel_size, iterations)
    return dilate(temp, kernel_size, iterations)


def morph_close(mask, kernel_size=3, iterations=1):
    """Morphological closing (dilate then erode) — fills small holes."""
    temp = dilate(mask, kernel_size, iterations)
    return erode(temp, kernel_size, iterations)


# ---------------------------------------------------------------------------
# Connected Components / Flood Fill
# ---------------------------------------------------------------------------

def flood_fill(mask, seed_row, seed_col):
    """Flood fill from a seed point on a binary mask.

    Uses a numpy visited array instead of a Python set for performance.

    Args:
        mask: 2D binary array (1 = fillable region)
        seed_row, seed_col: starting coordinates

    Returns:
        2D binary array with only the connected component containing the seed
    """
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
    """Label all connected components in a binary mask.

    Args:
        mask: 2D binary array (0/1)

    Returns:
        labels: 2D array where each component has a unique integer label
        num_components: total number of components found
    """
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
    """Extract only the largest connected component from a binary mask.

    Useful for isolating the main print region and discarding small noise.

    Args:
        mask: 2D binary array

    Returns:
        Binary mask containing only the largest component
    """
    labels, num = connected_components(mask)

    if num == 0:
        return mask

    # Find label with most pixels
    best_label = 0
    best_count = 0
    for label_id in range(1, num + 1):
        count = np.sum(labels == label_id)
        if count > best_count:
            best_count = count
            best_label = label_id

    return (labels == best_label).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full Masking Pipeline
# ---------------------------------------------------------------------------

def generate_roi_mask(image_rgb, method='hsv', cleanup=True):
    """Generate a Region of Interest mask for the printed area.

    Args:
        image_rgb: 3D numpy array (H, W, 3), uint8, RGB
        method: 'hsv' for color thresholding, 'adaptive' for brightness
        cleanup: if True, apply morphological ops and keep largest component

    Returns:
        Binary ROI mask (1 = print area, 0 = background)
    """
    if method == 'hsv':
        hsv = rgb_to_hsv(image_rgb)
        mask = threshold_background(hsv)
    elif method == 'adaptive':
        gray = rgb_to_grayscale(image_rgb)
        mask = adaptive_threshold(gray)
    else:
        raise ValueError(f"Unknown method: {method}")

    if cleanup:
        # Remove small noise, fill small holes
        mask = morph_open(mask, kernel_size=5, iterations=1)
        mask = morph_close(mask, kernel_size=5, iterations=1)
        # Keep only the largest connected region
        mask = largest_component_mask(mask)

    return mask
