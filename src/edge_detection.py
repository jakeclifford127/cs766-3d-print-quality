import numpy as np
from convolution import convolve2d_fast, separable_convolve2d


# ---------------------------------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma, size=None):
    """Generate a 1D Gaussian kernel.

    Args:
        sigma: standard deviation of the Gaussian
        size: kernel size (must be odd). If None, auto-computed as 6*sigma+1.

    Returns:
        1D numpy array, normalized to sum to 1
    """
    if size is None:
        size = int(np.ceil(sigma * 6)) | 1  # ensure odd
    if size % 2 == 0:
        size += 1

    center = size // 2
    x = np.arange(size) - center
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_kernel_2d(sigma, size=None):
    """Generate a 2D Gaussian kernel as outer product of two 1D kernels."""
    k1d = gaussian_kernel_1d(sigma, size)
    return np.outer(k1d, k1d)


def gaussian_blur(image, sigma=1.0):
    """Apply Gaussian blur using separable convolution.

    Args:
        image: 2D numpy array (grayscale, float64)
        sigma: standard deviation of Gaussian

    Returns:
        Blurred 2D numpy array
    """
    k1d = gaussian_kernel_1d(sigma)
    return separable_convolve2d(image, k1d, k1d, mode='reflect')


# ---------------------------------------------------------------------------
# Sobel Gradient Operators
# ---------------------------------------------------------------------------

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


def sobel_gradients(image):
    """Compute gradient magnitude and direction using Sobel operators.

    Args:
        image: 2D numpy array (grayscale, float64)

    Returns:
        magnitude: 2D array of gradient magnitudes
        direction: 2D array of gradient angles in radians (-pi to pi)
        gx: horizontal gradient
        gy: vertical gradient
    """
    gx = convolve2d_fast(image, SOBEL_X)
    gy = convolve2d_fast(image, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    return magnitude, direction, gx, gy


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_maximum_suppression(magnitude, direction):
    """Thin edges by keeping only local maxima along gradient direction.

    Args:
        magnitude: 2D gradient magnitude array
        direction: 2D gradient direction array (radians)

    Returns:
        2D array with suppressed (thinned) edges
    """
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantize angle to 4 directions: 0, 45, 90, 135 degrees
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]

            # Determine two neighbors to compare along gradient direction
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1 = magnitude[i, j - 1]
                n2 = magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1 = magnitude[i + 1, j - 1]
                n2 = magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                n1 = magnitude[i - 1, j]
                n2 = magnitude[i + 1, j]
            else:
                n1 = magnitude[i - 1, j - 1]
                n2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


# ---------------------------------------------------------------------------
# Double Threshold + Hysteresis
# ---------------------------------------------------------------------------

def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    """Classify edge pixels as strong, weak, or non-edge.

    Args:
        image: 2D array (NMS output)
        low_ratio: low threshold as fraction of max value
        high_ratio: high threshold as fraction of max value

    Returns:
        result: 2D array with strong=255, weak=75, non-edge=0
        strong_val: the strong pixel value (255)
        weak_val: the weak pixel value (75)
    """
    high_thresh = image.max() * high_ratio
    low_thresh = image.max() * low_ratio

    strong_val = 255
    weak_val = 75

    result = np.zeros_like(image, dtype=np.uint8)
    result[image >= high_thresh] = strong_val
    result[(image >= low_thresh) & (image < high_thresh)] = weak_val

    return result, strong_val, weak_val


def hysteresis(image, strong_val=255, weak_val=75):
    """Connect weak edges to strong edges via 8-connectivity.

    Weak pixels adjacent to at least one strong pixel are promoted
    to strong. All remaining weak pixels are suppressed.

    Args:
        image: 2D array from double_threshold
        strong_val: value representing strong edges
        weak_val: value representing weak edges

    Returns:
        2D binary edge map (0 or 255)
    """
    h, w = image.shape
    output = image.copy()

    # Iterative approach: keep promoting weak->strong until stable
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if output[i, j] == weak_val:
                    # Check 8 neighbors for a strong pixel
                    neighborhood = output[i - 1:i + 2, j - 1:j + 2]
                    if strong_val in neighborhood:
                        output[i, j] = strong_val
                        changed = True

    # Kill remaining weak pixels
    output[output != strong_val] = 0
    return output


# ---------------------------------------------------------------------------
# Full Canny Edge Detection Pipeline
# ---------------------------------------------------------------------------

def canny_edge_detection(image, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
    """Complete Canny edge detection from scratch.

    Pipeline:
        1. Gaussian blur (noise reduction)
        2. Sobel gradients (magnitude + direction)
        3. Non-maximum suppression (edge thinning)
        4. Double threshold (classify strong/weak/non-edge)
        5. Hysteresis (connect weak edges to strong)

    Args:
        image: 2D numpy array (grayscale, float64, values 0-255)
        sigma: Gaussian blur sigma
        low_ratio: low hysteresis threshold ratio
        high_ratio: high hysteresis threshold ratio

    Returns:
        edges: binary edge map (0 or 255)
        magnitude: gradient magnitude (useful for later analysis)
        direction: gradient direction (useful for later analysis)
    """
    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, sigma)

    # Step 2: Sobel gradients
    magnitude, direction, gx, gy = sobel_gradients(blurred)

    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction)

    # Step 4: Double threshold
    thresholded, strong_val, weak_val = double_threshold(
        suppressed, low_ratio, high_ratio
    )

    # Step 5: Hysteresis edge tracking
    edges = hysteresis(thresholded, strong_val, weak_val)

    return edges, magnitude, direction
