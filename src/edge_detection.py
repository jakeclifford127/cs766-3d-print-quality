import numpy as np
from convolution import convolve2d_fast, separable_convolve2d


def gaussian_kernel_1d(sigma, size=None):
    """1D gaussian kernel, normalized."""
    if size is None:
        size = int(np.ceil(sigma * 6)) | 1  # ensure odd
    if size % 2 == 0:
        size += 1

    center = size // 2
    x = np.arange(size) - center
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_kernel_2d(sigma, size=None):
    """2D gaussian as outer product."""
    k1d = gaussian_kernel_1d(sigma, size)
    return np.outer(k1d, k1d)


def gaussian_blur(image, sigma=1.0):
    """Separable gaussian blur."""
    k1d = gaussian_kernel_1d(sigma)
    return separable_convolve2d(image, k1d, k1d, mode='reflect')


# sobel kernels
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
    """Gradient magnitude + direction via sobel."""
    gx = convolve2d_fast(image, SOBEL_X)
    gy = convolve2d_fast(image, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    return magnitude, direction, gx, gy


def non_maximum_suppression(magnitude, direction):
    """Thin edges to local maxima."""
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # quantize to 4 directions
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]

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


def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    """Classify pixels: strong/weak/non-edge."""
    high_thresh = image.max() * high_ratio
    low_thresh = image.max() * low_ratio

    strong_val = 255
    weak_val = 75

    result = np.zeros_like(image, dtype=np.uint8)
    result[image >= high_thresh] = strong_val
    result[(image >= low_thresh) & (image < high_thresh)] = weak_val

    return result, strong_val, weak_val


def hysteresis(image, strong_val=255, weak_val=75):
    """Connect weak edges to strong ones."""
    h, w = image.shape
    output = image.copy()

    # promote weak->strong iteratively
    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if output[i, j] == weak_val:
                    neighborhood = output[i - 1:i + 2, j - 1:j + 2]
                    if strong_val in neighborhood:
                        output[i, j] = strong_val
                        changed = True

    output[output != strong_val] = 0
    return output


def canny_edge_detection(image, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
    """Full canny from scratch."""
    blurred = gaussian_blur(image, sigma)

    magnitude, direction, gx, gy = sobel_gradients(blurred)

    suppressed = non_maximum_suppression(magnitude, direction)

    thresholded, strong_val, weak_val = double_threshold(
        suppressed, low_ratio, high_ratio
    )

    edges = hysteresis(thresholded, strong_val, weak_val)

    return edges, magnitude, direction
