import numpy as np


def pad_image(image, pad_h, pad_w, mode='zero'):
    """Pad an image with zeros or by reflecting edges.

    Args:
        image: 2D numpy array (grayscale)
        pad_h: padding height (top and bottom)
        pad_w: padding width (left and right)
        mode: 'zero' or 'reflect'

    Returns:
        Padded 2D numpy array
    """
    if mode == 'reflect':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)


def convolve2d(image, kernel, mode='zero'):
    """Custom 2D convolution using only NumPy.

    Performs cross-correlation with a flipped kernel (true convolution).

    Args:
        image: 2D numpy array (grayscale, float64)
        kernel: 2D numpy array (filter kernel)
        mode: padding mode — 'zero' or 'reflect'

    Returns:
        2D numpy array, same size as input image
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Flip kernel for true convolution (not just cross-correlation)
    kernel_flipped = kernel[::-1, ::-1]

    kh, kw = kernel_flipped.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode)
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float64)

    # Sliding window convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel_flipped)

    return output


def convolve2d_fast(image, kernel, mode='zero'):
    """Optimized 2D convolution using NumPy stride tricks.

    Much faster than the naive loop version. Same results.

    Args:
        image: 2D numpy array (grayscale, float64)
        kernel: 2D numpy array (filter kernel)
        mode: padding mode — 'zero' or 'reflect'

    Returns:
        2D numpy array, same size as input image
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kernel_flipped = kernel[::-1, ::-1]

    kh, kw = kernel_flipped.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode)
    h, w = image.shape

    # Create a view of all sliding windows using stride tricks
    shape = (h, w, kh, kw)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Element-wise multiply all windows by kernel and sum
    output = np.einsum('ijkl,kl->ij', windows, kernel_flipped)

    return output


def separable_convolve2d(image, kernel_x, kernel_y, mode='zero'):
    """Separable 2D convolution — applies 1D kernel in x then y.

    Faster than full 2D convolution when the kernel is separable
    (e.g., Gaussian = outer product of two 1D Gaussians).

    Args:
        image: 2D numpy array
        kernel_x: 1D numpy array (horizontal kernel)
        kernel_y: 1D numpy array (vertical kernel)
        mode: padding mode

    Returns:
        2D numpy array, same size as input
    """
    # Reshape 1D kernels to 2D for reuse of convolve2d_fast
    kx = kernel_x.reshape(1, -1)
    ky = kernel_y.reshape(-1, 1)

    # Convolve horizontally, then vertically
    temp = convolve2d_fast(image, kx, mode)
    output = convolve2d_fast(temp, ky, mode)

    return output
