import numpy as np


def pad_image(image, pad_h, pad_w, mode='zero'):
    """Pad image with zeros or reflection."""
    if mode == 'reflect':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)


def convolve2d(image, kernel, mode='zero'):
    """Naive 2D convolution, numpy only."""
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # flip for true convolution
    kernel_flipped = kernel[::-1, ::-1]

    kh, kw = kernel_flipped.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode)
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel_flipped)

    return output


def convolve2d_fast(image, kernel, mode='zero'):
    """Stride tricks version of 2d conv."""
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)
    kernel_flipped = kernel[::-1, ::-1]

    kh, kw = kernel_flipped.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode)
    h, w = image.shape

    # sliding windows via strides
    shape = (h, w, kh, kw)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    output = np.einsum('ijkl,kl->ij', windows, kernel_flipped)

    return output


def separable_convolve2d(image, kernel_x, kernel_y, mode='zero'):
    """Separable conv — 1D in x then y."""
    kx = kernel_x.reshape(1, -1)
    ky = kernel_y.reshape(-1, 1)

    temp = convolve2d_fast(image, kx, mode)
    output = convolve2d_fast(temp, ky, mode)

    return output
