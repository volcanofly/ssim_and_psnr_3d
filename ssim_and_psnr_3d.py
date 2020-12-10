#!/usr/bin/env python

import numpy as np
from skimage.metrics import structural_similarity as calc_ssim
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes


def calc_psnr_3d(ref_image, test_image, mask=None, data_range=[None, None]):
    """Calculates 3D PSNR.

    Args:
        ref_image (numpy.ndarray): The reference image.
        test_image (numpy.ndarray): The testing image.
        mask (numpy.ndarray): Calculate PSNR in this mask.
        data_range (iterable[float]): The range of possible values.
    
    Returns:
        float: The calculated PSNR.

    """
    mask = np.ones_like(ref_image) > 0 if mask is None else mask > 0
    min_val = np.min(ref_image) if data_range[0] is None else data_range[0]
    max_val = np.max(ref_image) if data_range[1] is None else data_range[1]

    ref_image = _cutoff(ref_image, min_val, max_val)
    test_image = _cutoff(test_image, min_val, max_val)

    mse = np.mean((ref_image[mask] - test_image[mask]) ** 2)
    psnr = 10 * np.log10((max_val - min_val) ** 2 / mse)

    return psnr


def _cutoff(image, min_val, max_val):
    image = image.copy()
    image[image > max_val] = max_val
    image[image < min_val] = min_val
    return image


def calc_ssim_3d(ref_image, test_image, mask=None):
    """Calculates 3D SSIM.

    Args:
        ref_image (numpy.ndarray): The reference image.
        test_image (numpy.ndarray): The testing image.
        mask (numpy.ndarray): Calculate SSIM in this mask.
    
    Returns:
        float: The calculated SSIM.

    """
    mask = np.ones_like(ref_image) > 0 if mask is None else mask > 0

    ref_image = ref_image / np.max(ref_image) * 255
    ref_image = _cutoff(ref_image, 0, 255)

    test_image = test_image / np.mean(test_image[mask]) * np.mean(ref_image[mask])
    test_image = _cutoff(test_image, 0, 255)

    filter_size = np.max([1, np.round(np.min(test_image.shape) / 256)])
    if filter_size > 1:
        filter = np.ones([filter_size] * 3)
        filter = filter / np.sum(filter)
        ref_image = convolve(ref_image, filter, mode='constant')
        test_image = convolve(test_image, filter, mode='constant')
        ref_image = ref_image[::filter_size, ::filter_size, ::filter_size]
        test_image = test_image[::filter_size, ::filter_size, ::filter_size]

    _, ssim_map = calc_ssim(ref_image, test_image, full=True, data_range=255)
    ssim = np.mean(ssim_map[mask])

    return ssim


def calc_mask(image):
    """Calculates a head mask.

    Args:
        image (numpy.ndarray): The head image.

    Return:
        numpy.ndarray: The head mask in the binary format.

    """
    blur_image = gaussian_filter(image, 2)
    threshold = np.quantile(blur_image, 0.5)
    mask = blur_image > threshold
    mask = binary_fill_holes(mask)
    return mask
