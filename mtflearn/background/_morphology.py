import numpy as np
from scipy.ndimage import grey_opening


def estimate_background_opening(image, size=15):
    """
    Estimate a smooth image background using grayscale morphological opening.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    size : int or tuple of int, default=15
        Size of the grayscale opening footprint. Use a scale larger than
        the foreground peaks you want to preserve.

    Returns
    -------
    ndarray
        Estimated background image.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    if np.isscalar(size):
        size = int(size)
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        size = (size, size)
    else:
        size = tuple(int(v) for v in size)
        if len(size) != 2 or any(v <= 0 for v in size):
            raise ValueError("size must be a positive int or a length-2 tuple.")

    return grey_opening(image, size=size)


def remove_background_opening(image, size=15, clip=True):
    """
    Estimate and subtract a smooth background using grayscale opening.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    size : int or tuple of int, default=15
        Size of the grayscale opening footprint.
    clip : bool, default=True
        If True, clip the residual to be nonnegative.

    Returns
    -------
    residual : ndarray
        Background-subtracted image.
    background : ndarray
        Estimated background image.
    """
    image = np.asarray(image)
    background = estimate_background_opening(image, size=size)
    residual = image - background
    if clip:
        residual = np.clip(residual, 0, None)
    return residual, background
