import numpy as np
from scipy.ndimage import gaussian_filter


def estimate_background_baseline(image, sigma=20, num_iters=10):
    """
    Estimate a smooth lower-envelope background surface.

    This baseline model is a smooth trend image rather than a morphology-
    specific background. It is useful when the desired background is a
    slowly varying surface and the residual should stay nonnegative.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    sigma : float or tuple of float, default=20
        Gaussian smoothing scale for the background surface.
    num_iters : int, default=10
        Number of lower-envelope refinement iterations.

    Returns
    -------
    ndarray
        Estimated baseline/background image.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")

    baseline = gaussian_filter(image, sigma=sigma)
    baseline = np.minimum(baseline, image)

    for _ in range(int(num_iters) - 1):
        baseline = gaussian_filter(baseline, sigma=sigma)
        baseline = np.minimum(baseline, image)

    return baseline


def remove_background_baseline(image, sigma=20, num_iters=10, clip=True):
    """
    Estimate and subtract a smooth baseline/background surface.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    sigma : float or tuple of float, default=20
        Gaussian smoothing scale for the background surface.
    num_iters : int, default=10
        Number of lower-envelope refinement iterations.
    clip : bool, default=True
        If True, clip the residual to be nonnegative.

    Returns
    -------
    residual : ndarray
        Background-subtracted image.
    background : ndarray
        Estimated baseline/background image.
    """
    image = np.asarray(image, dtype=float)
    background = estimate_background_baseline(image, sigma=sigma, num_iters=num_iters)
    residual = image - background
    if clip:
        residual = np.clip(residual, 0, None)
    return residual, background
