import numpy as np
from skimage.restoration import rolling_ball


def estimate_background_rolling_ball(image, radius=50):
    """
    Estimate image background with the rolling-ball algorithm.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    radius : int or float, default=50
        Rolling-ball radius. Use a scale larger than the foreground
        features that should remain in the residual.

    Returns
    -------
    ndarray
        Estimated background image.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    return rolling_ball(image, radius=radius)


def remove_background_rolling_ball(image, radius=50, clip=True):
    """
    Estimate and subtract background with the rolling-ball algorithm.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    radius : int or float, default=50
        Rolling-ball radius.
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
    background = estimate_background_rolling_ball(image, radius=radius)
    residual = image - background
    if clip:
        residual = np.clip(residual, 0, None)
    return residual, background
