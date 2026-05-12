import numpy as np
from scipy.ndimage import grey_opening


def _disk_footprint(size):
    radius = size / 2.0
    grid = np.arange(size) - (size - 1) / 2.0
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    return (xx**2 + yy**2) <= radius**2


def estimate_background_opening(image, size=None, footprint=None, shape="disk"):
    """
    Estimate a smooth image background using grayscale morphological opening.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    size : int or tuple of int, optional
        Scale of the grayscale opening support. For scalar input, the
        interpretation depends on ``shape``. For tuple input, use
        ``shape='rectangle'``.
    footprint : ndarray of bool or int, optional
        Structuring footprint passed directly to ``grey_opening``. Mutually
        exclusive with ``size``.
    shape : {"disk", "square", "rectangle"}, default="disk"
        Structuring shape to build from ``size`` when ``footprint`` is not
        provided.

    Returns
    -------
    ndarray
        Estimated background image.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    if (size is None) == (footprint is None):
        raise ValueError("Provide exactly one of size or footprint.")

    if footprint is not None:
        footprint = np.asarray(footprint)
        if footprint.ndim != 2:
            raise ValueError("footprint must be a 2D array.")
        if not np.any(footprint):
            raise ValueError("footprint must contain at least one nonzero entry.")
        return grey_opening(image, footprint=footprint)

    shape = str(shape).lower()
    if shape not in {"disk", "square", "rectangle"}:
        raise ValueError("shape must be one of {'disk', 'square', 'rectangle'}.")

    if np.isscalar(size):
        size = int(size)
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        if shape == "disk":
            return grey_opening(image, footprint=_disk_footprint(size))
        return grey_opening(image, size=(size, size))

    size = tuple(int(v) for v in size)
    if len(size) != 2 or any(v <= 0 for v in size):
        raise ValueError("size must be a positive int or a length-2 tuple.")
    if shape != "rectangle":
        raise ValueError("tuple size requires shape='rectangle'.")

    return grey_opening(image, size=size)


def remove_background_opening(image, size=None, footprint=None, shape="disk", clip=True):
    """
    Estimate and subtract a smooth background using grayscale opening.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    size : int or tuple of int, optional
        Scale of the grayscale opening support. For scalar input, the
        interpretation depends on ``shape``. For tuple input, use
        ``shape='rectangle'``.
    footprint : ndarray of bool or int, optional
        Structuring footprint passed directly to ``grey_opening``. Mutually
        exclusive with ``size``.
    shape : {"disk", "square", "rectangle"}, default="disk"
        Structuring shape to build from ``size`` when ``footprint`` is not
        provided.
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
    background = estimate_background_opening(
        image,
        size=size,
        footprint=footprint,
        shape=shape,
    )
    residual = image - background
    if clip:
        residual = np.clip(residual, 0, None)
    return residual, background
