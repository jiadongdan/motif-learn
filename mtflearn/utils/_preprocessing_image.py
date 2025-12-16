import numpy as np
from skimage.morphology import disk
from skimage.morphology import white_tophat


def normalize_image(img, mode="minmax", eps=1e-8, vmin=0.0, vmax=1.0):
    """
    Normalize a 2D (or multi-channel) NumPy image.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    mode : str
        "l1", "l2", or "minmax".
    eps : float
        Small constant to avoid division by zero.
    vmin : float
        Lower bound used for min-max normalization.
    vmax : float
        Upper bound used for min-max normalization.

    Returns
    -------
    np.ndarray
        Normalized image with the same shape.
    """
    img = img.astype(np.float32, copy=False)

    if mode == "l1":
        norm = np.sum(np.abs(img)) + eps
        return img / norm

    elif mode == "l2":
        norm = np.sqrt(np.sum(img ** 2) + eps)
        return img / norm

    elif mode == "minmax":
        x_min = img.min()
        x_max = img.max()
        scale = (x_max - x_min) + eps
        return vmin + (img - x_min) * (vmax - vmin) / scale

    else:
        raise ValueError("mode must be 'l1', 'l2', or 'minmax'.")


def standardize_image(image):
    """
    Standardize the image to have mean 0 and standard deviation 1.

    Parameters:
    image : numpy array
        The input 2D image.

    Returns:
    standardized_image : numpy array
        The standardized 2D image.
    """
    mean = np.mean(image)
    std = np.std(image)

    if std == 0:
        raise ValueError("Standard deviation is zero, can't standardize the image.")

    standardized_image = (image - mean) / std
    return standardized_image


def remove_bg(img, disk_size):
    selem = disk(disk_size)
    return white_tophat(img, selem)
