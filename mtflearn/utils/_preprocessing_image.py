import numpy as np


def normalize_image(image, vmin=0., vmax=1.):
    img_max = np.max(image)
    img_min = np.min(image)
    img_norm = (image - img_min) / (img_max - img_min) * (vmax - vmin) + vmin
    return img_norm


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
