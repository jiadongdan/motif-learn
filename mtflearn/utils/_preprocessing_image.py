import numpy as np
from skimage.morphology import disk
from skimage.morphology import white_tophat

def normalize_image(image, vmin=0.0, vmax=1.0):
    """
    Normalize an image to a specified range [vmin, vmax].

    Parameters:
    image (np.ndarray): Input image array.
    vmin (float): Minimum value of the normalized image. Default is 0.0.
    vmax (float): Maximum value of the normalized image. Default is 1.0.

    Returns:
    np.ndarray: Normalized image array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.size == 0:
        raise ValueError("Input image cannot be empty.")
    if vmin >= vmax:
        raise ValueError("vmin must be less than vmax.")

    img_max = np.max(image)
    img_min = np.min(image)

    if img_max == img_min:
        raise ValueError("Image has no variation (max equals min).")

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


def remove_bg(img, disk_size):
    selem = disk(disk_size)
    return white_tophat(img, selem)
