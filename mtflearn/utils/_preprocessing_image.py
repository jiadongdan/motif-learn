import numpy as np
from skimage.morphology import disk
from skimage.morphology import white_tophat


def normalize_image(img, mode="minmax", eps=1e-8, vmin=0.0, vmax=1.0):
    """
    Normalize a 2D (or multi-channel) NumPy image.

    Note: Requires finite input. Use `ensure_finite()` or `mask_nonfinite()`
    helper functions if your data contains NaN/inf.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C). Must contain only finite values.
    mode : str
        "l1", "l2", or "minmax".
    eps : float
        Small constant to avoid division by zero.
    vmin, vmax : float
        Output range for minmax mode.

    Returns
    -------
    np.ndarray
        Normalized image.

    Raises
    ------
    ValueError
        If input contains NaN or inf values.
    """
    img = img.astype(np.float32, copy=False)

    # Explicit check - fail fast
    if not np.all(np.isfinite(img)):
        raise ValueError(
            "Input contains NaN or inf values. "
            "Use `ensure_finite(img)` or `mask_nonfinite(img)` to handle them first."
        )

    if mode == "l1":
        norm = np.sum(np.abs(img)) + eps
        return img / norm

    elif mode == "l2":
        norm = np.sqrt(np.sum(img ** 2) + eps)
        return img / norm

    elif mode == "minmax":
        x_min = img.min()
        x_max = img.max()

        if np.isclose(x_min, x_max):
            return np.full_like(img, (vmin + vmax) / 2)

        scale = (x_max - x_min) + eps
        return vmin + (img - x_min) * (vmax - vmin) / scale

    else:
        raise ValueError("mode must be 'l1', 'l2', or 'minmax'.")


# Helper functions - user chooses behavior explicitly
def ensure_finite(img, nan_value=0.0, inf_value=None):
    """
    Replace non-finite values with specified constants.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    nan_value : float
        Value to replace NaN with (default: 0.0).
    inf_value : float or None
        Value to replace +/-inf with. If None, uses min/max of finite values.

    Returns
    -------
    np.ndarray
        Image with non-finite values replaced.
    """
    img = img.copy()

    if inf_value is None and np.any(np.isinf(img)):
        finite_mask = np.isfinite(img)
        if np.any(finite_mask):
            finite_min = img[finite_mask].min()
            finite_max = img[finite_mask].max()
            img[img == -np.inf] = finite_min
            img[img == np.inf] = finite_max
    elif inf_value is not None:
        img[np.isinf(img)] = inf_value

    img[np.isnan(img)] = nan_value

    return img


def mask_nonfinite(img):
    """
    Create a boolean mask of finite values and the finite data.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    finite_data : np.ndarray
        Flattened array of finite values only.
    mask : np.ndarray
        Boolean mask of same shape as img (True = finite).

    Examples
    --------
    >>> img = np.array([[1, np.nan], [3, np.inf]])
    >>> finite_data, mask = mask_nonfinite(img)
    >>> finite_data
    array([1., 3.])
    >>> mask
    array([[ True, False],
           [ True, False]])
    """
    mask = np.isfinite(img)
    finite_data = img[mask]
    return finite_data, mask


def normalize_image_robust(img, mode="minmax", eps=1e-8, vmin=0.0, vmax=1.0):
    """
    Normalize image, preserving non-finite values.

    Non-finite values (NaN, inf) are excluded from normalization calculations
    but preserved in their original positions in the output.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    mode : str
        "l1", "l2", or "minmax".
    eps : float
        Small constant to avoid division by zero.
    vmin, vmax : float
        Output range for minmax mode.

    Returns
    -------
    np.ndarray
        Normalized image with non-finite values preserved.

    Raises
    ------
    ValueError
        If all values are non-finite.
    """
    img = img.astype(np.float32, copy=False)

    finite_mask = np.isfinite(img)

    if not np.any(finite_mask):
        raise ValueError("All values are non-finite (NaN or inf)")

    if mode == "l1":
        norm = np.sum(np.abs(img[finite_mask])) + eps
        return img / norm

    elif mode == "l2":
        norm = np.sqrt(np.sum(img[finite_mask] ** 2) + eps)
        return img / norm

    elif mode == "minmax":
        finite_vals = img[finite_mask]
        x_min = finite_vals.min()
        x_max = finite_vals.max()

        if np.isclose(x_min, x_max):
            result = np.full_like(img, (vmin + vmax) / 2)
            result[~finite_mask] = img[~finite_mask]
            return result

        scale = (x_max - x_min) + eps
        result = vmin + (img - x_min) * (vmax - vmin) / scale
        result[~finite_mask] = img[~finite_mask]

        return result

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
