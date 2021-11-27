import numpy as np
from scipy import ndimage as ndi


def gaussian(img, sigma, mode='nearest', cval=0, truncate=4.0):
    """Multi-dimensional Gaussian filter.
        Parameters
        ----------
        img : array-like
            Input image or images to filter.
        sigma : scalar or sequence of scalars, optional
            Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a
            sequence, or as a single number, in which case it is equal for
            all axes.
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The ``mode`` parameter determines how the array borders are
            handled, where ``cval`` is the value when mode is equal to
            'constant'. Default is 'nearest'.
        cval : scalar, optional
            Value to fill past edges of input if ``mode`` is 'constant'. Default
            is 0.0
        truncate : float, optional
            Truncate the filter at this many standard deviations.
        Returns
        -------
        filtered_image : ndarray
            the filtered array
        Notes
        -----
        This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.
        Integer arrays are converted to float.
    """
    # convert image to float
    img_float = img.astype(np.float)
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    output = ndi.gaussian_filter(img_float, sigma, mode=mode, cval=cval, truncate=truncate)
    return output

def local_mean(img, size, mode='reflect', cval=0.0):
    imgf = ndi.uniform_filter(img, size, mode=mode, cval=cval)
    return imgf

def local_median(img, size=None, footprint=None, mode='reflect', cval=0.0):
    imgf = ndi.median_filter(img, size=size, footprint=footprint, mode=mode, cval=cval)
    return imgf



