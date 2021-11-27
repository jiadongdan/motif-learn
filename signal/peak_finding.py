import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar

from collections import OrderedDict
from skimage.filters import (threshold_local,
                             threshold_li,
                             threshold_otsu,
                             threshold_mean,
                             threshold_minimum,
                             threshold_yen)
from skimage.feature import peak_local_max
from skimage.restoration import estimate_sigma

from .baseline import baseline_correction

def get_patch_size(img):
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    fft_log = np.log(fft_abs + 1)
    i, j = np.unravel_index(np.argmax(fft_log), shape=img.shape)
    line = warp_polar(fft_log, center=(i, j)).mean(axis=0)[0:i]
    # get the baseline
    baseline = baseline_correction(line)
    # remove the baseline
    line_ = line - baseline
    ind = np.argmax(line_)
    size = np.floor(img.shape[0] / ind)
    return 2*int(size) + 1


def _normalize(image, low, high):
    img_max = image.max()
    img_min = image.min()
    img_norm = (image - img_min) / (img_max - img_min) * (high - low) + low
    return img_norm


def local_max(img, block_size=None, method='gaussian', min_distance=None, threshold=None, plot=False):
    """
    Find peaks in an image as array with the shape of (num_pts, 2).

    Parameters
    ----------
    img: array
    block_size: int, optional
        Odd size of pixel neighborhood which is used to calculate the threshold value for smoothed image
    method: {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in weighted mean image.

        ‘generic’: use custom function (see param parameter)

        ‘gaussian’: apply gaussian filter (see param parameter for custom sigma value)

        ‘mean’: apply arithmetic mean filter

        ‘median’: apply median rank filter

        By default the ‘gaussian’ method is used.

    threshold: scalar, str {'li', 'otsu'}

    plot: bool
        If true, scatter plot of points will be displayed.
    Returns
    -------
    pts: array with shape of (num_pts, 2)
    """

    img = _normalize(img, 0, 255)
    # get a smooth version image
    sigma = estimate_sigma(img)
    if sigma < 1:
        t = img
    else:
        # step 1: determine the characteristic window size
        if block_size is None:
            block_size = get_patch_size(img)//3
            if block_size %2 == 0:
                block_size += 1
        # step 2: apply local thresholding
        t = threshold_local(img, block_size=block_size, method=method)
    # get local max from smooth threshold image

    threshold_func = OrderedDict({'li': threshold_li,
                                  'otsu': threshold_otsu,
                                  'mean': threshold_mean,
                                  'yen': threshold_yen,
                                  'min': np.min
                                 })
    if threshold is None:
        threshold_abs = threshold_func['li'](t)
    elif np.isscalar(threshold):
        if isinstance(threshold, str):
            threshold_abs = threshold_func[threshold](t)
        else:
            threshold_abs = threshold

    if min_distance is None:
        if block_size is None:
            block_size = get_patch_size(t)
            print(block_size)
        min_distance = block_size//3

    pts = peak_local_max(t, min_distance=min_distance, threshold_abs=threshold_abs)
    pts[:, 0], pts[:, 1] = pts[:, 1], pts[:, 0].copy()
    if plot is True:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.imshow(img)
        ax.plot(pts[:, 0], pts[:, 1], 'r.')
    return pts