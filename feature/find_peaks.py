import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

from skimage.transform import warp_polar
from skimage.filters import (threshold_local,
                             threshold_li,
                             threshold_otsu,
                             threshold_mean,
                             threshold_minimum,
                             threshold_yen)
from skimage.feature import peak_local_max
from skimage.restoration import estimate_sigma




# https://stackoverflow.com/a/50160920/5855131
def baseline_als(y, lam=105, p=0.1, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# see here: https://stackoverflow.com/q/57350711/5855131
def baseline_correction(y,niter=10):
    n = len(y)
    y_ = np.log(np.log(np.sqrt(y +1)+1)+1)
    yy = np.zeros_like(y)

    for pp in np.arange(1,niter+1):
        r1 = y_[pp:n-pp]
        r2 = (np.roll(y_,-pp)[pp:n-pp] + np.roll(y_,pp)[pp:n-pp])/2
        yy = np.minimum(r1,r2)
        y_[pp:n-pp] = yy

    baseline = (np.exp(np.exp(y_)-1)-1)**2 -1
    return baseline


def get_window_size(img):
    fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    fft_log = np.log(fft_abs + 1)
    i ,j = np.unravel_index(np.argmax(fft_log), shape=img.shape)
    line = warp_polar(fft_log, center=(i, j)).mean(axis=0)[0:i]
    # get the baseline
    baseline = baseline_correction(line)
    # remove the baseline
    line_ = line - baseline
    ind = np.argmax(line_)
    size = np.ceil(img.shape[0]/ind)
    return int(size)

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
            block_size = get_window_size(img)
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
            block_size = get_window_size(t)
            print(block_size)
        min_distance = block_size

    pts = peak_local_max(t, min_distance=min_distance, threshold_abs=threshold_abs)
    pts[:, 0], pts[:, 1] = pts[:, 1], pts[:, 0].copy()

    if plot is True:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.imshow(img)
        ax.plot(pts[:, 0], pts[:, 1], 'r.')
    return pts


def line_peaks(line, plot=True, **kwargs):
    peaks = find_peaks(line, **kwargs)[0]
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.plot(line)
        ax.plot(peaks, line[peaks], 'x')
    return peaks

def get_bg(img, niter=10):
    img_ = np.log(np.log(np.sqrt(img +1)+1)+1)
    for i in np.arange(1, niter+1):
        a1 = img_[i:-i, i:-i]
        p1 = np.roll(img_, i, axis=[0, 1])[i:-i, i:-i]
        p2 = np.roll(np.roll(img_, -i, axis=0), i, axis=1)[i:-i, i:-i]
        p3 = np.roll(np.roll(img_, i, axis=0), -i, axis=1)[i:-i, i:-i]
        p4 = np.roll(img_, -i, axis=[0, 1])[i:-i, i:-i]
        s1 = np.roll(img_, i, axis=0)[i:-i, i:-i]
        s2 = np.roll(img_, i, axis=1)[i:-i, i:-i]
        s3 = np.roll(img_, -i, axis=1)[i:-i, i:-i]
        s4 = np.roll(img_, -i, axis=0)[i:-i, i:-i]
        s1 = np.maximum(s1, (p1+p3)/2)
        s2 = np.maximum(s2, (p1+p2)/2)
        s3 = np.maximum(s3, (p3+p4)/2)
        s4 = np.maximum(s4, (p2+p4)/2)
        a2 = (s1+s2+s3+s4)/2 - (p1+p2+p3+p4)/4
        v = np.minimum(a1, a2)
        img_[i:-i, i:-i] = v
    bg = (np.exp(np.exp(img_)-1)-1)**2 -1
    return bg