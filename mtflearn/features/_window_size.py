import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.transform import warp_polar
from ..utils._preprocessing_image import standardize_image


def autocorrelation(image, mode='same', method='fft', standardize=True):
    # standardize the image
    if standardize is True:
        image = standardize_image(image)
    return correlate(image, image, mode=mode, method=method)


def compute_autocorrelation(image, standardize=True):
    # standardize the image
    if standardize is True:
        image = standardize_image(image)

    # Compute the FFT of the image
    f = np.fft.fft2(image)
    # Use the properties of the FFT to compute the autocorrelation
    autocorr = np.fft.ifft2(f * np.conj(f))
    # Shift the zero-frequency component to the center of the spectrum
    autocorr = np.fft.fftshift(autocorr)
    return np.abs(autocorr)


def radial_profile(data):
    i, j = np.unravel_index(np.argmax(data), shape=data.shape)
    line = warp_polar(data, center=(i, j)).mean(axis=0)[0:i]
    return line


def get_characteristic_length(image, standardize=True, debug=False):
    autocorr = autocorrelation(image=image, standardize=standardize)
    line_profile = radial_profile(autocorr)
    peaks, _ = find_peaks(line_profile)
    if debug:
        plt.plot(peaks, line_profile[peaks], "x")
        plt.plot(line_profile)
    if len(peaks) > 0:
        return peaks[0]
    else:
        raise ValueError("No peak detected in the radial profile.")


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


def get_characteristic_length_fft(image, niter=10, size=9, use_log=True, debug=True):

    fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    if use_log:
        fft_log = np.log(fft_abs + 1)
    else:
        fft_log = fft_abs
    i ,j = np.unravel_index(np.argmax(fft_log), shape=image.shape)
    y = warp_polar(fft_log, center=(i, j)).mean(axis=0)[0:i]
    bg = baseline_correction(y, niter=niter)
    y1 = y - bg
    y2 = uniform_filter1d(y1,size=size)

    ind = np.argmax(y2)
    size = image.shape[0] / ind
    return size

