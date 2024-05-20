import numpy as np
from scipy.signal import correlate
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
