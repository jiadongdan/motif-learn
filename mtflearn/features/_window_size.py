import numpy as np
from scipy.signal import correlate
from skimage.transform import warp_polar


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
