from scipy.signal import correlate
from skimage.transform import warp_polar


def autocorrelation(image, mode='same', method='fft', normalize=True):
    image_ = image.copy()
    # Normalize the image
    if normalize is True:
        image_ -= np.mean(image_)
        image_ /= np.std(image_)
    return correlate(image_, image, mode=mode, method=method)


def compute_autocorrelation(image, normalize=True):
    image_ = image.copy()
    # Normalize the image
    if normalize is True:
        image_ -= np.mean(image_)
        image_ /= np.std(image_)

    # Compute the FFT of the image
    f = np.fft.fft2(image_)
    # Use the properties of the FFT to compute the autocorrelation
    autocorr = np.fft.ifft2(f * np.conj(f))
    # Shift the zero-frequency component to the center of the spectrum
    autocorr = np.fft.fftshift(autocorr)

    return np.abs(autocorr)

def radial_profile(data):
    i, j = np.unravel_index(np.argmax(data), shape=data.shape)
    print(i, j)
    line = warp_polar(data, center=(i, j)).mean(axis=0)[0:i]
    return line