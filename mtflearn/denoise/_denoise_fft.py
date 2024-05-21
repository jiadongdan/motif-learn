import numpy as np


def denoise_fft(image, p):
    """
    Denoise a 2D image using FFT by retaining only the top p fraction of the power spectrum.

    Parameters:
    image (np.ndarray): Input 2D image array.
    p (float): Fraction of pixels to keep in Fourier space. Must be between 0 and 1.

    Returns:
    np.ndarray: Denoised image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if not (0 < p <= 1):
        raise ValueError("Fraction p must be between 0 and 1.")

    # Compute the 2D FFT of the image
    fft_image = np.fft.fft2(image)

    # Compute the power spectrum
    # power_spectrum = np.abs(fft_image) ** 2
    power_spectrum = np.abs(fft_image)
    # Flatten the power spectrum and sort the values in descending order
    flat_power_spectrum = power_spectrum.flatten()
    sorted_power_spectrum = np.sort(flat_power_spectrum)[::-1]

    # Determine the threshold value to keep the top p fraction of the power spectrum
    threshold_index = int(np.ceil(p * len(sorted_power_spectrum))) - 1
    threshold_value = sorted_power_spectrum[threshold_index]

    # Create a mask to retain only the top p fraction of the power spectrum
    mask = power_spectrum >= threshold_value

    # Apply the mask to the FFT image
    fft_filtered = fft_image * mask

    # Inverse FFT to get the denoised image
    denoised_image = np.fft.ifft2(fft_filtered)

    # Return the real part of the denoised image
    return np.real(denoised_image)
