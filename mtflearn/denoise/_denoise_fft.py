import numpy as np


def denoise_fft(image, p):
    """
    Denoise a 2D image using FFT by retaining only the top p fraction
    of Fourier coefficients by power.

    Parameters:
    image (np.ndarray): Input 2D image array.
    p (float): Fraction of Fourier coefficients to keep. Must be between 0 and 1.

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

    # Use squared magnitude so the selection criterion matches "power".
    power_spectrum = np.abs(fft_image) ** 2
    flat_power_spectrum = power_spectrum.ravel()

    # Keep exactly the requested number of coefficients, even when many
    # coefficients tie at the cutoff.
    num_coeffs = flat_power_spectrum.size
    num_keep = int(np.ceil(p * num_coeffs))
    top_indices = np.argpartition(flat_power_spectrum, -num_keep)[-num_keep:]
    mask = np.zeros(num_coeffs, dtype=bool)
    mask[top_indices] = True
    mask = mask.reshape(power_spectrum.shape)

    # Apply the mask to the FFT image
    fft_filtered = fft_image * mask

    # Inverse FFT to get the denoised image
    denoised_image = np.fft.ifft2(fft_filtered)

    # Return the real part of the denoised image
    return np.real(denoised_image)
