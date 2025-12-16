import numpy as np


def fft_power_spectrum(img):
    spectrum_complex = np.fft.fftshift(np.fft.fft2(img))
    spectrum = np.abs(spectrum_complex)
    return spectrum


import numpy as np

def angular_average(power_spectrum, num_bins=360, use_mask=True):
    """
    Compute the angularly averaged power spectrum in angular bins.

    Parameters:
    -----------
    power_spectrum : 2D numpy array
        The 2D FFT power spectrum (typically |FFT|^2)
    num_bins : int
        Number of angular bins (default 360 for 1-degree bins)
    use_mask : bool
        If True, only include pixels within a circular mask (default True)

    Returns:
    --------
    angular_profile : 1D numpy array
        The power spectrum averaged over each angular bin
    angles : 1D numpy array
        The angle values (in degrees) corresponding to each bin
    """
    # Get the center of the array
    center_y, center_x = np.array(power_spectrum.shape) // 2

    # Create coordinate arrays
    y, x = np.indices(power_spectrum.shape)

    # Calculate radial distance and angle from center for each pixel
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)

    # Convert to degrees and shift to 0-360 range
    theta_deg = np.degrees(theta) % 360

    # Create circular mask if requested
    if use_mask:
        max_radius = min(center_y, center_x)
        circular_mask = r <= max_radius
    else:
        circular_mask = np.ones_like(power_spectrum, dtype=bool)

    # Bin the angles
    angular_bins = np.linspace(0, 360, num_bins + 1)
    angles = (angular_bins[:-1] + angular_bins[1:]) / 2  # bin centers

    # Calculate the angular average
    angular_profile = np.zeros(num_bins)

    for i in range(num_bins):
        angle_mask = (theta_deg >= angular_bins[i]) & (theta_deg < angular_bins[i + 1])
        combined_mask = angle_mask & circular_mask

        if combined_mask.any():
            angular_profile[i] = power_spectrum[combined_mask].mean()

    return angles, angular_profile