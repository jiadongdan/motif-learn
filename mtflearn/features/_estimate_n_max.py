import numpy as np
from scipy.fft import fft2, fftshift
from skimage.restoration import estimate_sigma
from ._patch_size import radial_profile
from ..denoise._denoise_fft import denoise_fft

def _get_cumulative_energy(patch, window_type='hann', normalize=True,
                           return_profile=False, epsilon=1e-10):
    """
    Compute cumulative radial energy distribution from patch power spectrum.

    This function applies a window to reduce edge effects, computes the FFT,
    and calculates the cumulative energy as a function of spatial frequency.

    Parameters:
    -----------
    patch : ndarray
        Input image patch (2D)
    window_type : str or None
        Type of window function to apply:
        - 'hann' or 'hanning': Hann window (default)
        - 'hamming': Hamming window
        - 'blackman': Blackman window
        - 'tukey': Tukey window (tapered cosine)
        - None: No windowing
    normalize : bool
        If True, normalize cumulative energy to [0, 1] range
    return_profile : bool
        If True, also return the radial power profile
    epsilon : float
        Small constant for numerical stability

    Returns:
    --------
    cumulative_energy : ndarray
        Cumulative energy as function of radius
    profile : ndarray (optional)
        Radial power profile (if return_profile=True)
    """
    patch_size = patch.shape[0]

    # Apply window to reduce edge effects
    if window_type is not None:
        if window_type in ('hann', 'hanning'):
            window_1d = np.hanning(patch_size)
        elif window_type == 'hamming':
            window_1d = np.hamming(patch_size)
        elif window_type == 'blackman':
            window_1d = np.blackman(patch_size)
        elif window_type == 'tukey':
            from scipy.signal import windows
            window_1d = windows.tukey(patch_size, alpha=0.5)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

        window_2d = np.outer(window_1d, window_1d)
        patch_windowed = patch * window_2d
    else:
        patch_windowed = patch.copy()

    # Compute power spectrum
    fft_patch = fft2(patch_windowed)
    power = np.abs(fftshift(fft_patch))**2

    # Get radial profile
    profile = radial_profile(power)

    # Compute cumulative energy with proper weighting
    # Each annulus at radius r has circumference 2πr, so weight by r
    radii = np.arange(len(profile))
    weighted_power = profile * radii

    # Cumulative sum
    cumulative_energy = np.cumsum(weighted_power)

    # Normalize to [0, 1] if requested
    if normalize:
        total_energy = cumulative_energy[-1]
        if total_energy > epsilon:
            cumulative_energy = cumulative_energy / total_energy
        else:
            # Handle edge case of zero energy
            cumulative_energy = np.zeros_like(cumulative_energy)

    if return_profile:
        return cumulative_energy, profile
    return cumulative_energy

def estimate_n_max_from_patch(patch, p=0.01):
    patch_denoised = denoise_fft(patch, p=p)
    l_noise = _get_cumulative_energy(patch)
    l_clean = _get_cumulative_energy(patch_denoised)
    l = l_clean - l_noise
    n_max = min(max(12, np.argmax(l) * 2), patch.shape[0]//2)
    return n_max

def get_ps(img, n_samples, patch_size):
    h, w = img.shape
    ps = []
    for _ in range(n_samples):
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        patch = img[y:y+patch_size, x:x+patch_size]
        ps.append(patch)
    return np.array(ps)

def estimate_n_max(img, patch_size, n_samples=50, p=0.01):
    img_denoised = denoise_fft(img, p=p)
    ps = get_ps(img, patch_size=patch_size, n_samples=n_samples)
    ps_denoised = get_ps(img_denoised, patch_size=patch_size, n_samples=n_samples)
    n_max_list = []
    for (patch, patch_denoised) in zip(ps, ps_denoised):
        l_noise = _get_cumulative_energy(patch)
        l_clean = _get_cumulative_energy(patch_denoised)
        l = l_clean - l_noise
        n_max = min(max(12, np.argmax(l) * 2), patch.shape[0]//2)
        n_max_list.append(n_max)
    return np.median(n_max_list)