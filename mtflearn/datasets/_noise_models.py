import numpy as np

def apply_poisson_noise(img, counts_per_pixel, return_counts=False, seed=None):
    """
    Apply Poisson (shot) noise to a normalized image.

    Parameters
    ----------
    img : ndarray
        Clean image in [0, 1]. 1.0 corresponds to `counts_per_pixel` expected counts.
    counts_per_pixel : float
        Expected counts for a pixel with intensity 1.0.
    return_counts : bool
        If True, also return the raw Poisson counts.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    noisy_img : ndarray
        Noisy image in [0, 1].
    noisy_counts : ndarray, optional
        Raw Poisson counts (only if return_counts=True).
    """
    img = np.asarray(img, dtype=np.float32)
    if counts_per_pixel <= 0:
        raise ValueError("counts_per_pixel must be positive.")

    rng = np.random.default_rng(seed)

    lam = img * counts_per_pixel
    noisy_counts = rng.poisson(lam).astype(np.float32)
    noisy_img = noisy_counts / counts_per_pixel

    if return_counts:
        return noisy_img, noisy_counts
    return noisy_img


def add_gaussian_noise(img, sigma=0.1, seed=None):
    """
    Add zero-mean Gaussian noise to an image.

    Parameters
    ----------
    img : ndarray
        Input image (float). Can be any range, but typically [0, 1].
    sigma : float
        Standard deviation of Gaussian noise.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    noisy_img : ndarray
        Image with additive Gaussian noise.
    """
    img = np.asarray(img, dtype=np.float32)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)

    noisy = img + noise
    return noisy


def apply_poisson_gaussian_noise(img, counts_per_pixel, sigma, seed=None):
    """
    Apply Poisson (shot) + Gaussian (readout) noise.

    Model:
        k_ij  ~ Poisson(counts_per_pixel * img_ij)
        y_ij  = k_ij / counts_per_pixel + N(0, sigma)

    Parameters
    ----------
    img : ndarray
        Clean image in [0, 1].
    counts_per_pixel : float
        Mean number of counts for intensity = 1.
    sigma : float
        Std of Gaussian noise after normalization.
    seed : int or None
        RNG seed.

    Returns
    -------
    noisy_img : ndarray
        Image with Poisson + Gaussian noise.
    """
    img = np.asarray(img, dtype=np.float32)
    if counts_per_pixel <= 0:
        raise ValueError("counts_per_pixel must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    rng = np.random.default_rng(seed)

    # Poisson
    lam = img * counts_per_pixel
    noisy_counts = rng.poisson(lam).astype(np.float32)
    poisson_part = noisy_counts / counts_per_pixel

    # Gaussian
    gaussian_part = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)

    # Combine
    noisy_img = poisson_part + gaussian_part
    return noisy_img


def estimate_counts_per_pixel_mle(noisy_img, clean_img, mask=None, s_min=1e-3):
    """
    Estimate counts_per_pixel (C) from noisy+clean images.

    Uses a Gaussian approximation to the Poisson likelihood.

    Parameters
    ----------
    noisy_img : ndarray
        Observed noisy image.
    clean_img : ndarray
        Clean template image.
    mask : ndarray[bool], optional
        Restrict fit to masked pixels.
    s_min : float
        Minimum clean intensity allowed for fitting.

    Returns
    -------
    C_hat : float
        Estimated counts_per_pixel.
    """
    noisy = np.asarray(noisy_img, dtype=np.float64)
    clean = np.asarray(clean_img, dtype=np.float64)

    if noisy.shape != clean.shape:
        raise ValueError("noisy_img and clean_img must have same shape.")

    valid = clean >= s_min

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != noisy.shape:
            raise ValueError("mask must match image shape.")
        valid &= mask

    noisy = noisy[valid]
    clean = clean[valid]

    if noisy.size == 0:
        raise ValueError("No valid pixels to fit counts_per_pixel.")

    residual = noisy - clean
    A = (residual**2) / clean

    sum_A = A.sum()
    N_eff = A.size

    if sum_A <= 0:
        return np.inf

    return N_eff / sum_A

