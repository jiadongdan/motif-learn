import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d, median_filter
from skimage.transform import warp_polar


def standardize_image(image):
    """Standardize image to zero mean and unit variance."""
    mean = np.mean(image)
    std = np.std(image)

    if std == 0:
        raise ValueError("Standard deviation is zero, can't standardize the image.")

    standardized_image = (image - mean) / std
    return standardized_image


def autocorrelation(image, mode='same', method='fft', standardize=True):
    """
    Compute autocorrelation of an image.

    Parameters:
    -----------
    image : ndarray
        Input image
    mode : str
        Correlation mode ('same', 'valid', 'full')
    method : str
        Method for correlation ('fft' or 'direct')
    standardize : bool
        Whether to standardize the image first

    Returns:
    --------
    autocorr : ndarray
        Autocorrelation map
    """
    if standardize:
        image = standardize_image(image)
    return correlate(image, image, mode=mode, method=method)

def radial_profile(data, center=None, method="max"):
    """
    Compute radial profile from 2D data using polar coordinate transformation.

    Parameters:
    -----------
    data : ndarray
        2D array (e.g., autocorrelation map, power spectrum after fftshift)
    center : tuple of (row, col), optional
        Center point for radial averaging. If None, uses geometric center
        at (h//2, w//2) to match fftshift convention.
    method : str, optional
        Aggregation method for radial profile. Options:
        - "mean" : radial average (default)
        - "max"  : radial maximum
        - "sum"  : radial sum

    Returns:
    --------
    profile : ndarray
        Radial profile from center outward
    """
    h, w = data.shape

    if center is None:
        # Use fftshift convention: DC is at [h//2, w//2]
        center = (h // 2, w // 2)

    i, j = center

    # Maximum radius (distance to nearest edge)
    max_radius = min(i, j, h - i - 1, w - j - 1)

    # Convert to polar coordinates
    # warp_polar uses (row, col) convention for center
    polar_image = warp_polar(
        data,
        center=center,
        scaling='linear'
    )
    # polar_image shape (360, N)

    # Aggregate along angular dimension based on method
    if method == "mean":
        profile = np.mean(polar_image, axis=0)
    elif method == "max":
        profile = np.max(polar_image, axis=0)
    elif method == "sum":
        profile = np.sum(polar_image, axis=0)
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'mean', 'max', or 'sum'.")

    return profile

def find_highest_peak(radial_profile, min_distance=5, prominence_factor=0.15,
                      min_width=2, smooth_sigma=1.0, max_distance=None, debug=False):
    """
    Find highest (maximum intensity) peak in autocorrelation radial profile.

    Parameters:
    -----------
    radial_profile : ndarray
        1D radial average profile
    min_distance : int
        Skip central DC peak (recommend 5-10 for autocorrelation)
    prominence_factor : float
        Minimum prominence as fraction of profile range (0.15-0.25 recommended)
    min_width : float
        Minimum peak width in pixels to exclude noise shoulders (2-3 recommended)
    smooth_sigma : float
        Gaussian smoothing sigma to reduce noise (1.0-2.0 recommended)
    max_distance : int, optional
        Maximum search radius (avoid edge artifacts)
    debug : bool
        If True, plot detailed peak detection visualization

    Returns:
    --------
    highest_peak : int or None
        Position of highest peak in pixels, or None if no peak found
    all_peaks : ndarray or None
        Positions of all detected peaks
    properties : dict
        Peak properties from scipy.signal.find_peaks
    """
    search_profile = radial_profile[min_distance:]

    if max_distance is not None:
        search_profile = search_profile[:max_distance - min_distance]

    # Light smoothing to reduce noise
    smoothed = gaussian_filter1d(search_profile, sigma=smooth_sigma)

    profile_range = np.ptp(smoothed)
    min_prominence = prominence_factor * profile_range

    # Find peaks with multiple criteria
    peaks, properties = find_peaks(
        smoothed,
        prominence=min_prominence,
        width=min_width,
        distance=3  # Prevent detecting noise as multiple peaks
    )

    if debug:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(np.arange(len(search_profile)) + min_distance,
                search_profile, alpha=0.5, label='Raw Profile', linewidth=1.5, color='steelblue')
        ax.plot(np.arange(len(smoothed)) + min_distance,
                smoothed, label='Smoothed', linewidth=2.5, color='navy')

        if len(peaks) > 0:
            peak_positions = peaks + min_distance

            # Find the highest peak
            highest_idx = np.argmax(smoothed[peaks])
            highest_peak_pos = peaks[highest_idx]

            # Plot all peaks
            ax.scatter(peak_positions, smoothed[peaks],
                       color='red', s=100, zorder=5, label='All Peaks',
                       marker='o', edgecolors='darkred', linewidths=1.5, alpha=0.6)

            # Highlight the highest peak
            ax.scatter(highest_peak_pos + min_distance, smoothed[highest_peak_pos],
                       color='lime', s=250, zorder=6, label='Highest Peak',
                       marker='*', edgecolors='darkgreen', linewidths=2)

            # Annotate all peaks with just position
            for peak in peaks:
                ax.annotate(f'r={peak+min_distance}',
                            xy=(peak + min_distance, smoothed[peak]),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=8, ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.6))

            # Detailed annotation for highest peak only
            width = properties['widths'][highest_idx]
            prom = properties['prominences'][highest_idx]
            ax.annotate(f'Highest Peak\nr={highest_peak_pos+min_distance} px\nwidth={width:.1f}\nprom={prom:.2e}',
                        xy=(highest_peak_pos + min_distance, smoothed[highest_peak_pos]),
                        xytext=(20, 20), textcoords='offset points',
                        fontsize=10, ha='left', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                        lw=2, color='darkgreen'))

            # Show peak width for highest peak
            left_ips = properties['left_ips'][highest_idx]
            right_ips = properties['right_ips'][highest_idx]
            height_at_width = smoothed[highest_peak_pos] - prom/2
            ax.hlines(height_at_width,
                      left_ips + min_distance, right_ips + min_distance,
                      color='orange', linewidth=3, alpha=0.7, label='Peak Width')

        ax.axvline(min_distance, color='gray', linestyle='--', alpha=0.5,
                   linewidth=2, label='Search Start')
        ax.set_xlabel('Radius (pixels) - Real Space Distance', fontsize=12)
        ax.set_ylabel('Autocorrelation Strength', fontsize=12)
        ax.set_title('Autocorrelation Radial Profile - Highest Peak Detection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    if len(peaks) == 0:
        return None, None, properties

    # Find highest peak by intensity
    highest_idx = np.argmax(smoothed[peaks])
    highest_peak = peaks[highest_idx] + min_distance
    all_peaks = peaks + min_distance

    return highest_peak, all_peaks, properties


def estimate_patch_size(img, window_size=None, standardize=True, n_samples=None,
                        min_distance=5, prominence_factor=0.15,
                        min_width=2, smooth_sigma=1.0, debug=False):
    """
    Estimate lattice spacing from image using autocorrelation analysis.

    Parameters:
    -----------
    img : ndarray
        Input image (2D grayscale)
    window_size : int
        Size of square windows to extract for analysis
    standardize : bool
        Whether to standardize patches before autocorrelation
    n_samples : int, optional
        Number of random patches to sample (default: min(100, available patches))
    min_distance : int
        Minimum distance from DC peak to start searching (pixels)
    prominence_factor : float
        Minimum peak prominence as fraction of profile range
    min_width : float
        Minimum peak width to exclude shoulders
    smooth_sigma : float
        Gaussian smoothing for noise reduction
    debug : bool
        If True, display detailed diagnostic plots

    Returns:
    --------
    peak : int or None
        Estimated lattice spacing in pixels, or None if no peak found
    """
    h, w = img.shape

    if window_size is None:
        window_size = h // 2

    # Calculate number of samples
    if n_samples is None:
        n_samples = min(100, (h // window_size) * (w // window_size))

    if n_samples == 0:
        raise ValueError(f"Window size {window_size} is too large for image of size {img.shape}")

    # Extract random patches and compute autocorrelations
    autocorr_maps = []
    for _ in range(n_samples):
        y = np.random.randint(0, h - window_size)
        x = np.random.randint(0, w - window_size)
        patch = img[y:y+window_size, x:x+window_size]

        autocorr = autocorrelation(image=patch, standardize=standardize)
        autocorr_maps.append(autocorr)

    # Average autocorrelation maps
    autocorr_mean = np.mean(autocorr_maps, axis=0)
    autocorr_mean = median_filter(autocorr_mean, size=3) # handle the central DC

    # Compute radial profile
    line = radial_profile(autocorr_mean, method="max")

    # Search up to half the radius to avoid edge artifacts
    max_search = len(line)

    # Find peaks
    peak, all_peaks, props = find_highest_peak(
        line,
        min_distance=min_distance,
        prominence_factor=prominence_factor,
        min_width=min_width,
        smooth_sigma=smooth_sigma,
        max_distance=max_search,
        debug=False
    )

    if debug:
        plot_autocorr_analysis(
            autocorr_mean, img, window_size, line, peak, all_peaks,
            props, min_distance, max_search, n_samples
        )

    return peak


def plot_autocorr_analysis(autocorr_mean, img, window_size, line, peak, all_peaks,
                           props, min_distance, max_search, n_samples):
    """
    Plot autocorrelation analysis with 3-panel visualization.

    Parameters:
    -----------
    autocorr_mean : ndarray
        Mean autocorrelation map
    img : ndarray
        Original image
    window_size : int
        Size of analysis window
    line : ndarray
        Radial profile array
    peak : int or None
        Detected highest peak position
    all_peaks : ndarray or None
        All detected peak positions
    props : dict
        Peak properties from find_peaks
    min_distance : int
        Minimum search distance
    max_search : int
        Maximum search distance
    n_samples : int
        Number of samples used
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])

    # Autocorrelation map
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(autocorr_mean, cmap='RdBu_r')
    ax1.set_title('Mean Autocorrelation Map', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Original image sample
    ax2 = fig.add_subplot(gs[1])
    sample_patch = img[:window_size, :window_size]
    im2 = ax2.imshow(sample_patch, cmap='gray')
    ax2.set_title(f'Sample Patch ({window_size}×{window_size})', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Radial profile
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(line, label='Radial Profile', linewidth=2.5, alpha=0.8, color='navy')

    if peak is not None and all_peaks is not None and len(all_peaks) > 0:
        # Find the highest peak by intensity
        highest_idx = np.argmax(line[all_peaks])
        highest_peak = all_peaks[highest_idx]

        # Plot all peaks as circles
        ax3.scatter(all_peaks, line[all_peaks], color='red',
                    marker='o', s=100, linewidths=1.5,
                    edgecolors='darkred', alpha=0.6,
                    label=f'All Peaks (n={len(all_peaks)})', zorder=5)

        # Highlight the highest peak
        ax3.scatter(highest_peak, line[highest_peak], color='lime',
                    marker='*', s=250, linewidths=2,
                    edgecolors='darkgreen',
                    label=f'Highest Peak: {highest_peak} px', zorder=6)

        # Annotate all peaks with just position
        for pos in all_peaks:
            if pos != highest_peak:
                ax3.annotate(f'r={pos}',
                             xy=(pos, line[pos]),
                             xytext=(0, 10),
                             textcoords='offset points',
                             fontsize=8, ha='center',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='lightgray', alpha=0.6))

        # Detailed annotation for highest peak only
        if 'widths' in props and 'prominences' in props:
            widths = props['widths']
            prominences = props['prominences']
            profile_range = np.ptp(line[min_distance:max_search])

            width = widths[highest_idx]
            prom = prominences[highest_idx]
            rel_prom = prom / profile_range

            annotation = f'Highest Peak\nr={highest_peak} px\nwidth={width:.1f}\nprom={rel_prom:.2f}'

            ax3.annotate(
                annotation,
                xy=(highest_peak, line[highest_peak]),
                xytext=(20, 20),
                textcoords='offset points',
                fontsize=10,
                ha='left',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='yellow',
                          edgecolor='darkgreen',
                          alpha=0.8,
                          linewidth=2),
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3,rad=0.2',
                                color='darkgreen',
                                linewidth=2)
            )

        # Add text box with results
        textstr = f'Highest Peak: {peak} px\nWindow Size: {window_size} px\nSamples: {n_samples}'
        props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax3.text(0.98, 0.97, textstr, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props_box)
    else:
        ax3.text(0.5, 0.5, 'No significant peak found',
                 transform=ax3.transAxes, fontsize=14, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    ax3.axvline(min_distance, color='gray', linestyle=':', alpha=0.5,
                linewidth=2, label='Search Start')
    ax3.set_xlabel('Radius (pixels) - Real Space Distance', fontsize=11)
    ax3.set_ylabel('Autocorrelation Strength', fontsize=11)
    ax3.set_title('Radial Profile from Autocorrelation', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
