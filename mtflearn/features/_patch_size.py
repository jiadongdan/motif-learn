import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
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

def radial_profile(data, center=None):
    """
    Compute radial average profile from 2D data.

    Parameters:
    -----------
    data : ndarray
        2D array (e.g., autocorrelation map, power spectrum after fftshift)
    center : tuple of (row, col), optional
        Center point for radial averaging. If None, uses geometric center
        at (h//2, w//2) to match fftshift convention.

    Returns:
    --------
    profile : ndarray
        Radial average profile from center outward
    """
    h, w = data.shape

    if center is None:
        # Use fftshift convention: DC is at [h//2, w//2]
        center = (h // 2, w // 2)

    i, j = center

    # Maximum radius (distance to nearest edge)
    max_radius = min(i, j, h - i - 1, w - j - 1)

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    # Calculate distance from center
    r = np.sqrt((x - j)**2 + (y - i)**2)

    # Bin by integer radius
    r_int = r.astype(int)

    # Compute radial average
    profile = np.bincount(r_int.ravel(), weights=data.ravel(), minlength=max_radius+1)
    counts = np.bincount(r_int.ravel(), minlength=max_radius+1)

    # Avoid division by zero
    mask = counts > 0
    profile[mask] /= counts[mask]

    # Trim to max_radius
    profile = profile[:max_radius]

    return profile


def find_first_peak(radial_profile, min_distance=5, prominence_factor=0.15,
                    min_width=2, smooth_sigma=1.0, max_distance=None, debug=False):
    """
    Find first significant peak in autocorrelation radial profile.

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
    first_peak : int or None
        Position of first peak in pixels, or None if no peak found
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
            ax.scatter(peak_positions, smoothed[peaks],
                       color='red', s=150, zorder=5, label='Detected Peaks',
                       marker='o', edgecolors='darkred', linewidths=2)

            # Annotate with spacing info
            for i, (peak, width, prom) in enumerate(zip(peaks, properties['widths'],
                                                        properties['prominences'])):
                ax.annotate(f'Peak {i+1}\nr={peak+min_distance} px\nwidth={width:.1f}',
                            xy=(peak + min_distance, smoothed[peak]),
                            xytext=(15, 15), textcoords='offset points',
                            fontsize=9, ha='left',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                # Show peak width at half-prominence
                left_ips = properties['left_ips'][i]
                right_ips = properties['right_ips'][i]
                height_at_width = smoothed[peak] - prom/2
                ax.hlines(height_at_width,
                          left_ips + min_distance, right_ips + min_distance,
                          color='orange', linewidth=3, alpha=0.7, label='Peak Width' if i == 0 else '')

        ax.axvline(min_distance, color='gray', linestyle='--', alpha=0.5,
                   linewidth=2, label='Search Start')
        ax.set_xlabel('Radius (pixels) - Real Space Distance', fontsize=12)
        ax.set_ylabel('Autocorrelation Strength', fontsize=12)
        ax.set_title('Autocorrelation Radial Profile - Peak Detection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    if len(peaks) == 0:
        return None, None, properties

    first_peak = peaks[0] + min_distance
    all_peaks = peaks + min_distance

    return first_peak, all_peaks, properties


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

    # Compute radial profile
    line = radial_profile(autocorr_mean)

    # Search up to half the radius to avoid edge artifacts
    max_search = len(line)

    # Find peaks
    peak, all_peaks, props = find_first_peak(
        line,
        min_distance=min_distance,
        prominence_factor=prominence_factor,
        min_width=min_width,
        smooth_sigma=smooth_sigma,
        max_distance=max_search,
        debug=False
    )

    if debug:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

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

        if peak is not None:
            ax3.axvline(peak, color='red', linestyle='--', linewidth=2.5,
                        label=f'Lattice Spacing: {peak} px', zorder=10)

            if all_peaks is not None and len(all_peaks) > 0:
                ax3.scatter(all_peaks, line[all_peaks], color='green',
                            marker='o', s=200, linewidths=2,
                            edgecolors='darkgreen',
                            label=f'All Peaks (n={len(all_peaks)})', zorder=5)

                # Annotate each peak with width and relative prominence
                if 'widths' in props and 'prominences' in props:
                    widths = props['widths']
                    prominences = props['prominences']

                    # Calculate profile range for normalized prominence
                    profile_range = np.ptp(line[min_distance:max_search])

                    for i, pos in enumerate(all_peaks):
                        peak_height = line[pos]
                        width = widths[i]
                        prom = prominences[i]
                        rel_prom = prom / profile_range  # Relative to entire profile

                        # Clear annotation - no confusion with 'r' for radius
                        annotation = f'r={pos}px\nw={width:.1f}\np_norm={rel_prom:.2f}'
                        # Or: annotation = f'r={pos}px\nw={width:.1f}\nprom={rel_prom*100:.0f}%'

                        color = 'red' if i == 0 else 'green'
                        bbox_color = 'yellow' if i == 0 else 'lightgreen'

                        ax3.annotate(
                            annotation,
                            xy=(pos, peak_height),
                            xytext=(15, 15 if i % 2 == 0 else -45),
                            textcoords='offset points',
                            fontsize=8,
                            ha='left',
                            color=color,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor=bbox_color,
                                      edgecolor=color,
                                      alpha=0.8,
                                      linewidth=1.5),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=0.3',
                                            color=color,
                                            linewidth=1.5)
                        )


                # Annotate peak multiples (spacing ratios)
                if len(all_peaks) > 1:
                    ratios = all_peaks / peak
                    for i, (pos, ratio) in enumerate(zip(all_peaks[1:], ratios[1:]), 1):
                        ax3.text(pos, line[pos] * 1.15, f'{ratio:.1f}×',
                                 fontsize=10, ha='center', va='bottom',
                                 fontweight='bold', color='darkgreen')

            # Add text box with results
            textstr = f'First Peak: {peak} px\nWindow Size: {window_size} px\nSamples: {n_samples}'
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
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

    return peak

def estimate_patch_size_robust(img, standardize=True, n_samples=None,
                               min_distance=5, prominence_factor=0.15,
                               min_width=2, smooth_sigma=1.0,
                               n_window_sizes=5, debug=False):
    """
    Robustly estimate lattice spacing by testing multiple window sizes.

    This function runs estimate_patch_size with different window sizes ranging
    from 1/3 to 1/2 of the image size and returns the most consistent result.

    Parameters:
    -----------
    img : ndarray
        Input image (2D grayscale)
    standardize : bool
        Whether to standardize patches before autocorrelation
    n_samples : int, optional
        Number of random patches to sample per window size
    min_distance : int
        Minimum distance from DC peak to start searching (pixels)
    prominence_factor : float
        Minimum peak prominence as fraction of profile range
    min_width : float
        Minimum peak width to exclude shoulders
    smooth_sigma : float
        Gaussian smoothing for noise reduction
    n_window_sizes : int
        Number of different window sizes to test (default: 5)
    debug : bool
        If True, display detailed diagnostic plots for each window size

    Returns:
    --------
    peak : int or None
        Estimated lattice spacing in pixels (median of valid results)
    results : dict
        Dictionary containing:
        - 'all_peaks': list of peaks found for each window size
        - 'window_sizes': list of window sizes tested
        - 'median': median peak value
        - 'mean': mean peak value
        - 'std': standard deviation of peaks
    """
    h, w = img.shape
    min_size = min(h, w)

    # Generate window sizes from 1/3 to 1/2 of image size
    window_sizes = np.linspace(min_size // 3, min_size // 2, n_window_sizes, dtype=int)
    window_sizes = np.unique(window_sizes)  # Remove duplicates

    peaks = []
    valid_window_sizes = []

    if debug:
        print(f"Testing {len(window_sizes)} window sizes: {window_sizes}")
        print("-" * 60)

    for window_size in window_sizes:
        if debug:
            print(f"\nWindow size: {window_size} px")

        peak = estimate_patch_size(
            img=img,
            window_size=window_size,
            standardize=standardize,
            n_samples=n_samples,
            min_distance=min_distance,
            prominence_factor=prominence_factor,
            min_width=min_width,
            smooth_sigma=smooth_sigma,
            debug=debug
        )

        if peak is not None:
            peaks.append(peak)
            valid_window_sizes.append(window_size)
            if debug:
                print(f"  → Found peak at {peak} px")
        else:
            if debug:
                print(f"  → No peak found")

    if len(peaks) == 0:
        if debug:
            print("\n" + "=" * 60)
            print("No valid peaks found across any window size")
            print("=" * 60)
        return None, {
            'all_peaks': [],
            'window_sizes': list(window_sizes),
            'median': None,
            'mean': None,
            'std': None
        }

    # Calculate statistics
    peaks_array = np.array(peaks)
    median_peak = int(np.median(peaks_array))
    mean_peak = np.mean(peaks_array)
    std_peak = np.std(peaks_array)

    results = {
        'all_peaks': peaks,
        'window_sizes': valid_window_sizes,
        'median': median_peak,
        'mean': mean_peak,
        'std': std_peak
    }

    if debug:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"Valid results: {len(peaks)}/{len(window_sizes)}")
        print(f"Window sizes tested: {list(window_sizes)}")
        print(f"Window sizes with peaks: {valid_window_sizes}")
        print(f"Peaks found: {peaks}")
        print(f"Median peak: {median_peak} px")
        print(f"Mean peak: {mean_peak:.1f} px")
        print(f"Std dev: {std_peak:.1f} px")
        print(f"Coefficient of variation: {(std_peak/mean_peak)*100:.1f}%")
        print("=" * 60)

        # Create summary visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Peaks vs window size
        ax1.plot(valid_window_sizes, peaks, 'o-', markersize=10, linewidth=2, label='Detected peaks')
        ax1.axhline(median_peak, color='red', linestyle='--', linewidth=2, label=f'Median: {median_peak} px')
        ax1.fill_between(valid_window_sizes,
                         mean_peak - std_peak,
                         mean_peak + std_peak,
                         alpha=0.3, color='orange', label=f'Mean ± Std')
        ax1.set_xlabel('Window Size (pixels)', fontsize=12)
        ax1.set_ylabel('Detected Lattice Spacing (pixels)', fontsize=12)
        ax1.set_title('Lattice Spacing vs Window Size', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # Plot 2: Histogram of peaks
        ax2.hist(peaks, bins=min(10, len(peaks)), alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axvline(median_peak, color='red', linestyle='--', linewidth=2.5, label=f'Median: {median_peak} px')
        ax2.axvline(mean_peak, color='orange', linestyle='--', linewidth=2.5, label=f'Mean: {mean_peak:.1f} px')
        ax2.set_xlabel('Lattice Spacing (pixels)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Detected Peaks', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    return median_peak, results