import numpy as np
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree


def filter_peaks_by_distance(image, peaks, min_distance):
    """
    Filter peaks based on minimum distance and intensity.

    Parameters:
        image (ndarray): The original image, used to get peak intensities.
        peaks (ndarray): Initial array of peak coordinates, shape (N, 2), where each row is (row, col).
        min_distance (float): The minimum distance between peaks (can be a decimal).

    Returns:
        ndarray: Array of filtered peak coordinates.
    """
    # Get the intensity of each peak
    intensities = np.array([image[j, i] for (i, j) in peaks])

    # Sort by intensity in descending order
    sorted_indices = np.argsort(intensities)[::-1]
    peaks = peaks[sorted_indices]
    intensities = intensities[sorted_indices]

    # Use cKDTree to accelerate distance calculations
    tree = cKDTree(peaks)
    keep = np.ones(len(peaks), dtype=bool)  # Mark which peaks to keep

    for i, peak in enumerate(peaks):
        if not keep[i]:  # Skip if the current peak has already been suppressed
            continue

        # Find all neighbors within the min_distance
        neighbors = tree.query_ball_point(peak, r=min_distance)

        # Suppress all neighbors of the current peak (except itself)
        for neighbor in neighbors:
            if neighbor != i:
                keep[neighbor] = False

    # Return the kept peaks
    return peaks[keep]


def local_max(image, min_distance, threshold=None):
    """
    Custom peak detection function that combines `peak_local_max` and distance filtering.

    Parameters:
        image (ndarray): Input image.
        min_distance (float): The minimum distance between peaks (can be a decimal).
        threshold_abs (float): Absolute threshold for peak intensity.

    Returns:
        ndarray: Final array of filtered peak coordinates.
    """
    if threshold is None:
        threshold = image.min()
    # Initially detect all possible peaks
    peaks = peak_local_max(image, min_distance=1, threshold_abs=threshold)
    peaks[:, 0], peaks[:, 1] = peaks[:, 1], peaks[:, 0].copy()

    # Filter peaks based on minimum distance and intensity
    filtered_peaks = filter_peaks_by_distance(image, peaks, min_distance)
    return filtered_peaks

