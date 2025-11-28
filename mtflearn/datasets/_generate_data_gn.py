import numpy as np

def generate_data_gn(size, n=6, sigma=None, include_center=True, radius_frac=0.25, rotation_angle=0.0):
    """
    Generate a size×size grayscale image containing Gaussian blobs
    arranged with n-fold rotational symmetry. Optionally include a center blob
    and apply a global rotation offset in degrees.

    Parameters
    ----------
    size : int
        Width and height of the (square) output image.
    n : int, optional (default=6)
        Number of blobs placed around a circle (i.e., n‐fold symmetry).
    sigma : float or None, optional
        Standard deviation of each Gaussian. If None, sigma is set to size/24.
    include_center : bool, optional (default=True)
        Whether to add a Gaussian blob exactly at the image center.
    radius_frac : float, optional (default=0.25)
        Fraction of `size` to use as the circle radius. Actual radius = size * radius_frac.
    rotation_angle : float, optional (default=0.0)
        Global rotation offset in degrees. All blob angles are shifted by this amount.

    Returns
    -------
    img : ndarray, shape (size, size), dtype float
        A float array with values in [0, 1], containing the summed Gaussians.
    """
    if sigma is None:
        sigma = size / 24.0

    # Convert rotation angle from degrees to radians
    rot_rad = np.deg2rad(rotation_angle)

    # Create a meshgrid of coordinates (xv = column indices, yv = row indices)
    x = np.arange(size)
    y = np.arange(size)
    xv, yv = np.meshgrid(x, y)

    # Initialize the image to zero
    img = np.zeros((size, size), dtype=float)

    # Center of the image (using float so that blob centers need not land exactly on integer pixels)
    center = size / 2.0

    # Compute the radius from the fraction
    radius = size * radius_frac

    # Compute base angles for the n blobs (evenly spaced), then add rotation offset
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = base_angles + rot_rad

    # Add the n blobs on the circle
    for theta in angles:
        xc = center + radius * np.cos(theta)
        yc = center + radius * np.sin(theta)
        img += np.exp(-((xv - xc)**2 + (yv - yc)**2) / (2 * sigma**2))

    # Optionally add a blob at the exact center
    if include_center:
        img += np.exp(-((xv - center)**2 + (yv - center)**2) / (2 * sigma**2))

    # Normalize so that max value is 1.0 (i.e., in [0, 1])
    img /= img.max()

    return img
