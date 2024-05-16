import numpy as np
from scipy.ndimage import map_coordinates


def apply_poisson_noise(image, dose_per_pixel=100):
    """
    Applies Poisson noise to an image given a dose per pixel.

    :param image: 2D numpy array representing the image
    :param dose_per_pixel: Average number of photons (or events) per pixel, which corresponds to lambda in Poisson distribution
    :return: 2D numpy array with Poisson noise applied
    """
    # Scale the image by the dose per pixel
    scaled_image = image * dose_per_pixel

    # Apply Poisson noise, numpy.random.poisson expects the lambda parameter for each pixel
    noisy_image = np.random.poisson(scaled_image)

    # Scale back if necessary, depending on the maximum value of the original image
    # and your specific needs (e.g., uint8 format with max value 255)
    max_val = np.max(image)
    if max_val > 0:
        noisy_image = (noisy_image / np.max(noisy_image)) * max_val

    return noisy_image.astype(image.dtype)


def apply_scan_noise(images, jx=1, jy=0, seed=None, order=1, mode='reflect'):
    if jx is None:
        jx = 0
    if jy is None:
        jy = 0

    # Get the shape of the 3D array
    z_dim, y_dim, x_dim = images.shape

    # Create a 3D grid of coordinates
    z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')

    # Initialize the random number generator
    rng = np.random.default_rng(seed=seed)

    # Generate noise for each dimension
    dx = rng.normal(0, 1, (z_dim, y_dim, 1)) * jx
    dy = rng.normal(0, 1, (z_dim, 1, x_dim)) * jy

    # Apply the noise to the coordinates
    x_noisy = x + dx
    y_noisy = y + dy

    # Stack the noisy coordinates to pass to map_coordinates
    coords = np.array([z, y_noisy, x_noisy])

    # Apply map_coordinates to the entire 3D array
    noisy_images = map_coordinates(images, coords, order=order, mode=mode)

    return noisy_images
