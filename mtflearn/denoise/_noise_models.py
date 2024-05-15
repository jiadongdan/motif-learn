import numpy as np

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