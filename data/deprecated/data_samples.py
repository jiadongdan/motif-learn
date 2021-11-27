import numpy as np
from .poisson_disc_sampling import poisson_disc_samples

from ..io import load_pickle

import os

def gaussian2D(sigma, shape):
    h, w = shape
    Y, X = np.ogrid[0:h:h*1j, 0:w:w*1j]
    return np.exp(-((X-w/2)**2+(Y-h/2)**2)/(2*sigma*sigma))

def generate_random_pts(height, width, min_distance):
    pts = poisson_disc_samples(width-1, height-1, min_distance)
    return pts

def make_blobs(pts, sigma, shape):
    lattice = np.zeros(shape)
    for (x, y) in np.round(pts):
        lattice[int(y), int(x)] = 1.0
    # Generate psf function using sigma
    psf = gaussian2D(sigma, shape)
    return np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.fft2(lattice)*np.fft.fft2(psf))))

def periodic():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/periodic.pkl'
    return load_pickle(file_name)

def mose2():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/MoSe2.pkl'
    return load_pickle(file_name)

def lattice():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/lattice.pkl'
    return load_pickle(file_name)
