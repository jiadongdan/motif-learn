import numpy as np
from scipy.ndimage import map_coordinates


def rng_seeds(seed, N):
    INT32_MAX = np.iinfo(np.int32).max - 1

    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(0, INT32_MAX, N)
    return seeds

def apply_scan_noise_single_img(img, jx=1, jy=0, seed=48):
    if jx is None:
        jx = 0
    if jy is None:
        jy = 0

    t = range(img.shape[0])
    x, y = np.meshgrid(t, t)
    s = img.shape[0]
    rng = np.random.default_rng(seed=seed)
    dx = rng.normal(0, 1, (s, 1)) * jx
    dy = rng.normal(0, 1, (1, s)) * jy
    coords = np.array([y + dy, x + dx])
    kk = map_coordinates(img, coords)
    return kk


def apply_poisson_noise_single_img(img, dose=1 / 10, seed=48):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=img * dose)


def add_gaussian_noise_single_img(img, sigma=0.3, seed=48):
    rng = np.random.default_rng(seed)
    out = img + rng.normal(0, sigma, img.shape)
    return out


class DataSet:

    def __init__(self, ps, lbs):
        self.data = ps
        self.lbs = lbs
        self.n = self.data.shape[0]

    def add_gaussian_noise(self, sigma, seed=None):
        seeds = rng_seeds(seed, self.n)
        ps = np.array([add_gaussian_noise_single_img(img, sigma, seeds[i]) for i, img in enumerate(self.data)])
        return DataSet(ps, self.lbs)


    def apply_poisson_noise(self, dose, seed=None):
        seeds = rng_seeds(seed, self.n)
        ps = np.array([apply_poisson_noise_single_img(img, dose, seeds[i]) for i, img in enumerate(self.data)])
        return DataSet(ps, self.lbs)

    def apply_scan_noise(self, dx, dy, seed=None):
        seeds = rng_seeds(seed, self.n)
        ps = np.array([apply_scan_noise_single_img(img, dx, dy, seeds[i]) for i, img in enumerate(self.data)])
        return DataSet(ps, self.lbs)

    def reconstruct(self, method='zps'):
        pass

