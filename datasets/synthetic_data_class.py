import numpy as np
from scipy.ndimage import map_coordinates


import numpy as np
from scipy.ndimage import map_coordinates


def rotate_around(pts, p0=(0, 0), degree=0):
    angle = np.deg2rad(degree)
    # rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(p0)
    p = np.atleast_2d(pts)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def many_gaussians(pts, sigma, s):
    data = np.zeros((s, s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X - x) ** 2 - (Y - y) ** 2) / (2 * sigma * sigma))
    return data

def gn(n, size=128, sigma=None, l=None, theta=0, include_center=True, A=0.9):
    if sigma is None:
        sigma = size / 20
    if l is None:
        l = size * 0.25
    elif np.logical_and(l>0, l<1):
        l = size * l

    p0 = np.array([0, l])
    p0 = rotate_around(p0, degree=theta)
    pts = np.array([rotate_around(p0, degree=360 * i / n) for i in range(n)])
    if include_center:
        gs = many_gaussians(pts, sigma, size) + A * many_gaussians([(0, 0)], sigma, size)
    else:
        gs = many_gaussians(pts, sigma, size)

    return gs

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


class SyntheticData:

    def __init__(self, size=128, sigma=7, l=32, A=1, n_fold=3):

        # params
        self.size = size
        self.sigma = sigma
        self.l = l
        self.A = A
        self.n_fold = n_fold
        self.params = dict(size=size, sigma=sigma, l=l, A=A, n_fold=n_fold)

        self.data = gn(self.n_fold, self.size, self.sigma, self.l, theta=0, include_center=True, A=self.A)


    def add_gaussian_noise(self, n=500, sigma=0.4, seed=None):
        data = np.stack([self.data] * n)
        seeds = rng_seeds(seed, n)
        ps = np.array([add_gaussian_noise_single_img(img, sigma, seeds[i]) for i, img in enumerate(data)])
        return ps


    def apply_poisson_noise(self, n=500, dose=1/10, seed=None):
        data = np.stack([self.data] * n)
        data = (data * 255).astype(int)
        seeds = rng_seeds(seed, n)
        ps = np.array([apply_poisson_noise_single_img(img, dose, seeds[i]) for i, img in enumerate(data)])
        return ps

    def apply_scan_noise(self, n=500, dx=3, dy=0, seed=None):
        data = np.stack([self.data] * n)
        seeds = rng_seeds(seed, n)
        ps = np.array([apply_scan_noise_single_img(img, dx, dy, seeds[i]) for i, img in enumerate(data)])
        return ps


class BinarySyntheticData:

    def __init__(self, size=128, sigma=7, l=32, A=0.8, n_fold=3):
        # params
        self.size = size
        self.sigma = sigma
        self.l = l
        self.A = A
        self.n_fold = n_fold
        self.params = dict(size=size, sigma=sigma, l=l, A=A, n_fold=n_fold)

        self.data1 = gn(self.n_fold, self.size, self.sigma, self.l, theta=0, include_center=True, A=1.0)
        self.data2 = gn(self.n_fold, self.size, self.sigma, self.l, theta=0, include_center=True, A=self.A)

    def add_gaussian_noise(self, n1=500, n2=500, sigma=0.4, seed=None):
        data = np.stack([self.data1] * n1 + [self.data2] * n2)
        seeds = rng_seeds(seed, n1+n2)
        ps = np.array([add_gaussian_noise_single_img(img, sigma, seeds[i]) for i, img in enumerate(data)])
        lbs = np.array([0] * n1 + [1] * n2)
        return ps, lbs


    def apply_poisson_noise(self, n1=500, n2=500, dose=1/10, seed=None):
        data = np.stack([self.data1] * n1 + [self.data2] * n2)
        data = (data * 255).astype(int)
        seeds = rng_seeds(seed, n1 + n2)
        ps = np.array([apply_poisson_noise_single_img(img, dose, seeds[i]) for i, img in enumerate(data)])
        lbs = np.array([0] * n1 + [1] * n2)
        return ps, lbs

    def apply_scan_noise(self, n1=500, n2=500, dx=3, dy=0, seed=None):
        data = np.stack([self.data1] * n1 + [self.data2] * n2)
        seeds = rng_seeds(seed, n1 + n2)
        ps = np.array([apply_scan_noise_single_img(img, dx, dy, seeds[i]) for i, img in enumerate(data)])
        lbs = np.array([0] * n1 + [1] * n2)
        return ps, lbs

