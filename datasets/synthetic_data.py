import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise


def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


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
    else:
        l = size * l
    p0 = np.array([0, l])
    p0 = rotate_pts(p0, theta)
    pts = np.array([rotate_pts(p0, 360 * i / n) for i in range(n)])
    if include_center:
        gs = many_gaussians(pts, sigma, size) + A * many_gaussians([(0, 0)], sigma, size)
    else:
        gs = many_gaussians(pts, sigma, size)

    return gs


def make_rot_data(N, n=3, size=128, sigma=None, l=None, theta=0, include_center=True, A=0.9):
    img = gn(n, size, sigma, l, theta, include_center, A)

    angles = np.linspace(0, 360, N)
    ps = np.array([rotate(img, angle) for angle in angles])
    return ps

def add_noise(img, sigma=0.5, num_samples=1000, seed=48, clip=False, include=False):
    if len(img.shape) == 2:
        imgs = np.tile(img, (num_samples, 1, 1))
    elif len(img.shape) == 3:
        imgs = img
    imgs_noise = random_noise(imgs, clip=clip, var=sigma**2, seed=seed)
    if include:
        imgs_noise = np.vstack([img[np.newaxis, :, :], imgs_noise])
    return imgs_noise


def make_two_classes(N, sigma=None, n=3, size=128, A=0.8, seed=48, clip=False):
    img1 = gn(n, size, A=1.0)
    img2 = gn(n, size, A=A)
    imgs1 = np.tile(img1, (N, 1, 1))
    imgs2 = np.tile(img2, (N, 1, 1))
    imgs = np.vstack([imgs1, imgs2])
    if sigma is not None:
        imgs = add_noise(imgs, sigma, seed=seed, clip=clip)
    lbs = np.array([0]*len(imgs1) + [1]*len(imgs2))
    return imgs, lbs


def apply_poisson_noise(img, dose=1/100, N=100, seed=0):
    if len(img.shape) == 2:
        imgs = np.tile(img, (N, 1, 1))
    elif len(img.shape) == 3:
        imgs = img

    if np.any(imgs < 0):
        raise ValueError('data must not contain any negative values')

    rng = np.random.default_rng(seed)

    return rng.poisson(lam=imgs * dose)

def make_two_classes_poisson(N, dose=None, n=3, size=128, A=0.8, seed=48, include=False):
    img1 = gn(n, size, A=1.0)
    img2 = gn(n, size, A=A)
    img1 = (img1*255).astype(int)+10
    img2 = (img2*255).astype(int)+10
    imgs1 = np.tile(img1, (N, 1, 1))
    imgs2 = np.tile(img2, (N, 1, 1))
    imgs = np.vstack([imgs1, imgs2])
    if dose is not None:
        imgs = apply_poisson_noise(imgs, dose, seed=seed)
    lbs = np.array([0]*len(imgs1) + [1]*len(imgs2))
    if include:
        imgs = np.vstack([img1[np.newaxis,:]*dose, imgs, img2[np.newaxis,:]*dose])
        lbs = np.array([0]*(len(imgs1)+1) + [1]*(len(imgs2)+1))
    return imgs, lbs


class GN:

    def __init__(self, n=3, size=128, sigma=None, l=None, theta=0, include_center=True, A=0.9):
        self.n = n
        self.size = size
        self.sigma = sigma
        self.l = l
        self.theta = theta
        self.include_center = include_center
        self.A = A

        if not np.iterable(self.theta):
            self.data = gn(self.n, self.size, self.sigma, self.l, self.theta, self.include_center, self.A)
        else:
            self.data = np.array([gn(self.n, self.size, self.sigma, self.l, t, self.include_center, self.A) for t in self.theta])


    def duplicate(self, n=100):
        pass

    def apply_white_noise(self, seed=48):
        pass

    def apply_shot_noise(self, dose, constant=10, seed=48):
        pass



def rng_seeds(seed, N):
    INT32_MAX = np.iinfo(np.int32).max - 1

    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(0, INT32_MAX, N)
    return seeds

