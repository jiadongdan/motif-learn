import numpy as np
from numba import jit
from scipy.ndimage import center_of_mass


def disk_patch(radius, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


def center_of_mass_refine(data, pts, size=3, mode=None):
    labels = np.zeros_like(data)
    index = []
    for ii, (x, y) in enumerate(pts):
        if mode == 'disk':
            labels[y - size:y + size + 1, x - size:x + size + 1] = (ii + 1) * disk_patch(size)
        else:
            labels[y - size:y + size + 1, x - size:x + size + 1] = ii + 1
        index.append(ii + 1)
    pts_ = center_of_mass(data, labels, index)

    return np.fliplr(np.array(pts_))


def clear_border(pts, shape, size):
    x, y = pts[:, 0], pts[:, 1]
    mask1 = np.logical_and(x > size // 2 + 1, x < shape[1] - size // 2 - 1)
    mask2 = np.logical_and(y > size // 2 + 1, y < shape[1] - size // 2 - 1)
    return pts[mask1 * mask2]


class KeyPoints:

    def __init__(self, pts, img, size):
        self.shape = img.shape
        self.size = size
        self.img = img
        self.pts = clear_border(pts, self.shape, self.size)
        self.patches = None

    def extract_patches(self, size=None, flat=False):
        if size is None:
            size = self.size
        else:
            size = size
        if size % 2 == 0:
            s1 = size // 2
            s2 = size // 2
        else:
            s1 = size // 2
            s2 = size // 2 + 1

        pts = self.pts.astype(int)
        if flat:
            self.patches = np.array([self.img[y - s1:y + s2, x - s1:x + s2].flatten() for (x, y) in pts])
        else:
            self.patches = np.array([self.img[y - s1:y + s2, x - s1:x + s2] for (x, y) in pts])
        return self.patches

    def clear_border(self, size):
        x, y = self.pts[:, 0], self.pts[:, 1]
        mask1 = np.logical_and(x > size // 2 + 1, x < self.shape[1] - size // 2 - 1)
        mask2 = np.logical_and(y > size // 2 + 1, y < self.shape[1] - size // 2 - 1)
        self.pts = self.pts[mask1 * mask2]

    def refine(self, data=None, r=3, mode=None):
        pts = self.pts.astype(int)
        if data is None:
            data = self.img
        self.pts = center_of_mass_refine(data, pts, size=r, mode=mode)


# https://stackoverflow.com/q/49998879/5855131
# dot method is faster than full grid method
def gauss2d(X, Y, x, y, sigma, A=1.):
    gauss = np.exp(-(Y - y) ** 2 / sigma).dot(np.exp(-(X - x) ** 2 / sigma))  # dot product
    return A * gauss


def many_gauss2d(size, pts, sigma=10, A=1.):
    t = np.arange(size).astype(float)
    X, Y = np.meshgrid(t, t, sparse=True)
    gauss = np.zeros(shape=(size, size))
    for (x, y) in pts:
        gauss += gauss2d(X, Y, x, y, sigma, A)
    return gauss


# this is 3x faster
@jit(nopython=True, parallel=True)
def many_gauss2d_numba(pts, size=1024, sigma=10., A=1.):
    X = np.arange(0, size).reshape(1, size)
    Y = np.arange(0, size).reshape(size, 1)
    c = np.zeros((size, size))
    for (x, y) in pts:
        a = np.exp(-(Y - y) ** 2 / sigma)
        b = np.exp(-(X - x) ** 2 / sigma)
        c += a.dot(b)
    return A*c