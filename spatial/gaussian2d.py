import numpy as np
from numba import njit
# https://stackoverflow.com/q/49998879/5855131
# dot method is faster than full grid method
def gauss2d(X, Y, x, y, sigma, A=1.):
    gauss = np.exp(-(Y - y) ** 2 / sigma).dot(np.exp(-(X - x) ** 2 / sigma))  # dot product
    return A * gauss


def many_gauss2d(pts, size, sigma=10, A=1.):
    t = np.arange(size).astype(float)
    X, Y = np.meshgrid(t, t, sparse=True)
    gauss = np.zeros(shape=(size, size))
    for (x, y) in pts:
        gauss += gauss2d(X, Y, x, y, sigma, A)
    return gauss


# this is 3x faster for (1024, 1024)
@njit(parallel=True)
def many_gauss2d_numba(pts, size=1024, sigma=10., A=1.):
    X = np.arange(0, size).reshape(1, size)
    Y = np.arange(0, size).reshape(size, 1)
    c = np.zeros((size, size))
    for (x, y) in pts:
        a = np.exp(-(Y - y) ** 2 / sigma)
        b = np.exp(-(X - x) ** 2 / sigma)
        c += a.dot(b)
    return A*c


def filter_xy(pts, n=19):
    d = np.hypot(pts[:, 0], pts[:, 1])
    ind = np.argsort(d)[0:19]
    return pts[ind]

def get_template(a=15, b=16, theta=0, size=64, sigma=10, A=1.):
    if sigma is None:
        sigma = a / 3 * 2

    size_ = size + 2 * int(max(a, b))

    cost3 = np.cos(np.deg2rad(theta))
    sint3 = np.sin(np.deg2rad(theta))
    cost4 = np.cos(np.deg2rad(theta + 60))
    sint4 = np.sin(np.deg2rad(theta + 60))

    # get uv
    u = np.array([a * cost3, a * sint3])
    v = np.array([b * cost4, b * sint4])
    uv = np.vstack([u, v])

    m = int(size_ * 4 / np.sqrt(6) / a) + 1
    x, y = np.indices((m, m)) - m // 2
    xy = np.array([x.ravel(), y.ravel()]).T
    pts = xy.dot(uv)

    pts = filter_xy(pts)

    pts += (size // 2, size // 2)

    dd = many_gauss2d(pts, size, sigma, A)
    return dd
