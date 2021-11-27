import numpy as np

# dot method is faster than full grid method
def gauss2d(X, Y, x, y, sigma, A=1.):
    gauss = np.exp(-(X - x) ** 2 / sigma).dot(np.exp(-(Y - y) ** 2 / sigma))  # dot product
    return A*gauss

def many_gauss2d(X, Y, pts, sigma, A=1.):
    gauss = np.zeros_like(X)
    for (x, y) in pts:
        gauss += gauss2d(X, Y, x, y, sigma, A)
    return gauss