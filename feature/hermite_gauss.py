import numpy as np
from scipy.special import hermite, factorial


def hermite_gauss(n, x):
    A = np.sqrt(1/(2**n*factorial(n)*np.sqrt(np.pi)))
    return A*hermite(n)(x)*np.exp(-x**2/2)

def get_HG_from_nm(n, m, size):
    N = np.atleast_1d(n)
    M = np.atleast_1d(m)
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    HGs = np.array([hermite_gauss(n, x)*hermite_gauss(m, y) for n, m in zip(N, M)])
    return HGs

class HG:

    def __init__(self, n_max, size):

        pass

    def fit(self, X):
        pass


