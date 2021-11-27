import numpy as np
from scipy.special import genlaguerre, factorial


from .share_utils import grid_rt

def get_LG_from_nm(n, m, size):
    N = np.atleast_1d(n)
    M = np.atleast_1d(m)
    M_abs = np.abs(M)

    r, t = grid_rt(size)
    r2 = 2*np.pi*r**2
    expr2 = np.exp(-r2/2)

    f = np.array([np.sin, np.cos])
    mask = (M >= 0) * 1
    funcs = f[mask]

    disk_array = (r > 0) * 1.
    disk_array[size // 2, size // 2] = 1.

    LGs = []
    for i, (n, m) in enumerate(zip(N, M_abs)):
        coeff = (-1)**n*np.sqrt(2*(2*np.pi)**m*factorial(n)/factorial(n+m))
        Rnm = coeff*genlaguerre(n, m)(r2)*(r**m)*expr2

        mt = funcs[i](m * t) * disk_array
        LGs.append(Rnm * mt)

    return np.array(LGs)
