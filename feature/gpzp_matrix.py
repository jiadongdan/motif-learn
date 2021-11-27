import numpy as np
from scipy.special import poch
from scipy.special import factorial

from .zmarray import zmarray
from .zmarray import construct_complex_matrix

def gpzp_j2nm(j):
    """Convert single index j to pair (n, m)"""
    n = np.floor(np.sqrt(j)).astype(np.int)
    # Here m can be negative
    m = j - n * (n + 1)
    return np.array([n, m]).T


def gpzp_nm2j(n, m):
    """Convert pair (n, m) to single index j"""
    return n * (n + 1) + m

def grid_rt(size):
    """Get grid array r and t"""
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    r = np.sqrt(x * x + y * y)
    t = np.arctan2(y, x)
    return r, t

def gpzp_get_coeffs(n, m, alpha=0, normalize=True):
    n = np.array(n)
    m = np.array(m)
    n_max = n.max()
    N = n + np.zeros_like(m)
    M = m + np.zeros_like(n)
    if not np.iterable(N):
        N = np.array([N])
        M = np.array([M])
    M_abs = np.abs(M)

    A = N + M_abs
    B = N - M_abs
    C = factorial(A + 1) / poch(alpha + 1, A + 1)

    l = []
    for i in range(len(N)):
        a, b, c, n = A[i], B[i], C[i], N[i]
        coeffs = [
            c * (-1) ** k * poch(alpha + 1, 2 * n + 1 - k) / (factorial(k) * factorial(a + 1 - k) * factorial(b - k))
            for k in np.arange(0, b + 1)]
        coeffs = [0] * (n_max - n) + coeffs + [0] * (n - b)
        l.append(coeffs)
    c_matrix = np.array(l)
    if normalize:
        c_matrix = c_matrix * (gpzp_norm(N, M, alpha=alpha)[:, np.newaxis])
    return c_matrix


def gpzp_norm(n, m, alpha=0):
    n = np.array(n)
    m = np.array(m)
    n = n + np.zeros_like(m)
    m = m + np.zeros_like(n)
    if not np.iterable(n):
        n = np.array([n])
        m = np.array([m])
    m_abs = np.abs(m)
    a = n + m_abs
    b = n - m_abs
    mask = (m == 0)
    norm = np.zeros(len(n))
    norm[mask] = (np.sqrt((n + alpha / 2 + 1) * poch(b + alpha + 1, 2 * m_abs + 1) / (poch(b + 1, 2 * m_abs + 1))))[
        mask]
    norm[~mask] = (np.sqrt((2 * n + alpha + 2) * poch(b + alpha + 1, 2 * m_abs + 1) / (poch(b + 1, 2 * m_abs + 1))))[
        ~mask]
    return norm


def gpzp_get_Rn_matrix(n, size, alpha=0, weighted=True):
    if not np.iterable(n):
        n = np.array([n])
    else:
        n = np.array(n)
    n_max = n.max()

    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    r, t = r * mask, t * mask
    r, mask = r.ravel(), mask.ravel()
    if weighted:
        w = (1 - r) ** (alpha / 2) * mask
    else:
        w = 1.
    # shape of Rn_matrix is (n_max+1, size*size)
    Rn_matrix = np.array([np.power(r, i) * w for i in np.arange(n_max + 1)[::-1]])
    return Rn_matrix


def gpzp_get_mt_matrix(m, size):
    if not np.iterable(m):
        m = np.array([m])
    else:
        m = np.array(m)
    t = grid_rt(size)[1].ravel()
    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]
    return np.array([func(e * t) for e, func in zip(np.abs(m), funcs)])


def gpzps_from_nm(n, m, size, alpha=0, reshape=True):
    c_matrix = gpzp_get_coeffs(n, m, alpha)
    Rn_matrix = gpzp_get_Rn_matrix(n, size, alpha)
    Rnm_ = c_matrix.dot(Rn_matrix)
    mt_matrix = gpzp_get_mt_matrix(m, size)
    if reshape:
        return (Rnm_ * mt_matrix).reshape((c_matrix.shape[0], size, size))
    else:
        return Rnm_ * mt_matrix



def _get_m_states(m, states=None):
    if not np.iterable(states):
        if states is None:
            states = list(set(np.abs(m)))
        elif states < 0:  # states is negative number
            uniq = np.unique(np.abs(m))
            states = uniq[np.nonzero(uniq)]
        else:
            states = [states]
    else:
        states = np.unique(np.abs(states))
    return states


class GPZPs:
    def __init__(self, n_max, size, alpha=0, states=None):
        self.alpha = alpha
        self.j_max = gpzp_nm2j(n_max, n_max)
        nm = gpzp_j2nm(np.arange(self.j_max+1))
        self.n, self.m = nm[:, 0], nm[:, 1]

        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        self.n = self.n[ind]
        self.m = self.m[ind]

        self.data = gpzps_from_nm(self.n, self.m, size, self.alpha)
        self.moments = None


    def fit(self, X):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        N = self.data.shape[0]
        data_ = self.data.reshape(N, -1)
        self.moments = zmarray(X.dot(np.linalg.pinv(data_)), self.n, self.m, mode='pzm')


    def complex(self):
        if self.dtype != np.complex:
            c_matrix = construct_complex_matrix(self.n, self.m, mode='pzm')
            zm_complex = c_matrix.dot(self.T).T
            # update n and m
            c_matrix[c_matrix == 1j] = 0
            m = c_matrix.dot(np.abs(self.m)).real.astype(np.int)
            n = c_matrix.dot(np.abs(self.n)).real.astype(np.int)
            return zmarray(zm_complex, n, m, mode='pzm')
        else:
            return self

    def abs(self):
        if self.dtype != np.complex:
            return self
        else:
            return zmarray(np.abs(self), self.n, self.m, mode='pzm')
