import numpy as np
from scipy.special import poch
from scipy.special import factorial


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


def gpzp_calculate_coefficients(n, m, alpha=0):
    """Calculate coefficients from (n, m, alpha)"""
    m_abs = abs(m)
    a = n + m_abs
    b = n - m_abs
    c = factorial(a + 1) / poch(alpha + 1, a + 1)
    coeffs = np.array(
        [c * (-1) ** k * poch(alpha + 1, 2 * n + 1 - k) / (factorial(k) * factorial(a + 1 - k) * factorial(b - k)) for k
         in np.arange(0, b + 1)])
    return coeffs


def gpzp_calculate_norm(n, m, alpha=0):
    """Calculate normalization factor from (n, m, alpha)"""
    m_abs = abs(m)
    a = n + m_abs
    b = n - m_abs
    if m_abs == 0:
        norm = np.sqrt((n + alpha / 2 + 1) * poch(b + alpha + 1, 2 * m_abs + 1) / (poch(b + 1, 2 * m_abs + 1)))
    else:
        norm = np.sqrt((2 * n + alpha + 2) * poch(b + alpha + 1, 2 * m_abs + 1) / (poch(b + 1, 2 * m_abs + 1)))
    return norm


def gpzp_calculate_coefficients_matrix(N, alpha=0, normalize=True):
    """Calculate coefficient matrix"""
    n_max, m = gpzp_j2nm(N - 1)
    shape = (N, n_max + 1)
    c_matrix = np.zeros(shape)
    for j in range(N):
        n, m = gpzp_j2nm(j)
        coeffs = gpzp_calculate_coefficients(n, m, alpha)
        if normalize:
            norm = gpzp_calculate_norm(n, m, alpha)
        else:
            norm = 1.
        c_matrix[j, n_max - n:n_max - n + len(coeffs)] = coeffs * norm
    return c_matrix


def gpzp_calculate_Rn_matrix(N, size, alpha, weighted=True):
    n, m = gpzp_j2nm(N - 1)
    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    r, t = r * mask, t * mask
    r, mask = r.ravel(), mask.ravel()
    if weighted:
        w = (1 - r) ** (alpha / 2) * mask
    else:
        w = 1.
    Rn_matrix = np.array([np.power(r, i) * w for i in np.arange(n + 1)[::-1]])
    return Rn_matrix


def gpzp_calculate_mt_matrix(N, size):
    nm = gpzp_j2nm(range(N))
    n, m = nm[:, 0], nm[:, 1]
    t = grid_rt(size)[1].ravel()
    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]
    return np.array([func(e * t) for e, func in zip(np.abs(m), funcs)])


def generate_gpzps(N, alpha, size):
    c_matrix = gpzp_calculate_coefficients_matrix(N, alpha)
    Rn_matrix = gpzp_calculate_Rn_matrix(N, size, alpha)
    Rnm_alpha = c_matrix.dot(Rn_matrix)
    mt_matrix = gpzp_calculate_mt_matrix(N, size)
    return (Rnm_alpha * mt_matrix).reshape((N, size, size))


def Rnm_alpha(r, n, m, alpha=0):
    coeffs = gpzp_calculate_coefficients(n, m, alpha)
    norm = gpzp_calculate_norm(n, m, alpha)
    Rnm = np.zeros_like(r)
    for k, c in enumerate(coeffs):
        Rnm = Rnm + c * np.power(r, n - k)
    return Rnm * norm * (1 - r) ** (alpha / 2)


def gpzp_nm(n, m, alpha, size):
    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    Rnm = Rnm_alpha(r, n, m, alpha)
    if m >= 0:
        tt = np.cos(m * t)
    else:
        tt = np.sin(-m * t)
    gpzp = Rnm * tt
    return gpzp * mask


def gpzp_j(j, alpha, size):
    n, m = gpzp_j2nm(j)
    return gpzp_nm(n, m, alpha, size)


class GPZPs:

    def __init__(self, N, alpha, size):
        self.num_components = N
        self.alpha = alpha
        self.s = size
        self.data = generate_gpzps(N, alpha, size).reshape((-1, size, size))
        self.ind = np.arange(self.num_components)
        self.data_selected = self.data[self.ind]
        self.j = np.arange(N)
        self.nm = gpzp_j2nm(self.j)

    def set_symm_states(self, states=None):
        if np.isscalar(states):
            nm_pair = np.array([(n, m) for (n, m) in self.nm if abs(m) == states])
        else:
            nm_pair = np.array([(n, m) for state in states for (n, m) in self.nm if abs(m) == state])
        if nm_pair.size is not 0:
            self.ind = gpzp_nm2j(nm_pair[:, 0], nm_pair[:, 1])
            self.data_selected = self.data[self.ind]

    def extract_moments(self, patches):
        zps = self.data_selected
        if len(patches.shape) == 2:
            patches = patches[np.newaxis, :, :]
        s = patches.shape[1]
        num_patches = patches.shape[0]
        num_zps = zps.shape[0]
        m = np.array([(p * zp).sum() for p in patches for zp in zps]).reshape((num_patches, num_zps))
        m = m / (s * s / 4)
        if m.shape[0] == 1:
            m = m[0]
        return m



