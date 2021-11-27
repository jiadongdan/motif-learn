import numpy as np
from scipy.special import binom
from sklearn.utils import check_array
from sklearn.utils import deprecated
from sklearn.preprocessing import normalize


def zp_j2nm(j):
    if not np.isscalar(j):
        j = check_array(j, ensure_2d=False)
    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(np.int)
    m = 2 * j - n * (n + 2)
    return np.array([n, m]).T


def zp_nm2j(n, m):
    n = np.array(n)
    m = np.array(m)
    j = ((n + 2) * n + m) // 2
    return j


def zp_nm2j_complex(n, m):
    j = np.array(n ** 2 + 2 * n + 2 * m)
    mask = np.array(n) % 2 == 0
    j[mask] = j[mask] // 4
    j[~mask] = (j[~mask] - 1) // 4
    if j.size == 1:
        j = j.item()
    return j


def zp_j2nm_complex(j):
    if not np.iterable(j):
        j = np.array([j])
    else:
        j = np.array(j)
    n = np.floor(np.sqrt(4 * j + 1) - 1).astype(np.int)
    mask = n % 2 == 0
    m = (4 * j - n ** 2 - 2 * n + 1) // 2
    m[mask] = (4 * j[mask] - n[mask] ** 2 - 2 * n[mask] + 1) // 2
    nm = np.array([n, m]).T
    if nm.shape[0] == 1:
        nm = nm[0]
    return nm


def get_j1_j2(N):
    n, m = zp_j2nm(N - 1)
    if m < n:
        n = n - 1
    j1 = zp_nm2j(n, n)
    j2 = zp_nm2j_complex(n, n)
    return j1, j2

def get_n1_n2(N):
    n, m = zp_j2nm(N - 1)
    if m < n:
        n = n - 1
    j1 = zp_nm2j(n, n)
    j2 = zp_nm2j_complex(n, n)
    return j1+1, j2+1


def grid_rt(size):
    """Get grid array r and t"""
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    r = np.sqrt(x * x + y * y)
    t = np.arctan2(y, x)
    return r, t


def zp_calculate_coeffs(n, m):
    n = np.array(n)
    m = np.array(m)
    n_max = n.max()
    N = n + np.zeros_like(m)
    M = m + np.zeros_like(n)
    if not np.iterable(N):
        N = np.array([N])
        M = np.array([M])
    M_abs = np.abs(M)
    assert(((N - M_abs)%2 == 0).all())
    l = []
    for (n, m) in zip(N, M_abs):
        num = (n - m) // 2
        c = np.r_[[0]*(n_max-n), [np.power(-1, k) * binom(n - k, k) * binom(n - 2*k, num - k) if i%2==0 else 0 for i, k in enumerate(np.arange(0, n+1)//2)]]
        l.append(c)
    return np.array(l)


def zp_calculate_coefficients(n, m):
    """Calculate coefficients from (n, m)"""
    m = np.abs(m)
    assert ((n - m) % 2 == 0)
    num = (n - m) // 2
    coeffs = np.array([np.power(-1, k) * binom(n - k, k) * binom(n - 2 * k, num - k) for k in np.arange(0, num + 1)])
    return coeffs


def zp_calculate_norm(n, m):
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * n + 2)
    return norm


def zp_calculate_coefficients_matrix(N, normalize=True):
    """Calculate coefficient matrix"""
    n_max, m = zp_j2nm(N - 1)
    shape = (N, n_max + 1)
    c_matrix = np.zeros(shape)
    for j in range(N):
        n, m = zp_j2nm(j)
        coeffs = zp_calculate_coefficients(n, m)
        if normalize:
            norm = zp_calculate_norm(n, m)
        else:
            norm = 1.
        # different from gpzp case
        c_matrix[j, n_max - n:n_max - n + 2 * len(coeffs):2] = coeffs * norm
    return c_matrix


def zp_calculate_Rn_matrix(N, size):
    n, m = zp_j2nm(N - 1)
    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    r, mask = r.ravel(), mask.ravel()
    Rn_matrix = np.array([np.power(r, i) for i in np.arange(n + 1)[::-1]])
    return Rn_matrix * mask


def zp_calculate_mt_matrix(N, size):
    nm = zp_j2nm(range(N))
    n, m = nm[:, 0], nm[:, 1]
    t = grid_rt(size)[1].ravel()
    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]
    return np.array([func(e * t) for e, func in zip(np.abs(m), funcs)])


def generate_zps(N, size, reshape=True):
    c_matrix = zp_calculate_coefficients_matrix(N)
    Rn_matrix = zp_calculate_Rn_matrix(N, size)
    Rnm_ = c_matrix.dot(Rn_matrix)
    mt_matrix = zp_calculate_mt_matrix(N, size)
    if reshape:
        return (Rnm_ * mt_matrix).reshape((N, size, size))
    else:
        return (Rnm_ * mt_matrix)


def get_complex_matrix(num_rows, num_cols):
    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=np.complex)
    for i in np.arange(num_rows):
        n, m = zp_j2nm_complex(i)
        j1, j2 = zp_nm2j(n, -m), zp_nm2j(n, m)
        if j1 == j2:
            c_matrix[i, j1] = 1
        else:
            c_matrix[i, j1] = 1
            c_matrix[i, j2] = 1j
    return c_matrix


@deprecated('zmarray_ has been deprecated, use zmarray instead.')
class zmarray_(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # num_components
        obj.n_components = obj.shape[1]
        # num_components used for rotation-invariant and complex representations
        obj.n1, obj.n2 = get_n1_n2(obj.n_components)
        obj.j1, obj.j2 = get_j1_j2(obj.n_components)
        obj.complex_matrix = get_complex_matrix(obj.n2, obj.n1)
        # obj.moments_rot = obj[:, 0: obj.j1+1]
        obj.m_array = zp_j2nm(np.arange(obj.n_components))[:, 1]
        obj.m_array_ = zp_j2nm_complex(np.arange(obj.n2))[:, 1]
        obj.ind = np.arange(obj.n1)
        # Finally, we must return the newly created object:
        return obj

    def to_complex(self, states=None):
        moments_rot = self[:, 0: self.n1]
        moments_complex = self.complex_matrix.dot(moments_rot.T).T
        ind = np.arange(self.n2)
        if states is not None:
            if not np.iterable(states):
                states = [states]
            ind = np.hstack([ind[self.m_array_ == s] for s in states])
            ind.sort()
        return moments_complex[:, ind]

    def to_symm_states(self, states):
        if not np.iterable(states):
            states = [states]
        ind = np.arange(self.N)
        ind = np.hstack([ind[self.m_array == s] for s in states])
        ind.sort()
        return self[:, ind]

    def select_symm(self, states):
        if not np.iterable(states):
            states = [states]
        self.ind = np.arange(self.n1)
        self.ind = np.hstack([self.ind[np.abs(self.m_array) == s] for s in states])
        self.ind.sort()
        # calculate complex matrix here
        self.complex_matrix = get_complex_matrix(self.n2, self.n1)
        self.complex_matrix = self.complex_matrix[:, self.ind]
        return self[:, self.ind]

@deprecated('deprecated')
class ZPs_:

    def __init__(self, size, N=None, n_max=None):
        if n_max is None:
            self.N = N
        else:
            self.N = (n_max*(n_max+2)+n_max)//2+1
        self.N_complex = get_n1_n2(self.N)[1]
        self.size = size
        self.data = generate_zps(self.N, self.size, reshape=True)
        self.moments = None

    def fit(self, X):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        data_ = self.data.reshape(self.N, -1)
        self.moments = zmarray(X.dot(np.linalg.pinv(data_)))


    def get_symm_index(self, m, mode=None):
        if not np.iterable(m):
            m = [m]
        if mode is None:
            M = np.abs(zp_j2nm(range(self.N))[:, 1])
        elif mode == 'complex':
            M = np.abs(zp_j2nm_complex(range(self.N_complex))[:, 1])
        ind = np.where(np.in1d(M, m))[0]
        return ind


class zmarray(np.ndarray):
    def __new__(cls, input_array, n=None, m=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.N = input_array.shape[1]
        if n is None:
            obj.n = zp_j2nm(range(obj.N))[:, 0]
        else:
            obj.n = n
        if m is None:
            obj.m = zp_j2nm(range(obj.N))[:, 1]
        else:
            obj.m = m
        return obj

    def select(self, states=None):
        if not np.iterable(states):
            if states is None:
                states = list(set(np.abs(self.m)))
            elif states < 0: # states is negative number
                uniq = np.unique(np.abs(self.m))
                states = uniq[np.nonzero(uniq)]
            else:
                states = [states]
        else:
            states = np.unique(np.abs(states))
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        return zmarray(self[:, ind], n=self.n[ind], m=self.m[ind])

    def complex(self):
        if self.dtype != np.complex:
            c_matrix = construct_complex_matrix(self.n, self.m)
            zm_complex = c_matrix.dot(self.T).T
            # update n and m
            c_matrix[c_matrix == 1j] = 0
            m = c_matrix.dot(np.abs(self.m)).real.astype(np.int)
            n = c_matrix.dot(np.abs(self.n)).real.astype(np.int)
            return zmarray(zm_complex, n, m)
        else:
            return self

    def abs(self):
        if self.dtype != np.complex:
            return self
        else:
            return zmarray(np.abs(self), self.n, self.m)

    def normalize(self, norm='l2'):
        if self.dtype != np.complex:
            return zmarray(normalize(self, axis=1, norm=norm), self.n, self.m)
        else:
            return self



def construct_complex_matrix(n, m):
    j1 = zp_nm2j(n, m)
    j2 = zp_nm2j_complex(n, np.abs(m))
    uniq_j2 = np.unique(j2)
    num_rows, num_cols = len(uniq_j2), len(j1)
    vals = np.zeros(len(j1), dtype=np.complex)
    vals[m > 0] = 1j
    vals[m <= 0] = 1

    d = dict(zip(uniq_j2, range(num_rows)))

    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=np.complex)
    for i, (ind, v) in enumerate(zip(j2, vals)):
        c_matrix[d[ind], i] = v
    return c_matrix