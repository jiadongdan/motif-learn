import numpy as np
from sklearn.utils import check_array
from scipy.signal import fftconvolve

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ZPs-like
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def zp_j2nm(j):
    j = np.atleast_1d(j)
    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(int)
    m = 2 * j - n * (n + 2)
    return np.array([n, m]).T


def zp_nm2j(n, m):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)

    n = np.array(n)
    m = np.array(m)
    j = ((n + 2) * n + m) // 2
    return j


def zp_nm2i(n, m):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    i = np.array(n ** 2 + 2 * n + 2 * m)
    mask = np.array(n) % 2 == 0
    i[mask] = i[mask] // 4
    i[~mask] = (i[~mask] - 1) // 4
    if i.size == 1:
        i = i.item()
    return i


def zp_i2nm(i):
    i = np.atleast_1d(i)
    n = np.floor(np.sqrt(4 * i + 1) - 1).astype(int)
    mask = n % 2 == 0
    m = (4 * i - n ** 2 - 2 * n + 1) // 2
    m[mask] = (4 * i[mask] - n[mask] ** 2 - 2 * n[mask] + 1) // 2
    nm = np.array([n, m]).T
    if nm.shape[0] == 1:
        nm = nm[0]
    return nm

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GPZPs-like
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def gpzp_j2nm(j):
    """Convert single index j to pair (n, m)"""
    j = np.atleast_1d(j)

    n = np.floor(np.sqrt(j)).astype(int)
    # Here m can be negative
    m = j - n * (n + 1)
    return np.array([n, m]).T


def gpzp_nm2j(n, m):
    """Convert pair (n, m) to single index j"""
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    return n * (n + 1) + m


def gpzp_nm2i(n, m):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)

    # i = (n+1)*(n+2)//2-(n-m) - 1
    i = (n * n + n + 2 * m) // 2
    return i


def gpzp_i2nm(i):
    i = np.atleast_1d(i)

    n = ((np.sqrt(8 * i + 1) - 1) // 2).astype(int)
    m = (2 * i - n * n - n) // 2
    nm = np.array([n, m]).T
    return nm

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# share
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def nm2j(n, m, alpha=None):
    if alpha is None:
        return zp_nm2j(n, m)
    else:
        return gpzp_nm2j(n, m)


def j2nm(j, alpha=None):
    if alpha is None:
        return zp_j2nm(j)
    else:
        return gpzp_j2nm(j)


def nm2j_complex(n, m, alpha=None):
    if alpha is None:
        return zp_nm2i(n, m)
    else:
        return gpzp_nm2i(n, m)


def j2nm_complex(j, alpha=None):
    if alpha is None:
        return zp_i2nm(j)
    else:
        return gpzp_i2nm(j)

# get r theta grids
def grid_rt(size):
    """Get grid array r and t"""
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    r = np.sqrt(x * x + y * y)
    t = np.arctan2(y, x)
    # exclude r > 1.
    mask = (r <= 1.) * 1
    return r * mask, t

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


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# marray
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class marray(np.ndarray):

    def __new__(cls, input_array, n=None, m=None, alpha=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.N = input_array.shape[1]
        obj.alpha = alpha
        if n is None:
            obj.n = j2nm(range(obj.N), obj.alpha)[:, 0]
        else:
            obj.n = n
        if m is None:
            obj.m = j2nm(range(obj.N), obj.alpha)[:, 1]
        else:
            obj.m = m
        return obj

    def select(self, states=None):
        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        return marray(self[:, ind], n=self.n[ind], m=self.m[ind], alpha=self.alpha)

    def complex(self):
        if self.dtype != complex:
            c_matrix = construct_complex_matrix(self.n, self.m, self.alpha)
            zm_complex = c_matrix.dot(self.T).T
            # update n and m
            c_matrix[c_matrix == 1j] = 0
            m = c_matrix.dot(np.abs(self.m)).real.astype(int)
            n = c_matrix.dot(np.abs(self.n)).real.astype(int)
            return marray(zm_complex, n, m, alpha=self.alpha)
        else:
            return self

    def rotinv(self):
        zm_complex = self.complex()
        mask = (zm_complex.m == 0)
        zm_complex[:, ~mask] = np.abs(zm_complex[:, ~mask])
        zm_rotinv = zm_complex.real
        return marray(zm_rotinv, zm_complex.n, zm_complex.m, alpha=self.alpha)

    def rot(self, theta):
        rot_matrix = construct_rot_matrix(self.n, self.m, theta)
        zm_rot = rot_matrix.dot(self.T).T
        return marray(zm_rot, self.n, self.m, alpha=self.alpha)

    def ref(self, theta):
        ref_matrix = construct_ref_matrix(self.n, self.m, theta)
        zm_rot = ref_matrix.dot(self.T).T
        return marray(zm_rot, self.n, self.m, alpha=self.alpha)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.n = getattr(obj, 'n', None)
        self.m = getattr(obj, 'm', None)
        self.alpha = getattr(obj, 'alpha', None)


def construct_complex_matrix(n, m, alpha=None):
    j1 = nm2j(n, m, alpha)
    j2 = nm2j_complex(n, np.abs(m), alpha)

    # j2 has duplicate elements
    uniq_j2 = np.unique(j2)

    num_rows, num_cols = len(uniq_j2), len(j1)
    vals = np.zeros(len(j1), dtype=complex)
    vals[m >= 0] = 1
    vals[m < 0] = 1j

    d = dict(zip(uniq_j2, range(num_rows)))

    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=complex)
    for i, (ind, v) in enumerate(zip(j2, vals)):
        c_matrix[d[ind], i] = v
    return c_matrix


def construct_rot_matrix(n, m, theta, alpha=None):
    theta = np.deg2rad(theta)
    rot_matrix = np.zeros(shape=(len(n), len(n)))
    for i, (en, em) in enumerate(zip(n, m)):
        if em == 0:
            rot_matrix[i, nm2j(en, em, alpha)] = 1.
        elif em < 0:
            # DON'T forget to multiply m
            t = theta * (-em)
            rot_matrix[i, nm2j(en, em, alpha)] = np.cos(t)
            rot_matrix[i, nm2j(en, -em, alpha)] = -np.sin(t)
        else:
            t = theta * em
            rot_matrix[i, nm2j(en, em, alpha)] = np.cos(t)
            rot_matrix[i, nm2j(en, -em, alpha)] = np.sin(t)
    return rot_matrix


# construct reflection matrix
def construct_ref_matrix(n, m, theta, alpha=None):
    theta = 2 * np.deg2rad(theta)
    ref_matrix = np.zeros(shape=(len(n), len(n)))
    for i, (en, em) in enumerate(zip(n, m)):
        if em == 0:
            ref_matrix[i, nm2j(en, em, alpha=None)] = 1.
        elif em < 0:
            # DON'T forget to multiply m
            t = theta * (-em)
            ref_matrix[i, nm2j(en, em, alpha=None)] = -np.cos(t)
            ref_matrix[i, nm2j(en, -em, alpha=None)] = -np.sin(t)
        else:
            t = theta * em
            ref_matrix[i, nm2j(en, em, alpha=None)] = np.cos(t)
            ref_matrix[i, nm2j(en, -em, alpha=None)] = -np.sin(t)
    return ref_matrix



def _fit(X, data, n, m, alpha, method='matrix'):

    X = check_array(X, allow_nd=True)
    if len(X.shape) == 2:
        if X.shape[0] == data.shape[1]:
            X = X[np.newaxis, :]
            method = method
        elif X.shape[0] > data.shape[1]:
            n_components_ = len(n)
            shape = (n_components_, X.shape[0], X.shape[1])
            X = np.broadcast_to(X, shape)
            method = 'fftconv'

    n_samples_, n_features_ = X.shape[0], np.prod(X.shape[1:])

    if method == 'matrix':
        X = X.reshape(X.shape[0], -1)
        N = data.shape[0]
        data_ = data.reshape(N, -1)
        # self.moments = zmarray(X.dot(np.linalg.pinv(data_)), self.n, self.m)
        # linear regression method
        moments = np.linalg.inv(data_.dot(data_.T)).dot(data_).dot(X.T)
        moments = marray(moments.T, n, m, alpha)

    # this is very slow, here for validation
    elif method == 'direct':
        zps = data
        s = X.shape[1]
        num_zps = len(n)
        moments = np.array([(p * zp).sum() for p in X for zp in zps]).reshape((n_samples_, num_zps))
        moments = moments / (s * s / 4) / np.pi
        moments = marray(moments, n, m, alpha)

    elif method == 'fftconv':
        moments = fftconvolve(X, data, mode='same', axes=[1, 2])
        f = 1 - n % 2
        f[f == 0] = -1
        area = np.pi * (data.shape[1]) ** 2 / 4
        moments = moments * f[:, np.newaxis, np.newaxis] / area
        moments = marray(moments.reshape(len(m), -1).T, n, m, alpha)

    return moments
