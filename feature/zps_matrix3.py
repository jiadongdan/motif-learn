import numpy as np
from scipy.special import binom
from sklearn.preprocessing import normalize
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from scipy.signal import fftconvolve
from scipy.linalg import norm


def zp_j2nm(j):
    if not np.iterable:
        j = np.array([j])
    else:
        j = np.array(j)
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


def grid_rt(size):
    """Get grid array r and t"""
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    r = np.sqrt(x * x + y * y)
    t = np.arctan2(y, x)
    return r, t


def zp_norm(n, m):
    n = np.array(n)
    m = np.array(m)
    n = n + np.zeros_like(m)
    m = m + np.zeros_like(n)
    if not np.iterable(n):
        n = np.array([n])
        m = np.array([m])
    mask = (m == 0)
    # don't use np.zeros_like(n) as n.dtype is np.int
    norm = np.zeros(len(n))
    norm[mask] = np.sqrt(n[mask] + 1)
    norm[~mask] = np.sqrt(2 * n[~mask] + 2)
    return norm


def zp_get_Rn_matrix(n, size):
    if not np.iterable(n):
        n = np.array([n])
    else:
        n = np.array(n)
    n_max = n.max()
    r, t = grid_rt(size)
    disk_array = (r <= 1) * 1.
    r[r > 1.] = 0.
    r = r.ravel()
    # shape of Rn_matrix is (n_max+1, size*size)
    # treat np.power(r, 0) separately
    # Rn_matrix = np.array([np.power(r, i) for i in np.arange(n_max + 1)[::-1]])
    Rn_matrix = np.array([np.power(r, i) for i in np.arange(1, n_max + 1)[::-1]] + [disk_array.ravel()])
    return Rn_matrix


def zp_get_coeffs(n, m, normalize=True):
    n = np.array(n)
    m = np.array(m)
    n_max = n.max()
    N = n + np.zeros_like(m)
    M = m + np.zeros_like(n)
    if not np.iterable(N):
        N = np.array([N])
        M = np.array([M])
    M_abs = np.abs(M)
    assert (((N - M_abs) % 2 == 0).all())
    l = []
    for (n, m) in zip(N, M_abs):
        num = (n - m) // 2
        c = np.r_[
            [0] * (n_max - n), [np.power(-1, k) * binom(n - k, k) * binom(n - 2 * k, num - k) if i % 2 == 0 else 0 for
                                i, k in enumerate(np.arange(0, n + 1) // 2)]]
        l.append(c)
    c_matrix = np.array(l)
    if normalize:
        c_matrix = c_matrix * (zp_norm(N, M)[:, np.newaxis])
    return c_matrix


def zp_get_mt_matrix(m, size):
    if not np.iterable(m):
        m = np.array([m])
    else:
        m = np.array(m)
    t = grid_rt(size)[1].ravel()
    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]
    return np.array([func(e * t) for e, func in zip(np.abs(m), funcs)])


def generate_zps_from_nm(n, m, size, reshape=True):
    c_matrix = zp_get_coeffs(n, m)
    Rn_matrix = zp_get_Rn_matrix(n, size)
    Rnm_ = c_matrix.dot(Rn_matrix)
    mt_matrix = zp_get_mt_matrix(m, size)
    if reshape:
        return (Rnm_ * mt_matrix).reshape((c_matrix.shape[0], size, size))
    else:
        return (Rnm_ * mt_matrix)


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


class ZPs(TransformerMixin, BaseEstimator):
    """
    The Zernike polynomials class

    Parameters
    ----------
    n_max: int
        the maximum radial index `n`
    size: int
        the size of arrays for generated polynomials
    states: int or array-like, default: None
        the symmetry states to select polynomials

    Attributes
    ----------
    n_components_: int
        the number of Zernike polynomials used to approximate `X`.
    n: array, shape (n_components\_,)
        the radial indices of Zernike polynomials.
    m: array, shape (n_components\_,)
        the azimuthal indices of Zernike polynomials.
    n_features_ : int
        number of features in the training data.
    n_samples_ : int
        number of samples in the training data.

    Notes
    -----
    For example, :code:`states = 3` will only select polynomial terms with :math:`|m|=3`,
    and :code:`states = [3, 6]` will select polynomial terms with :math:`|m|=3, 6`.

    Examples
    --------
    >>> from stempy.feature import ZPs
    >>> # intialize zps
    >>> zps = ZPs(n_max=10, size=256, states=None)
    ZPs(n_max=10, size=256)
    >>> zps.n
    array([ 0,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,
        5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
        7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    >>> zps.m
    array([  0,  -1,   1,  -2,   0,   2,  -3,  -1,   1,   3,  -4,  -2,   0,
         2,   4,  -5,  -3,  -1,   1,   3,   5,  -6,  -4,  -2,   0,   2,
         4,   6,  -7,  -5,  -3,  -1,   1,   3,   5,   7,  -8,  -6,  -4,
        -2,   0,   2,   4,   6,   8,  -9,  -7,  -5,  -3,  -1,   1,   3,
         5,   7,   9, -10,  -8,  -6,  -4,  -2,   0,   2,   4,   6,   8,
        10])
    """

    def __init__(self, n_max, size, states=None):
        self.n_max = n_max
        self.size = size
        self.states = states

        self.j_max = (n_max * (n_max + 2) + n_max) // 2
        nm = zp_j2nm(range(self.j_max + 1))
        self.n, self.m = nm[:, 0], nm[:, 1]

        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        self.n = self.n[ind]
        self.m = self.m[ind]

        self.n_components_ = len(self.n)
        self.data = generate_zps_from_nm(self.n, self.m, size)
        self.moments = None

    def fit(self, X, y=None, method='matrix'):
        """
        Fit the model using X as training data

        Parameters
        ----------
        X: array_like
            Training image data. If array or matrix, shape [n_samples, image_height, image_width]
        y : None
            Ignored variable.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, allow_nd=True)
        if len(X.shape) == 2:
            if X.shape[0] == self.data.shape[1]:
                X = X[np.newaxis, :]
                method = method
            elif X.shape[0] > self.data.shape[1]:
                shape = (self.n_components_, X.shape[0], X.shape[1])
                X = np.broadcast_to(X, shape)
                method = 'fftconv'

        self.n_samples_, self.n_features_ = X.shape[0], np.prod(X.shape[1:])

        if method == 'matrix':
            X = X.reshape(X.shape[0], -1)
            N = self.data.shape[0]
            data_ = self.data.reshape(N, -1)
            # self.moments = zmarray(X.dot(np.linalg.pinv(data_)), self.n, self.m)
            # linear regression method
            moments = np.linalg.inv(data_.dot(data_.T)).dot(data_).dot(X.T)
            self.moments = zmarray(moments.T, self.n, self.m)

        # this is very slow, here for validation
        elif method == 'direct':
            zps = self.data
            s = X.shape[1]
            num_zps = len(self.n)
            moments = np.array([(p * zp).sum() for p in X for zp in zps]).reshape((self.n_samples_, num_zps))
            moments = moments / (s * s / 4) / np.pi
            self.moments = zmarray(moments, self.n, self.m)

        elif method == 'fftconv':
            self.moments = fftconvolve(X, self.data, mode='same', axes=[1, 2])
            f = 1 - self.n % 2
            f[f == 0] = -1
            area = np.pi * (self.data.shape[1]) ** 2 / 4
            self.moments = self.moments * f[:, np.newaxis, np.newaxis] / area
            self.moments = zmarray(self.moments.reshape(len(self.m), -1).T, self.n, self.m)

        return self

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the Zernike dimensionality reduction on X

        Parameters
        ----------
        X: array_like
            Training image data. If array or matrix, shape [n_samples, image_height, image_width]
        y : None
            Ignored variable.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        X_new = self.moments
        if X_new.shape[0] == 1:
            X_new = X_new[0]
        return X_new


class zmarray(np.ndarray):
    def __new__(cls, input_array, n=None, m=None, lbs=None):
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
        obj.lbs = lbs
        return obj

    def select(self, states=None):
        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        return zmarray(self[:, ind], n=self.n[ind], m=self.m[ind])

    def set_lbs(self, lbs):
        self.lbs = lbs

    def take_lbs(self, lbs):
        ind = np.in1d(self.lbs, lbs)
        return zmarray(self[ind, :], self.n, self.m, self.lbs[ind])

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

    def rotinv(self):
        zm_complex = self.complex()
        mask = (zm_complex.m == 0)
        zm_complex[:, ~mask] = np.abs(zm_complex[:, ~mask])
        zm_rotinv = zm_complex.real
        return zmarray(zm_rotinv, zm_complex.n, zm_complex.m)

    def rot(self, theta):
        rot_matrix = construct_rot_matrix(self.n, self.m, theta)
        zm_rot = rot_matrix.dot(self.T).T
        return zmarray(zm_rot, self.n, self.m)

    def ref(self, theta):
        ref_matrix = construct_ref_matrix(self.n, self.m, theta)
        zm_rot = ref_matrix.dot(self.T).T
        return zmarray(zm_rot, self.n, self.m)

    def maps(self, n_fold, normalize=True):
        mat = get_rotation_symm_coeffs(n_fold, self.m)
        t = self.copy()
        t[:, self.m == 0] = 0
        if normalize:
            nn = norm(t, axis=1)[:, np.newaxis]
        else:
            nn = 1.
        t2 = ((t / nn) ** 2)
        symm_map = (t2.dot(mat)).sum(axis=1)
        size = np.sqrt(symm_map.shape[0])
        if size.is_integer():
            symm_map = symm_map.reshape(int(size), int(size))
        return symm_map

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

    def reconstruct(self, size):
        zps = generate_zps_from_nm(self.n, self.m, size, reshape=False)
        return self.dot(zps).reshape((-1, size, size))

    def __array_finalize__(self, obj):
        if obj is None: return
        self.n = getattr(obj, 'n', None)
        self.m = getattr(obj, 'm', None)
        self.lbs = getattr(obj, 'lbs', None)


def get_rotation_symm_coeffs(n_fold, m):
    a = np.zeros(shape=(len(m), len(m)))
    c = np.array([1. if e % n_fold == 0 else -1. / (n_fold - 1) for e in np.abs(m)])
    c[m == 0] = 0.
    return np.diag(c)


def construct_complex_matrix(n, m):
    j1 = zp_nm2j(n, m)
    j2 = zp_nm2j_complex(n, np.abs(m))
    uniq_j2 = np.unique(j2)
    num_rows, num_cols = len(uniq_j2), len(j1)
    vals = np.zeros(len(j1), dtype=np.complex)
    # m >= 0 should be 1
    vals[m >= 0] = 1
    vals[m < 0] = 1j

    d = dict(zip(uniq_j2, range(num_rows)))

    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=np.complex)
    for i, (ind, v) in enumerate(zip(j2, vals)):
        c_matrix[d[ind], i] = v
    return c_matrix


def construct_rot_matrix(n, m, theta):
    theta = np.deg2rad(theta)
    rot_matrix = np.zeros(shape=(len(n), len(n)))
    for i, (en, em) in enumerate(zip(n, m)):
        if em == 0:
            rot_matrix[i, zp_nm2j(en, em)] = 1.
        elif em < 0:
            # DON'T forget to multiply m
            t = theta * (-em)
            rot_matrix[i, zp_nm2j(en, em)] = np.cos(t)
            rot_matrix[i, zp_nm2j(en, -em)] = -np.sin(t)
        else:
            t = theta * em
            rot_matrix[i, zp_nm2j(en, em)] = np.cos(t)
            rot_matrix[i, zp_nm2j(en, -em)] = np.sin(t)
    return rot_matrix


def construct_ref_matrix(n, m, theta):
    theta = 2 * np.deg2rad(theta)
    ref_matrix = np.zeros(shape=(len(n), len(n)))
    for i, (en, em) in enumerate(zip(n, m)):
        if em == 0:
            ref_matrix[i, zp_nm2j(en, em)] = 1.
        elif em < 0:
            # DON'T forget to multiply m
            t = theta * (-em)
            ref_matrix[i, zp_nm2j(en, em)] = -np.cos(t)
            ref_matrix[i, zp_nm2j(en, -em)] = -np.sin(t)
        else:
            t = theta * em
            ref_matrix[i, zp_nm2j(en, em)] = np.cos(t)
            ref_matrix[i, zp_nm2j(en, -em)] = -np.sin(t)
    return ref_matrix