import numpy as np
from sklearn.preprocessing import normalize


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


def zp_nm2i(n, m):
    i = np.array(n ** 2 + 2 * n + 2 * m)
    mask = np.array(n) % 2 == 0
    i[mask] = i[mask] // 4
    i[~mask] = (i[~mask] - 1) // 4
    if i.size == 1:
        i = i.item()
    return i


def zp_i2nm(i):
    if not np.iterable(i):
        i = np.array([i])
    else:
        i = np.array(i)
    n = np.floor(np.sqrt(4 * i + 1) - 1).astype(np.int)
    mask = n % 2 == 0
    m = (4 * i - n ** 2 - 2 * n + 1) // 2
    m[mask] = (4 * i[mask] - n[mask] ** 2 - 2 * n[mask] + 1) // 2
    nm = np.array([n, m]).T
    if nm.shape[0] == 1:
        nm = nm[0]
    return nm


def pzp_j2nm(j):
    """Convert single index j to pair (n, m)"""
    if not np.iterable:
        j = np.array([j])
    else:
        j = np.array(j)
    n = np.floor(np.sqrt(j)).astype(np.int)
    # Here m can be negative
    m = j - n * (n + 1)
    return np.array([n, m]).T

def pzp_nm2j(n, m):
    n = np.array(n)
    m = np.array(m)
    return n * (n + 1) + m

def pzp_nm2i(n, m):
    n = np.array(n)
    m = np.array(m)
    #i = (n+1)*(n+2)//2-(n-m) - 1
    i = (n*n+n+2*m)//2
    return i

def pzp_i2nm(i):
    if not np.iterable:
        i = np.array([i])
    else:
        i = np.array(i)
    n = ((np.sqrt(8*i+1)-1)//2).astype(np.int)
    m = (2*i - n*n-n)//2
    nm = np.array([n, m]).T
    return nm


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


def construct_complex_matrix(n, m, mode='zm'):
    if mode == 'zm':
        j1 = zp_nm2j(n, m)
        j2 = zp_nm2i(n, np.abs(m))
    elif mode == 'pzm':
        j1 = pzp_nm2j(n, m)
        j2 = pzp_nm2i(n, np.abs(m))
    # j2 has duplicate elements
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


class zmarray(np.ndarray):
    """
    Class representing (pseudo) Zernike moments, which is a subclass of :py:class:`numpy.ndarray`.

    Parameters
    ----------
    input_array: array
        the input array to be converted to :py:class:`zmarray`.
    n: array_like
        the radial indices of the (pseudo) Zernike moments
    m: array_like
        the azimuthal indicices of the (pseudo) Zernike moments
    mode: str
        if :code:`mode='zm'`, it represents zernike moments.
        if :code:`mode='pzm'`, it represents pseudo Zernike moments.

    Attributes
    ----------
    n: array
        the radial indices of the (pseudo) Zernike moments.
    m: array
        the azimuthal indicices of the (pseudo) Zernike moments.
    n_samples_: int
        number of samples in the moments array.
    n_features_: int
        number of features in the moments array.

    Notes
    -----
    The attributes ``n``, ``m`` and ``n_features_`` will be updated after applying methods :py:meth:`select` and
    :py:meth:`complex`.

    Examples
    --------
    >>> import numpy as np
    >>> from stempy.feature import zmarray
    >>> X = np.random.rand(100,66)
    >>> zm = zmarray(X)
    >>> zm.n
    array([ 0,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,
        5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
        7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    >>> zm.m
    array([  0,  -1,   1,  -2,   0,   2,  -3,  -1,   1,   3,  -4,  -2,   0,
         2,   4,  -5,  -3,  -1,   1,   3,   5,  -6,  -4,  -2,   0,   2,
         4,   6,  -7,  -5,  -3,  -1,   1,   3,   5,   7,  -8,  -6,  -4,
        -2,   0,   2,   4,   6,   8,  -9,  -7,  -5,  -3,  -1,   1,   3,
         5,   7,   9, -10,  -8,  -6,  -4,  -2,   0,   2,   4,   6,   8,
        10])
    >>> zm_selected = zm.select(states=[3, 4])
    >>> zm_selected.m
    array([-3,  3, -4,  4, -3,  3, -4,  4, -3,  3, -4,  4, -3,  3, -4,  4])
    """
    def __new__(cls, input_array, n=None, m=None, mode='zm'):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.N = input_array.shape[1]
        obj.mode = mode
        if n is None:
            if obj.mode == 'zm':
                obj.n = zp_j2nm(np.arange(obj.N))[:, 0]
            elif obj.mode == 'pzm':
                obj.n = pzp_j2nm(np.arange(obj.N))[:, 0]
        else:
            obj.n = n
        if m is None:
            if obj.mode == 'zm':
                obj.m = zp_j2nm(np.arange(obj.N))[:, 1]
            elif obj.mode == 'pzm':
                obj.m = pzp_j2nm(np.arange(obj.N))[:, 1]
        else:
            obj.m = m
        obj.n_samples_, obj.n_features_ = obj.shape
        return obj


    def select(self, states=None):
        """
        Select (pseudo) Zernike moments with specified ``states``.

        Parameters
        ----------
        states: int or array-like, default: None
            the symmetry states to select polynomials

        Returns
        -------
        zm: :py:class:`zmarray`
            the moments array with selected components

        """
        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        zm = zmarray(self[:, ind], n=self.n[ind], m=self.m[ind], mode=self.mode)
        return zm

    def complex(self):
        """
        Convert (pseudo) Zernike moments to complex form.

        Returns
        -------
        complex_moments: :py:class:`zmarray`
            the complex form of moments array.

        """
        if self.dtype != np.complex:
            c_matrix = construct_complex_matrix(self.n, self.m, self.mode)
            zm_complex = c_matrix.dot(self.T).T
            # update n and m
            c_matrix[c_matrix == 1j] = 0
            m = c_matrix.dot(np.abs(self.m)).real.astype(np.int)
            n = c_matrix.dot(np.abs(self.n)).real.astype(np.int)
            complex_moments = zmarray(zm_complex, n, m, self.mode)
            return complex_moments
        else:
            return self

    def abs(self):
        """
        Calculate the absolute value element-wise.

        Returns
        -------
        absolute: :py:class:`zmarray`
            A :py:class:`zmarray` containing the absolute value of each element in original `zmarray`.
        """

        absolute = zmarray(np.abs(self), self.n, self.m, self.mode)
        return absolute

    def normalize(self, norm='l2'):
        """
        Scale the moments array individually to unit norm (vector length).

        Parameters
        ----------
        norm: 'l1', 'l2', or 'max', optional ('l2' by default)
            the norm to use to normalize each non zero sample.
        Returns
        -------
        normalized_moments: :py:class:`zmarray`
            normalized moments array.
        """
        if self.dtype != np.complex:
            return zmarray(normalize(self, axis=1, norm=norm), self.n, self.m, self.mode)
        else:
            return self

    def reconstruct(self, data):
        """
        .. todo::
            need modification
        """
        data_ = data.reshape(data.shape[0], -1)
        shape = (self.shape[0], data.shape[1], data.shape[2])
        return self.dot(data_).reshape(shape)
