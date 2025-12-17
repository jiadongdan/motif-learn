import numpy as np
from sklearn.preprocessing import normalize
def nm2j(n, m):
    """
    Convert Zernike radial order `n` and azimuthal frequency `m` to a single index `j`.

    Parameters
    ----------
    n : int or array_like of int
        Radial order(s), must be non-negative integer(s).
    m : int or array_like of int
        Azimuthal frequency (frequencies), must satisfy |m| ≤ n and (n - |m|) even.

    Returns
    -------
    j : int or ndarray of int
        Single index corresponding to the given `n` and `m`.

    Raises
    ------
    ValueError
        If input validation fails.

    Notes
    -----
    The mapping from (n, m) to `j` is given by:

        j = ((n + 2) * n + m) // 2

    This mapping ensures that each valid pair (n, m) maps to a unique integer `j`.

    Examples
    --------
    >>> nm2j(0, 0)
    0
    >>> nm2j([0, 1, 1], [0, -1, 1])
    array([0, 1, 2])
    """
    n = np.asarray(n)
    m = np.asarray(m)

    if n.shape != m.shape:
        raise ValueError("`n` and `m` must have the same shape.")

    # Validate that n and m are integer-valued
    if not np.all(np.isclose(n % 1, 0)):
        raise ValueError("Radial order `n` must be integer-valued.")
    if not np.all(np.isclose(m % 1, 0)):
        raise ValueError("Azimuthal frequency `m` must be integer-valued.")

    n = n.astype(int)
    m = m.astype(int)

    # Validate inputs
    if np.any(n < 0):
        raise ValueError("Radial order `n` must be non-negative.")
    if np.any(np.abs(m) > n):
        raise ValueError("Azimuthal frequency `m` must satisfy |m| ≤ n.")
    if np.any((n - np.abs(m)) % 2 != 0):
        raise ValueError("`n - |m|` must be even.")

    # Compute `j`
    j = ((n + 2) * n + m) // 2

    # Return scalar if inputs are scalars
    if j.shape == ():
        return j.item()
    else:
        return j

def nm2j_complex(n, m):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    
    # Validate inputs
    if not np.all(n >= 0):
        raise ValueError("Radial order n must be non-negative.")
    if not np.all(m >= 0):
        raise ValueError("Azimuthal frequency m must be non-negative.")
    if not np.all(np.abs(m) <= n):
        raise ValueError("Azimuthal frequency m must satisfy |m| ≤ n.")
    if not np.all((n - np.abs(m)) % 2 == 0):
        raise ValueError("n - |m| must be even.")
    
    i = np.array(n ** 2 + 2 * n + 2 * m)
    mask = np.array(n) % 2 == 0
    i[mask] = i[mask] // 4
    i[~mask] = (i[~mask] - 1) // 4
    if i.size == 1:
        i = i.item()
    return i


def check_array1d(data):
    """
    Converts a scalar or array-like input into a 1-dimensional numpy array.

    Parameters:
    input (scalar or array-like): The input to convert to a 1D numpy array.

    Returns:
    numpy.ndarray: A 1D numpy array based on the provided input.
    """
    # Convert the input to a numpy array, np.atleast_1d ensures it's at least 1D
    array = np.atleast_1d(data)

    # Flatten the array to ensure it is 1D
    return array.ravel()


def construct_complex_matrix(n, m):
    sort_idx = np.lexsort((m, n))
    n = n[sort_idx]
    m = m[sort_idx]

    j1 = nm2j(n, m)
    j2 = nm2j_complex(n, np.abs(m))

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

def construct_real_matrix(n, m):
    """
    Construct matrix to convert complex Zernike moments back to real representation.

    Parameters
    ----------
    n : array_like
        Radial orders for complex moments (n >= 0).
    m : array_like
        Azimuthal frequencies for complex moments (m >= 0).

    Returns
    -------
    inv_matrix : ndarray
        Transformation matrix to convert complex to real moments.
        Shape: (num_real, num_complex)
    n_real : ndarray
        Radial orders for real moments.
    m_real : ndarray
        Azimuthal frequencies for real moments (includes ±m).
    """
    n = np.asarray(n)
    m = np.asarray(m)

    # Reconstruct original (n, ±m) pairs
    n_real = []
    m_real = []

    for n_val, m_val in zip(n, m):
        if m_val == 0:
            n_real.append(n_val)
            m_real.append(0)
        else:  # m_val > 0
            n_real.extend([n_val, n_val])
            m_real.extend([m_val, -m_val])

    n_real = np.array(n_real)
    m_real = np.array(m_real)
    # Sort by (n, m) to get a canonical ordering
    sort_idx = np.lexsort((m_real, n_real))
    n_real = n_real[sort_idx]
    m_real = m_real[sort_idx]

    # Get forward transformation matrix: complex = c_matrix @ real
    c_matrix = construct_complex_matrix(n=n_real, m=m_real)
    # c_matrix.shape = (num_complex, num_real)

    num_complex, num_real = c_matrix.shape

    # Build inverse transformation matrix: real = inv_matrix @ complex
    # inv_matrix.shape = (num_real, num_complex)
    inv_matrix = np.zeros((num_real, num_complex), dtype=complex)

    for j in range(num_real):
        for i in range(num_complex):
            if c_matrix[i, j] == 1:
                # Real moment j comes from real part of complex moment i
                inv_matrix[j, i] = 1.0
            elif c_matrix[i, j] == 1j:
                # Real moment j comes from imaginary part of complex moment i
                inv_matrix[j, i] = -1j  # Multiply by -1j to extract imaginary part

    return inv_matrix, n_real, m_real


def construct_rot_maps_matrix(n_folds, m):
    """
    Constructs a matrix where each row is a transformation based on the array `m`.
    Each element in `n_folds` corresponds to a specific condition in `m`.

    Parameters:
        n_folds (array-like): Indices used for certain conditions.
        m (array-like): Array of markers to determine placement in matrix.

    Returns:
        numpy.ndarray: Matrix of transformations.
    """
    # Ensure inputs are 1D numpy arrays
    n_folds = check_array1d(n_folds)
    m = check_array1d(m)
    m = np.abs(m)

    # Initialize the matrix with zeros
    matrix = np.zeros((len(n_folds), len(m)))

    for i, n_fold in enumerate(n_folds):
        # Create a row initialized to zero
        # Do NOT use np.zeros_like
        row = np.zeros(len(m))
        is_current_fold = (m % n_fold == 0) & (m > 1)
        is_special_case = (m == 0) | (m == 1)
        # is_special_case = (m == 0)
        not_covered = ~(is_current_fold | is_special_case)

        # Set conditions based on `m`
        row[is_current_fold] = 1
        row[not_covered] = -1. / (n_fold - 1) if n_fold > 1 else 0  # Avoid division by zero

        # Place the row in the matrix
        matrix[i] = row

    return matrix


class zmoments:

    def __init__(self, data, n, m):
        self.data = data
        self.n = n
        self.m = m

    def to_complex(self):
        if self.data.dtype != complex:
            c_matrix = construct_complex_matrix(n=self.n, m=self.m)
            if self.data.ndim == 2:
                # zm_complex = c_matrix.dot((self.data).T).T
                zm_complex = np.dot(c_matrix, self.data.T).T
            elif self.data.ndim == 3:
                zm_complex = np.tensordot(c_matrix, self.data, axes=([1], [0]))
            else:
                raise ValueError("Invalid Zernike moment array shape.")
            # update n and m
            c_matrix[c_matrix == 1j] = 0
            m = c_matrix.dot(np.abs(self.m)).real.astype(int)
            n = c_matrix.dot(np.abs(self.n)).real.astype(int)
            return zmoments(data=zm_complex, n=n, m=m)
        else:
            return self

    def to_real(self):
        """
        Convert complex Zernike moments back to real representation.

        Returns
        -------
        zmoments
            Zernike moments in real representation.
        """
        if self.data.dtype == complex:
            # Get inverse transformation matrix
            inv_matrix, n_real, m_real = construct_real_matrix(self.n, self.m)

            if self.data.ndim == 2:
                zm_real = np.dot(self.data, inv_matrix.T).real
            elif self.data.ndim == 3:
                zm_real = np.tensordot(inv_matrix, self.data, axes=([1], [0])).real
            else:
                raise ValueError("Invalid Zernike moment array shape.")

            return zmoments(data=zm_real, n=n_real, m=m_real)
        else:
            # Already in real representation
            return self


    def normalize(self, order=None):
        # Check the dimension of the array
        if self.data.ndim == 2:
            # Normalize along axis 1 for 2D array
            norms = np.linalg.norm(self.data, ord=order, axis=1, keepdims=True)
            normalized_data = self.data / norms
        elif self.data.ndim == 3:
            # Normalize along axis 0 for 3D array
            norms = np.linalg.norm(self.data, ord=order, axis=0, keepdims=True)
            normalized_data = self.data / norms
        else:
            raise ValueError("Input must be a 2D or 3D array.")
        return zmoments(data=normalized_data, n=self.n, m=self.m)


    def select(self, m_select):
        m_select = check_array1d(m_select)
        m_select = np.unique(np.abs(m_select))

        ind = np.where(np.in1d(np.abs(self.m), m_select))[0]
        if self.data.ndim == 2:
            return zmoments(data=self.data[:, ind], n=self.n[ind], m=self.m[ind])
        elif self.data.ndim == 3:
            return zmoments(data=self.data[ind, :, :], n=self.n[ind], m=self.m[ind])
        else:
            raise ValueError("Invalid Zernike moment array shape, it can only be 2D or 3D.")

    def unselect(self, m_unselect):
        m_unselect = check_array1d(m_unselect)
        m_select = np.array([m for m in np.unique(np.abs(self.m)) if m not in m_unselect])
        return self.select(m_select)

    @property
    def is_complex(self):
        return self.data.dtype == 'complex128'

    def rotate(self, theta):
        zm_complex = self.to_complex()
        data_complex = zm_complex.data * np.exp(-np.deg2rad(theta)*1j)
        return zmoments(data=data_complex, n=zm_complex.n, m=zm_complex.m)


    def rot_maps(self, n_folds, p=2, m_unselect=None):
        # Normalize data before squaring (if p is not None)
        if self.data.ndim not in (2, 3):
            raise ValueError("Input must be a 2D or 3D array.")
        if m_unselect is None:
            m_unselect = (0, 1)
        elif 0 not in m_unselect:
            raise ValueError("m=0 must be included in m_unselect.")

        # Normalize if p is not None, otherwise use data as-is
        if p is not None:
            normalized_data = self.unselect(m_unselect).normalize(order=p).data
            # if unselect() is used, matrix changes too!
            matrix = construct_rot_maps_matrix(n_folds, self.unselect(m_unselect).m)
        else:
            normalized_data = self.unselect(m_unselect).data
            matrix = construct_rot_maps_matrix(n_folds, self.unselect(m_unselect).m)

        data2 = normalized_data ** 2

        if self.data.ndim == 2:
            return np.dot(data2, matrix.T)
        else:  # ndim == 3
            print(matrix.shape, data2.shape)
            return np.tensordot(matrix, data2, axes=([1], [0]))

    def mirror_map(self, theta=None, norm_order=2, m_unselect=[0, 1]):
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, 361)[0:360]
        if norm_order is None:
            zm = self.unselect(m_unselect=m_unselect)
        else:
            zm = self.unselect(m_unselect=m_unselect).normalize(order=norm_order)
            #zm_temp = self.unselect(m_unselect=m_unselect)
            #zm = zm_temp.normalize(order=norm_order)

        A = zm.to_complex().data.real
        B = zm.to_complex().data.imag
        part1 = A ** 2 - B ** 2
        part2 = 2 * A * B
        data = np.vstack([part1, part2])
        # self.moments_data = data

    # Notice
        ms = zm.to_complex().m
        cosmt = np.array([np.cos(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
        sinmt = np.array([np.sin(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
        matrix = np.hstack([cosmt, sinmt])  # 360 x N
        self.weights = matrix
        if self.data.ndim == 2:
            return np.dot(data, matrix.T).max(axis=1)
        elif self.data.ndim == 3:
            return np.tensordot(matrix, data, axes=([1], [0])).max(axis=0)
