import numpy as np


def nm2j(n, m):
    n = np.atleast_1d(n)
    n = np.array(n)
    m = np.array(m)
    j = ((n + 2) * n + m) // 2
    return j


def nm2j_complex(n, m):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
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

    def rot_maps(self, n_folds):
        data2 = self.data ** 2
        matrix = construct_rot_maps_matrix(n_folds, self.m)
        if self.data.ndim == 2:
            return np.dot(data2, matrix.T)
        elif self.data.ndim == 3:
            return np.tensordot(matrix, data2, axes=([1], [0]))

    def mirror_map(self, theta=None, norm_order=None, m_unselect=[0, 1]):
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, 361)[0:360]
        zm = self.unselect(m_unselect=m_unselect).normalize(order=norm_order)
        A = zm.to_complex().data.real
        B = zm.to_complex().data.imag
        part1 = A ** 2 - B ** 2
        part2 = 2 * A * B
        data = np.vstack([part1, part2])

        # Notice
        ms = zm.to_complex().m
        cosmt = np.array([np.cos(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
        sinmt = np.array([np.sin(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
        matrix = np.hstack([cosmt, sinmt])  # 360 x 60
        if self.data.ndim == 2:
            return np.dot(data, matrix.T).max(axis=1)
        elif self.data.ndim == 3:
            return np.tensordot(matrix, data, axes=([1], [0])).max(axis=0)
