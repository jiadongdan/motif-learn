import numpy as np

from .share_utils import (nm2j, j2nm, nm2j_complex, _get_m_states)


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

