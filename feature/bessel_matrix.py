import numpy as np
from scipy.special import jn, jn_zeros
from sklearn.base import TransformerMixin, BaseEstimator

from .share_utils import nm2j, j2nm, grid_rt, _get_m_states, _fit


# from this paper : Zernike vs. Bessel circular functions in visual optics

def get_nm_coeff(n, m, cmn):
    if m == 0:
        return 1 / jn(m + 1, cmn)
    else:
        return np.sqrt(2) / jn(m + 1, cmn)

def get_jn_zeros(n_max):
    return np.vstack([jn_zeros(i, n_max+1) for i in range(0, n_max+1)])

def get_data_from_nm(n, m, size):
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    m_abs = np.abs(m)

    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]

    r, t = grid_rt(size)
    n_max = n.max()

    disk_array = (r > 0) * 1.
    disk_array[size // 2, size // 2] = 1.

    c = get_jn_zeros(n_max)
    Jnm = []
    for i, (n_, m_) in enumerate(zip(n, m_abs)):
        cmn = c[m_, n_]
        Rnm_ = jn(m_, cmn * r)
        coeff = get_nm_coeff(n_, m_, cmn)
        # disk_array is needed for m=0
        mt = funcs[i](m_ * t) * disk_array
        Jnm.append(coeff * Rnm_ * mt)
    return np.array(Jnm)



class Bessel(TransformerMixin, BaseEstimator):

    def __init__(self, n_max, size, states=None):
        self.n_max = n_max
        self.size = size
        self.states = states


        self.j_max = nm2j(n_max, n_max, alpha=0)
        nm = j2nm(np.arange(self.j_max + 1), alpha=0)
        self.n, self.m = nm[:, 0], nm[:, 1]

        states = _get_m_states(self.m, states)
        ind = np.where(np.in1d(np.abs(self.m), states))[0]
        self.n = self.n[ind]
        self.m = self.m[ind]

        self.n_components_ = len(self.n)
        self.data = get_data_from_nm(self.n, self.m, size)
        self.moments = None

    def fit(self, X, method='matrix'):
        self.moments = _fit(X, self.data, self.n, self.m, alpha=0, method=method)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        X_new = self.moments
        if X_new.shape[0] == 1:
            X_new = X_new[0]
        return X_new

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# functional form
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def ps_bessel(ps, n_components):
    n_max, _ = j2nm(n_components-1, alpha=0)[0]
    model = Bessel(n_max=n_max, size=ps.shape[1])
    model.fit(ps)
    X = model.moments
    return X[:, 0:n_components]