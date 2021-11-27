import numpy as np
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve



# https://stackoverflow.com/a/50160920/5855131
def baseline_als(y, lam=105, p=0.1, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# see here: https://stackoverflow.com/q/57350711/5855131
def baseline_correction(y, niter=20):
    n = len(y)
    y_ = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    yy = np.zeros_like(y)

    for pp in np.arange(1, niter + 1):
        r1 = y_[pp:n - pp]
        r2 = (np.roll(y_, -pp)[pp:n - pp] + np.roll(y_, pp)[pp:n - pp]) / 2
        yy = np.minimum(r1, r2)
        y_[pp:n - pp] = yy

    baseline = (np.exp(np.exp(y_) - 1) - 1) ** 2 - 1
    return baseline