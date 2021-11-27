import numpy as np
from .noise_models import apply_scan_noise
from .noise_models import apply_poisson_noise
from .noise_models import add_gaussian_noise
from .noise_models import gn

from ..feature.zps_matrix6 import ps_zps
from ..feature.zps_matrix6 import ps_pzp
from ..feature.dim_reduction import ps_pca

def _make_data(n_fold, N, doses=None, jxs=None, sigmas=None):
    img = gn(n_fold)

    if sigmas is not None:
        for sigma in sigmas:
            data = [add_gaussian_noise(img, N=N, sigma=sigma) for sigma in sigmas]

    elif doses is not None:
        data = [apply_poisson_noise(img, N=N, dose=dose) for dose in doses]

    elif jxs is not None:
        data = [apply_scan_noise(img, N=N, jx=jx) for jx in jxs]


def pca_evalaute_psnr(ps, img, ndim=66):
    ps_ = np.vstack([ps, img])
    X = ps_pca(ps_, ndim)
    


def zps_evalaute_psnr(ps, img, ndim=66):
    ps_ = np.vstack([ps, img])
    X = ps_zps(ps_, ndim)

class PSNRModel:

    def __init__(self, n_fold, N, doses, jxs, sigmas):
        self.n_fold = n_fold
        self.N = N
        self.sigmas = sigmas
        self.doses = doses
        self.jxs = jxs
        self.data = _make_data(self.n_fold, self.N, doses=self.doses, jxs=self.jxs, sigmas=self.sigmas)
        self.img = gn(self.n_fold)

    def evaluate(self, method='zps'):
        if method is 'zps':
            pass
        elif method is 'pca':
            pass

