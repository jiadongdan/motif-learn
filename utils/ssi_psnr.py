import numpy as np
from .noise_models import gn
from .noise_models import apply_scan_noise
from .noise_models import apply_poisson_noise
from .noise_models import add_gaussian_noise
from ..feature.zps_matrix6 import ZPs

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def noise_param2seed(noise_param):
    if noise_param < 1:
        noise_param = 1 / noise_param
    s = int(100000 * noise_param)
    return s


class NoiseData:

    def __init__(self, n_fold=3, sigma=8, N=100, sigmas=None, doses=None, jxs=None):
        img = gn(n_fold, sigma=sigma)
        self.N = N

        if doses is not None:
            ps = [apply_poisson_noise(img, N=self.N, dose=dose, seed=noise_param2seed(dose)) / dose for dose in doses]
            self.img = (img / img.max() * 255 + 10).astype(float)
            self.ps = np.vstack(ps)

        elif jxs is not None:
            ps = [apply_scan_noise(img, N=self.N, jx=jx, seed=noise_param2seed(jx)) for jx in jxs]
            self.img = img
            self.ps = np.vstack(ps)

        elif sigma is not None:
            ps = [add_gaussian_noise(img, N=self.N, sigma=sigma, seed=noise_param2seed(sigma)) for sigma in sigmas]
            self.img = img
            self.ps = np.vstack(ps)

        else:
            doses = [1 / 10, 1 / 100, 1 / 200, 1 / 500, 1 / 1000, 1 / 2000, 1 / 5000]
            ps = [apply_poisson_noise(self.img, N=self.N, dose=dose, seed=noise_param2seed(dose)) / dose for dose in
                  doses]
            self.img = (img / img.max() * 255 + 10).astype(float)
            self.ps = np.vstack(ps)

    def get_psnr(self, mode='baseline', n_max=12):
        if mode == 'baseline':
            data_range = np.ptp(self.img)
            ss = np.array([peak_signal_noise_ratio(self.img, e, data_range=data_range) for e in self.ps])
        elif mode == 'zps':
            zps = ZPs(n_max=n_max, size=self.ps.shape[1])
            zps.fit(self.ps)
            X = zps.moments
            ps_ = X.reconstruct(zps.data)
            data_range = np.ptp(self.img)
            ss = np.array([peak_signal_noise_ratio(self.img, e, data_range=data_range) for e in ps_])

        return ss.reshape(-1, self.N).T

    def get_ssi(self, mode='baseline', n_max=12):
        if mode == 'baseline':
            ss = np.array([structural_similarity(self.img, e) for e in self.ps])
        elif mode == 'zps':
            zps = ZPs(n_max=n_max, size=self.ps.shape[1])
            zps.fit(self.ps)
            X = zps.moments
            ps_ = X.reconstruct(zps.data)
            ss = np.array([structural_similarity(self.img, e) for e in ps_])

        return ss.reshape(-1, self.N).T
