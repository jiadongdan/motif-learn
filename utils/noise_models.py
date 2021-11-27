from ..feature.zps_matrix6 import ps_zps
from ..feature.zps_matrix6 import ps_pzp
from ..feature.dim_reduction import ps_pca
from ..feature.dim_reduction import ps_kpca
from ..feature.bessel_matrix import ps_bessel

from .evaluation import get_db
from .evaluation import get_sc
from .evaluation import get_ari
from .evaluation import get_ami
from .evaluation import get_fmi
from .evaluation import get_ch
from .evaluation import get_auc

from ..clustering.auto_clustering import kmeans_lbs

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.util import img_as_ubyte

def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


def many_gaussians(pts, sigma, s):
    data = np.zeros((s, s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X - x) ** 2 - (Y - y) ** 2) / (2 * sigma * sigma))
    return data

def gn(n, size=128, sigma=None, l=None, theta=0, include_center=True, A=0.9):
    if sigma is None:
        sigma = size / 20
    if l is None:
        l = size * 0.25
    else:
        l = size * l
    p0 = np.array([0, l])
    p0 = rotate_pts(p0, theta)
    pts = np.array([rotate_pts(p0, 360 * i / n) for i in range(n)])
    if include_center:
        gs = many_gaussians(pts, sigma, size) + A * many_gaussians([(0, 0)], sigma, size)
    else:
        gs = many_gaussians(pts, sigma, size)

    return gs/gs.max()



def rng_seeds(seed, N):
    INT32_MAX = np.iinfo(np.int32).max - 1

    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(0, INT32_MAX, N)
    return seeds


def apply_scan_noise_single_img(img, jx=1, jy=0, seed=48):
    if jx is None:
        jx = 0
    if jy is None:
        jy = 0

    t = range(img.shape[0])
    x, y = np.meshgrid(t, t)
    s = img.shape[0]
    rng = np.random.default_rng(seed=seed)
    dx = rng.normal(0, 1, (s, 1)) * jx
    dy = rng.normal(0, 1, (1, s)) * jy
    coords = np.array([y + dy, x + dx])
    kk = map_coordinates(img, coords)
    return kk


def apply_poisson_noise_single_img(img, dose=1 / 10, seed=48):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=img * dose)


def add_gaussian_noise_single_img(img, sigma=0.3, seed=48):
    rng = np.random.default_rng(seed)
    out = img + rng.normal(0, sigma, img.shape)
    return out


def apply_poisson_noise(imgs, dose=1 / 10, N=10, seed=48):
    imgs = np.asarray(imgs)
    imgs = img_as_ubyte(imgs).astype(int) + 10
    if imgs.ndim == 2:
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([apply_poisson_noise_single_img(imgs, dose, seed) for seed in seeds])
    elif imgs.ndim == 3:
        # set N to the number of imgs
        N = imgs.shape[0]
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([apply_poisson_noise_single_img(imgs[ii], dose, seed) for ii, seed in enumerate(seeds)])
    else:
        raise ValueError('only accept 2d or 3d image as input.')

    if imgs_noise.shape[0] == 1:
        return imgs_noise[0]
    else:
        return imgs_noise


def apply_scan_noise(imgs, jx=1, jy=0, N=10, seed=48):
    imgs = np.asarray(imgs)
    if imgs.ndim == 2:
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([apply_scan_noise_single_img(imgs, jx, jy, seed) for seed in seeds])
    elif imgs.ndim == 3:
        # set N to the number of imgs
        N = imgs.shape[0]
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([apply_scan_noise_single_img(imgs[ii], jx, jy, seed) for ii, seed in enumerate(seeds)])
    else:
        raise ValueError('only accept 2d or 3d image as input.')

    if imgs_noise.shape[0] == 1:
        return imgs_noise[0]
    else:
        return imgs_noise


def add_gaussian_noise(imgs, sigma=0.3, N=10, seed=48):
    imgs = np.asarray(imgs)
    if imgs.ndim == 2:
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([add_gaussian_noise_single_img(imgs, sigma, seed) for seed in seeds])
    elif imgs.ndim == 3:
        # set N to the number of imgs
        N = imgs.shape[0]
        seeds = rng_seeds(seed, N)
        imgs_noise = np.array([add_gaussian_noise_single_img(imgs[ii], sigma, seed) for ii, seed in enumerate(seeds)])
    else:
        raise ValueError('only accept 2d or 3d image as input.')

    if imgs_noise.shape[0] == 1:
        return imgs_noise[0]
    else:
        return imgs_noise


def make_two_classes(n1=100, n2=100, n_fold=3, sigma=None, dose=None, jx=None, jy=None, size=128, A=0.8, seed=48):
    img1 = gn(n_fold, A=1.0, size=size)
    img2 = gn(n_fold, A=A, size=size)

    seed1, seed2 = rng_seeds(seed, 2)

    if dose is not None:
        ps1 = apply_poisson_noise(img1, N=n1, dose=dose, seed=seed1)
        ps2 = apply_poisson_noise(img2, N=n2, dose=dose, seed=seed2)
    elif jx is not None or jy is not None:
        ps1 = apply_scan_noise(img1, N=n1, jx=jx, jy=jy, seed=seed1)
        ps2 = apply_scan_noise(img2, N=n2, jx=jx, jy=jy, seed=seed2)
    elif sigma is not None:
        ps1 = add_gaussian_noise(img1, N=n1, sigma=sigma, seed=seed1)
        ps2 = add_gaussian_noise(img2, N=n2, sigma=sigma, seed=seed2)
    else:
        dose = 1/10
        ps1 = apply_poisson_noise(img1, N=n1, dose=dose, seed=seed1)
        ps2 = apply_poisson_noise(img2, N=n2, dose=dose, seed=seed2)
    ps = np.vstack([ps1, ps2])
    lbs = np.array([0]*n1+[1]*n2)
    return ps, lbs


def make_data_batch(n1=100, n2=100, n=3, sigmas=None, jxs=None, doses=None, size=128, A=0.8, seed=48):
    ps_list = []
    lbs_list = []
    if sigmas is not None:
        for sigma in sigmas:
            ps, lbs = make_two_classes(n1, n2, n, sigma=sigma, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    elif doses is not None:
        for dose in doses:
            ps, lbs = make_two_classes(n1, n2, n, dose=dose, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    elif jxs is not None:
        for jx in jxs:
            ps, lbs = make_two_classes(n1, n2, n, jx=jx, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    return ps_list, lbs_list

def _make_data(n1, n2, n, sigmas, jxs, doses, size, A, seed):
    ps_list = []
    lbs_list = []
    if sigmas is not None:
        for sigma in sigmas:
            ps, lbs = make_two_classes(n1, n2, n, sigma=sigma, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    elif doses is not None:
        for dose in doses:
            ps, lbs = make_two_classes(n1, n2, n, dose=dose, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    elif jxs is not None:
        for jx in jxs:
            ps, lbs = make_two_classes(n1, n2, n, jx=jx, size=size, A=A, seed=seed)
            ps_list.append(ps)
            lbs_list.append(lbs)

    return ps_list, lbs_list


dim_methods = {'pca': ps_pca, 'zps':ps_zps, 'pzp':ps_pzp, 'bessel':ps_bessel, 'kpca':ps_kpca, }
eval_methods = {'sc':get_sc, 'ari':get_ari, 'ami':get_ami, 'fmi':get_fmi, 'db':get_db, 'auc':get_auc, 'ch': get_ch}

def _single_dim_eval(ps, lbs, dim_method, eval_method, ndim=66):
    X = dim_methods[dim_method](ps, ndim)

    classification_settings = ['auc']
    clustering_settings = ['ari', 'ami', 'sc', 'fmi', 'db']

    if eval_method in classification_settings:
        pass
    elif eval_method in clustering_settings:
        # kmeans
        lbs_pre = kmeans_lbs(X, 2)
    # get score
    if eval_method in ['sc', 'db', 'ch']:
        score = eval_methods[eval_method](X, lbs)
    elif eval_method in ['ari', 'ami', 'fmi']:
        score = eval_methods[eval_method](lbs, lbs_pre)
    return score

class NoiseDataModel:

    def __init__(self, n1=1000, n2=1000, sigmas=None, jxs=None, doses=None, seed=48):
        self.n1 = n1
        self.n2 = n2
        # noise params
        self.sigmas = sigmas
        self.jxs = jxs
        self.doses = doses
        self.seed = seed

        self.scores = None
        self.ps_list = None
        self.lbs_list = None


    def evaluate(self, n, dim_method='pca', eval_method='sc', ndim=66, size=128, A=0.8, verbose=True):
        # make data:
        if self.ps_list is None and self.lbs_list is None:
            self.ps_list, self.lbs_list = _make_data(self.n1, self.n2, n, self.sigmas, self.jxs, self.doses, size=size, A=A, seed=self.seed)
        # dim reduction and evaluate
        self.scores = np.array([_single_dim_eval(ps, lbs, dim_method, eval_method, ndim) for ps, lbs in zip(self.ps_list, self.lbs_list)])

        return self.scores
