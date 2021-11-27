import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.transform import warp_polar
from skimage.filters import threshold_otsu
from sklearn.neighbors import NearestNeighbors

from ..utils.blob_detection import local_max
from ..filters.common_filters import gaussian

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


def length(v):
    if len(v.shape) == 1:
        return np.hypot(v[0], v[1])
    elif len(v.shape) == 2:
        return np.hypot(v[:, 0], v[:, 1])

def hexagonal_uv(img, threshold=None, sigma=2, verbose=True):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    fft_abs = np.abs(fft_complex)
    line = warp_polar(fft_abs).mean(axis=0)
    d = find_peaks(line, distance=len(line))[0]

    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2 / 2))
        ax.plot(line[5:])
        ax.axvline(d - 5, color='C1')

    # find possible peaks in Fourier space
    fft_log = np.log(fft_abs + 1)
    if sigma is None:
        fft_log_ = fft_log.copy()
    else:
        fft_log_ = gaussian(fft_log, sigma=sigma)
    if threshold is None:
        threshold = threshold_otsu(fft_log_)
    pts = local_max(fft_log_, min_distance=3, threshold=threshold, plot=False)
    # get central points
    p0 = np.array(img.shape) // 2
    pts_ = pts - p0
    # select peaks within strongest ring
    dist = np.hypot(pts_[:, 0], pts_[:, 1])
    mask = np.logical_and(dist >= d - 2, dist <= d + 2)
    pts1 = pts_[mask]
    d_mean = length(pts1).mean()

    if pts1.shape[0] > 10:
        n = pts1.shape[0] // 2
        num_layers = 2
    else:
        n = pts1.shape[0]
        num_layers = 1

    p0 = pts1[0] / length(pts1[0]) * d_mean
    # lattice propagation to clean peaks
    pp1 = np.array([rotate_pts(p0, angle) for angle in np.linspace(0, 360, n, endpoint=False)])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts1)
    ind1 = nbrs.kneighbors(pp1)[1][:, 0]
    ind2 = np.array([e for e in np.arange(len(pts1)) if e not in ind1])
    print(ind1, ind2)

    u1 = pts1[0] / length(pts1[0]) * d_mean
    v1 = rotate_pts(u1, 60)
    uv1 = np.vstack([u1, v1])

    if num_layers == 2:
        u2 = pts1[ind2[0]] / length(pts1[ind2[0]]) * d_mean
        v2 = rotate_pts(u2, 60)
        uv2 = np.vstack([u2, v2])
        return (uv1, uv2)

    else:
        return uv1


def _get_first_ring_pts(img, threshold=None, sigma=2, verbose=True):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    fft_abs = np.abs(fft_complex)
    line = warp_polar(fft_abs).mean(axis=0)
    d = find_peaks(line, distance=len(line))[0]

    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2 / 2))
        ax.plot(line[5:])
        ax.axvline(d - 5, color='C1')

    # find possible peaks in Fourier space
    fft_log = np.log(fft_abs + 1)
    if sigma is None:
        fft_log_ = fft_log.copy()
    else:
        fft_log_ = gaussian(fft_log, sigma=sigma)
    if threshold is None:
        threshold = threshold_otsu(fft_log_)
    pts = local_max(fft_log_, min_distance=3, threshold=threshold, plot=verbose)
    # get central points
    p0 = np.array(img.shape) // 2
    pts_ = pts - p0
    # select peaks within strongest ring
    dist = np.hypot(pts_[:, 0], pts_[:, 1])
    mask = np.logical_and(dist >= d - 2, dist <= d + 2)
    pp = pts_[mask]
    return pp

def _get_basis(pp):
    d_mean = length(pp).mean()
    num_pts = pp.shape[0]
    if num_pts > 10:
        n = num_pts // 2
    else:
        n = num_pts
    p0 = pp[0]/length(pp[0])*d_mean
    pts1 = np.array([rotate_pts(p0, angle) for angle in np.linspace(0, 360, n, endpoint=False)])



def extract_uv(img, threshold=None, sigma=2, verbose=True):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    fft_abs = np.abs(fft_complex)
    line = warp_polar(fft_abs).mean(axis=0)
    d = find_peaks(line, distance=len(line))[0]

    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2/2))
        ax.plot(line[5:])
        ax.axvline(d-5, color='C1')

    # find possible peaks in Fourier space
    fft_log = np.log(fft_abs + 1)
    if sigma is None:
        fft_log_ = fft_log.copy()
    else:
        fft_log_ = gaussian(fft_log, sigma=sigma)
    if threshold is None:
        threshold = threshold_otsu(fft_log_)
    pts = local_max(fft_log_, min_distance=3, threshold=threshold, plot=verbose)
    # get central points
    p0 = np.array(img.shape) // 2
    pts_ = pts - p0
    # select peaks within strongest ring
    dist = np.hypot(pts_[:, 0], pts_[:, 1])
    mask = np.logical_and(dist >= d - 2, dist <= d + 2)
    pp = pts_[mask]
    d_mean = np.hypot(pp[:, 0], pp[:, 1]).mean()

    if pp.shape[0] > 10:
        n = pp.shape[0] // 2
        num_layers = 2
    else:
        n = pp.shape[0]
        num_layers = 1

    pp0 = pp[0]/length(pp[0])*d_mean
    # lattice propagation to clean peaks
    pts1 = np.array([rotate_pts(pp0, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
    # nbrs model
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pp)
    ind1 = nbrs.kneighbors(pts1)[1][:, 0]

    u1, v1 = pts1[0:2]
    if length(2 * u1) < length(u1 + v1):
        pts2 = np.array([rotate_pts(2 * u1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
        pts3 = np.array([rotate_pts(u1 + v1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
    else:
        pts2 = np.array([rotate_pts(u1 + v1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
        pts3 = np.array([rotate_pts(2 * u1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])

    if num_layers == 2:
        ind2 = np.array([e for e in np.arange(len(pp)) if e not in ind1])
        pp0 = pp[ind2[0]] / length(pp[ind2[0]]) * d_mean
        pts1 = np.array([rotate_pts(pp0, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
        u1, v1 = pts1[0:2]
        if length(2 * u1) < length(u1 + v1):
            pts2 = np.array([rotate_pts(2 * u1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
            pts3 = np.array([rotate_pts(u1 + v1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
        else:
            pts2 = np.array([rotate_pts(u1 + v1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])
            pts3 = np.array([rotate_pts(2 * u1, angle) for angle in np.linspace(0, 360, n, endpoint=False)])


def _nbrs_pts(pts, pp, radius=0):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)
    ind = nbrs.kneighbors(pp)[1][:, 0]
    return pts[ind]

class fft_pts(np.ndarray):
    def __new__(cls, pts, u=None, v=None, radius=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pts).view(cls)
        obj.u = u
        obj.v = v

    def clean(self, delta=0.1, radius=None):
        uv = np.vstack([self.u, self.v])
        pq = self.pts.dot(np.linalg.inv(uv))
        dist = np.round(pq) - pq
        r = np.hypot(dist[:, 0], dist[:, 1])
        pts = self.pts[r<delta]
        if radius is not None:
            r1 = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[r1 < radius]
        return fft_pts(pts, u=self.u, v=self.v)

class hex_fft_pts(fft_pts):
    def __init__(self, pts, u, v):
        self.pts = pts
        self.u = u
        self.v = v

    def select(self, n):
        pass








def extract_fft_peaks(img, threshold=None, tol=0.1, sigma=2, verbose=True):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    fft_abs = np.abs(fft_complex)
    line = warp_polar(fft_abs).mean(axis=0)
    d = find_peaks(line, distance=len(line))[0]

    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2/2))
        ax.plot(line[5:])
        ax.axvline(d-5, color='C1')

    # find possible peaks in Fourier space
    fft_log = np.log(fft_abs + 1)
    if sigma is None:
        fft_log_ = fft_log.copy()
    else:
        fft_log_ = gaussian(fft_log, sigma=sigma)
    if threshold is None:
        threshold = threshold_otsu(fft_log_)
    pts = local_max(fft_log_, min_distance=3, threshold=threshold, plot=verbose)
    # get central points
    p0 = np.array(img.shape) // 2
    pts_ = pts - p0
    # select peaks within strongest ring
    dist = np.hypot(pts_[:, 0], pts_[:, 1])
    mask = np.logical_and(dist >= d - 2, dist <= d + 2)
    pp = pts_[mask]
    d_mean = np.hypot(pp[:, 0], pp[:, 1]).mean()

    if pp.shape[0] > 10:
        n = pp.shape[0] // 2
        num_layers = 2
    else:
        n = pp.shape[0]
        num_layers = 1

    # lattice propagation to clean peaks
    pts1 = np.array([rotate_pts(pp[0], angle) for angle in np.linspace(0, 360, n, endpoint=False)])
    # nbrs model
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pp)
    ind1 = nbrs.kneighbors(pts1)[1][:, 0]

    u1 = pp[ind1[0]]
    v1 = pp[ind1[1]]
    uv1 = np.vstack([u1, v1])
    kk1 = pts_.dot(np.linalg.inv(uv1))

    dd1 = np.round(kk1) - kk1
    r1 = np.hypot(dd1[:, 0], dd1[:, 1])
    mask1 = r1 < tol

    if num_layers == 2:
        ind2 = np.array([e for e in range(pp.shape[0]) if e not in ind1])

        u2 = pp[ind2[0]]
        v2 = pp[ind2[1]]
        uv2 = np.vstack([u2, v2])
        kk2 = pts_.dot(np.linalg.inv(uv2))

        dd2 = np.round(kk2) - kk2
        r2 = np.hypot(dd2[:, 0], dd2[:, 1])
        mask2 = r2 < 0.1

        return pts[mask1], pts[mask2]
    else:
        return pts[mask1]


from skimage.morphology import disk


def generate_mask_collections(img, pts, size=10, sigma=None):
    aa = np.zeros_like(img)
    # pad the array
    aa = np.pad(aa, size)

    pts_ = pts.copy()
    # shift the points
    pts_[:, 0] += size
    pts_[:, 1] += size
    for (x, y) in pts_:
        aa[y - size:y + size + 1, x - size:x + size + 1] += disk(size)
    mask = aa > 0
    mask = mask[size:-size, size:-size]
    if sigma is not None:
        mask = gaussian(mask, sigma)
        mask = mask / mask.max()
    return mask