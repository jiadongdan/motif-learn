import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from skimage.morphology import disk
from skimage.filters import gaussian, threshold_otsu, threshold_multiotsu
from sklearn.neighbors import NearestNeighbors
from .blob_detection import local_max

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

def extract_pts_fft(img, offset=10, verbose=True):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    fft_abs = np.abs(fft_complex)
    line= warp_polar(fft_abs).mean(axis=0)[offset:]
    d = np.argmax(line)+offset

    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2/2))
        ax.plot(line, 'r-.')
        ax.axvline(d-offset)

    fft_log = np.log(fft_abs + 1)
    fft_log_ = gaussian(fft_log, sigma=3)
    t = threshold_otsu(fft_log_)
    # possible points in fft space
    pts = local_max(fft_log_, min_distance=3, threshold=t, plot=False)
    # get central points
    p0 = np.array(img.shape)//2
    pts_ = pts - p0

    dd = np.hypot(pts_[:, 0], pts_[:, 1])
    mask = np.logical_and(dd >= d - 2, dd <= d + 2)
    # selected points of strongest ring in fft space
    pp = pts_[mask]

    if pp.shape[0] > 10:
        n = pp.shape[0] // 2
        num_layers = 2
    else:
        n = pp.shape[0]
        num_layers = 1

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
    mask1 = r1 < 0.1

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


def get_img_from_fft_pts(img, pts, size=10, include_center=True):
    x0, y0 = np.array(img.shape) // 2

    selem = disk(size)
    aa = np.zeros_like(img)
    for (x, y) in pts:
        aa[y - size:y + size + 1, x - size: x + size + 1] = selem

    aa = np.fft.fftshift(aa)
    aa = np.fft.fft2(img) * aa
    bb = np.log(np.abs(aa) + 1)
    # get threshold from bb
    n = 3
    t = threshold_multiotsu(bb, n)
    mask = (bb > t[n - 2]) * 1.0
    mask_shift = np.fft.fftshift(mask)

    if include_center:
        mask_shift[y0 - size - 1:y0 + size + 1, x0 - size - 1: x0 + size + 1] *= 0.5
    else:
        mask_shift[y0 - size - 1:y0 + size + 1, x0 - size - 1: x0 + size + 1] = 0.
    mask = np.fft.fftshift(mask_shift)
    gg = np.fft.ifft2(np.fft.fft2(img) * mask)

    return np.real(gg)


def get_pts_bilayer(pts1, pts2, t=None):
    if pts1.shape[0] > pts2.shape[0]:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts1)
        d, inds = nbrs.kneighbors(pts2)
        ind = inds[:, 0]
        d = d[:, 0]
        pts = (pts1[ind] + pts2) / 2

    else:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts2)
        d, inds = nbrs.kneighbors(pts1)
        ind = inds[:, 0]
        d = d[:, 0]
        pts = (pts2[ind] + pts1) / 2
    if t is None:
        t = threshold_otsu(d)
    return pts[d <= t]

