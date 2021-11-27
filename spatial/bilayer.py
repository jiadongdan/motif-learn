import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import warp_polar
from skimage.filters import gaussian, threshold_li
from skimage.feature import peak_local_max
import numbers

from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import NearestNeighbors


def ring(r1, r2, shape=(1024, 1024)):
    if r1 >= r2:
        raise ValueError('r1 must be smaller than r2')
    if isinstance(shape, numbers.Number):
        shape = tuple(shape, shape)
    d = shape[0]
    radius = (d - 1) / 2
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    mask1 = np.array((X ** 2 + Y ** 2) >= r1 ** 2, dtype=np.uint8)
    mask2 = np.array((X ** 2 + Y ** 2) <= r2 ** 2, dtype=np.uint8)

    return mask1 * mask2


def get_r_from_line(l, j=3):
    h = l[j:int(len(l) * np.sqrt(2) / 2)].mean() * 1.2
    r = find_peaks(l, distance=len(l), height=h)[0]
    if len(r) == 0:
        return (None, None, None, None)
    else:
        r = r[0]

    i, j = peak_widths(l, [r], rel_height=0.7)[2:]
    i, j = int(i), int(j)
    l[0:j] = l.max()
    return (r, i, j, l)


def get_rs_from_line(l, j=3, return_ij=False):
    status = True
    rs = []
    ii = []
    jj = []
    cnt = 0
    while status:
        r, i, j, l = get_r_from_line(l, j)
        if r is None:
            status = False
        else:
            status = True
            rs.append(r)
            ii.append(i)
            jj.append(j)
            cnt += 1
            if cnt == 4:
                break
    if return_ij:
        return np.array(rs), np.array(ii), np.array(jj)
    else:
        return np.array(rs)

def get_strong_pts(img, j=3):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    a = np.abs(fft_complex)
    b = warp_polar(a)
    l = b.mean(axis=0)
    r, i, j, l = get_r_from_line(l, j=j)

    ring_mask = ring(i, j, shape=img.shape)
    a_mask = gaussian(a * ring_mask, 1)
    a_mask = a_mask / a_mask.max()

    yx = peak_local_max(a_mask, min_distance=3, threshold_abs=0.5)
    xy = np.fliplr(yx)
    y0, x0 = np.unravel_index(np.argmax(a), a.shape)
    xy = xy - [x0, y0]
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    xy = xy[np.argsort(angles)]
    pts1 = xy[range(0, len(xy), 2)] + [x0, y0]
    pts2 = xy[range(1, len(xy), 2)] + [x0, y0]

    return pts1, pts2


def get_ring_pts(img, threshold=0.4, tol=0.1):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    a = np.abs(fft_complex)
    b = warp_polar(a)
    l = b.mean(axis=0)

    ll = []

    aa = np.zeros_like(img)
    rs, ii, jj = get_rs_from_line(l, return_ij=True)
    for (r, i, j) in zip(rs, ii, jj):
        ring_mask = ring(i, j, shape=img.shape)
        a_mask = gaussian(a * ring_mask, 1)
        a_mask = a_mask / a_mask.max()
        aa += a_mask

    yx = peak_local_max(aa, min_distance=1, threshold_abs=threshold)
    xy = np.fliplr(yx)

    # fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    # ax.imshow(aa)
    # ax.scatter(xy[:, 0], xy[:, 1], color='r', s=8)

    y0, x0 = np.unravel_index(np.argmax(a), a.shape)
    xy = xy - [x0, y0]

    d = np.hypot(xy[:, 0], xy[:, 1])
    # first ring xy
    xy1 = xy[np.where(d < jj[0])[0]]
    angles = np.arctan2(xy1[:, 1], xy1[:, 0])
    xy1 = xy1[np.argsort(angles)]
    u1, v1 = xy1[0], xy1[2]
    u2, v2 = xy1[1], xy1[3]
    pts1 = get_pts_from_uv(xy, u1, v1, tol) + [x0, y0]
    pts2 = get_pts_from_uv(xy, u2, v2, tol) + [x0, y0]
    return aa, pts1, pts2


def get_mask1_mask2(pts1, pts2, a, n=5000, alpha=0.5):
    ind = np.argsort(a.ravel())[::-1][n]
    ij = np.unravel_index(ind, a.shape)
    mask = a > a[ij]
    pp = np.flipud(np.stack(np.where(mask))).T

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts1)
    d1, ind = nbrs.kneighbors(pp)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts2)
    d2, ind = nbrs.kneighbors(pp)

    d12 = np.hstack([d1, d2])

    # yy = softmax(d12, axis=1)
    yy = np.exp(alpha * d12) / np.sum(np.exp(alpha * d12), axis=1)[:, np.newaxis]

    mask1 = np.zeros_like(a)
    mask1[np.where(mask)] = yy[:, 0]
    mask2 = np.zeros_like(a)
    mask2[np.where(mask)] = yy[:, 1]
    return mask1, mask2


def get_pts_from_uv(xy, u, v, tol=0.1):
    uv = np.vstack([u, v])
    mn = xy.dot(np.linalg.inv(uv))
    xmin, xmax, ymin, ymax = mn[:, 0].min(), mn[:, 0].max(), mn[:, 1].min(), mn[:, 1].max()
    xy_ = np.array(
        [(x, y) for x in np.arange(int(xmin - 2), int(xmax + 2)) for y in np.arange(int(ymin - 2), int(ymax + 2))])

    # fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    # ax.scatter(mn[:, 0], mn[:, 1], color=cc.muted[0], s=20)
    # ax.scatter(xy_[:, 0], xy_[:, 1], color=cc.muted[3], s=8)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mn)
    d, ind = nbrs.kneighbors(xy_)
    pts = xy[ind[d <= tol]]
    return pts


def disk_mask(r, shape=(1024, 1024)):
    d = shape[0]
    radius = (d - 1) / 2
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    mask = np.array((X ** 2 + Y ** 2) <= r ** 2, dtype=np.bool)
    return mask


def get_img1_img2_from_pts1_pts2(pts1, pts2, img, n=1000, alpha=0.5):
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    a = np.abs(fft_complex)
    y0, x0 = np.unravel_index(np.argmax(a), a.shape)
    pp1 = pts1 - [x0, y0]
    r = round(np.hypot(pp1[:, 0], pp1[:, 1]).mean())
    mask = disk_mask(r + 5, shape=img.shape)
    a[~mask] = 0

    mask1, mask2 = get_mask1_mask2(pts1, pts2, a, n=n, alpha=alpha)
    img1 = np.abs(np.fft.ifft2(np.fft.fftshift(fft_complex * mask1)))
    img2 = np.abs(np.fft.ifft2(np.fft.fftshift(fft_complex * mask2)))

    return img1, img2

def get_img1_img2(img, n=1000, alpha=0.5):
    pts1, pts2 = get_strong_pts(img)
    img1, img2 = get_img1_img2_from_pts1_pts2(pts1, pts2, img, n, alpha)
    return img1, img2

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angles_between(v, u):
    a = np.array([angle_between(v1, v2) for v1, v2 in zip(v, u)])
    return np.rad2deg(a)

