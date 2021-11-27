import numbers
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.transform import warp_polar
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from sklearn.neighbors import NearestNeighbors



# https://stackoverflow.com/a/50160920/5855131
def baseline_als(y, lam=105, p=0.1, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# see here: https://stackoverflow.com/q/57350711/5855131
def baseline_correction(y, niter=20):
    n = len(y)
    y_ = np.log(np.log(np.sqrt(y +1)+1)+1)
    yy = np.zeros_like(y)

    for pp in np.arange(1,niter+1):
        r1 = y_[pp:n-pp]
        r2 = (np.roll(y_,-pp)[pp:n-pp] + np.roll(y_,pp)[pp:n-pp])/2
        yy = np.minimum(r1,r2)
        y_[pp:n-pp] = yy

    baseline = (np.exp(np.exp(y_)-1)-1)**2 -1
    return baseline


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

def ring(r1, r2, center=None, shape=(1024, 1024)):
    if center is None:
        x, y = (0, 0)
    else:
        x, y = center
    if r1 >= r2:
        raise ValueError('r1 must be smaller than r2')
    if isinstance(shape, numbers.Number):
        shape = tuple(shape, shape)
    d = shape[0]
    radius = (d - 1) / 2
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    mask1 = np.array(((X-x) ** 2 + (Y-y) ** 2) >= r1 ** 2, dtype=np.uint8)
    mask2 = np.array(((X-x) ** 2 + (Y-y) ** 2) <= r2 ** 2, dtype=np.uint8)

    return mask1 * mask2

# this is not used
def _ring(r1, r2, shape=(1024, 1024)):
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




def get_strong_pts(img, j=3, sigma=None, shift=True, debug=False):
    if sigma is None:
        sigma = 1
    fft_complex = np.fft.fftshift(np.fft.fft2(img))
    a = np.abs(fft_complex)
    b = warp_polar(a)
    l = b.mean(axis=0)
    l_log = np.log(l + 1)
    base_line = baseline_correction(l_log)
    l_log_corrected = l_log - base_line
    l_log_corrected[0:j] = 0
    l_log_corrected[img.shape[0]//3:] = 0

    if debug:
        plt.plot(l_log_corrected)
        plt.plot(l_log)
        plt.plot(l_log*l_log_corrected)

    r, i, j, l = get_r_from_line(l_log_corrected, j=j)

    ring_mask = ring(i, j, shape=img.shape)
    a_mask = gaussian(a * ring_mask, 1)
    a_mask = a_mask / a_mask.max()


    yx = peak_local_max(a_mask, min_distance=3, threshold_abs=0.5)
    xy = np.fliplr(yx)
    y0, x0 = np.unravel_index(np.argmax(a), a.shape)
    xy = xy - [x0, y0]
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    xy = xy[np.argsort(angles)]
    if shift:
        pts1 = xy[range(0, len(xy), 2)] + [x0, y0]
        pts2 = xy[range(1, len(xy), 2)] + [x0, y0]
    else:
        pts1 = xy[range(0, len(xy), 2)]
        pts2 = xy[range(1, len(xy), 2)]
    return pts1, pts2

def _get_angle(pts1, pts2, n=512):
    # check centers of pts1 and pts2 are close to (0, 0)
    c1 = np.array(pts1).mean(axis=0)
    c2 = np.array(pts2).mean(axis=0)
    pts1 = pts1 - c1
    pts2 = pts2 - c2
    #d1 = np.hypot(c1[0], c1[1])
    #d2 = np.hypot(c2[0], c2[1])

    costs = []
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts2)
    angles = np.linspace(0, 60, n)
    for angle in angles:
        pts1_ = rotate_pts(pts1, angle)
        ind = nbrs.kneighbors(pts1_, return_distance=False)[:, 0]
        pts2_ = pts2[ind]
        dd = pts1_ - pts2_
        cost = np.hypot(dd[:, 0], dd[:, 1]).mean()
        costs.append(cost)
    costs = np.array(costs)
    return angles[np.argmin(costs)]

def _get_lattice_orientations(pts1, pts2):
    c1 = np.array(pts1).mean(axis=0)
    c2 = np.array(pts2).mean(axis=0)
    pts1 = pts1 - c1
    pts2 = pts2 - c2
    pts12 = np.vstack([pts1, pts2])
    r = np.hypot(pts12[:, 0], pts12[:, 1]).mean()
    angles = [60*i for i in range(6)]
    pts = np.array([rotate_pts([0, r], angle) for angle in angles])
    angle1 = _get_angle(pts, pts1)
    angle2 = _get_angle(pts, pts2)
    return (angle1, angle2)



def _get_motifs(a, angle1, angle2):
    def _generate_hex_lattice(a, n, angle, clip=False):
        u, v = (0, a), (np.sqrt(3)/2*a, 0.5*a)
        uv = np.stack([u, v])
        uv = rotate_pts(uv, angle)
        X, Y = np.meshgrid(range(-n, n), range(-n, n))
        xy = np.dstack([X, Y])
        xy = xy.reshape(-1, 2).dot(uv)

        d2 = X ** 2 + Y ** 2 + X * Y
        mask = d2 <= np.unique(d2)[n]
        xy = xy[mask.ravel()]

        l = 2 * np.sqrt(np.unique(d2)[n])
        if clip:
            mask1 = np.logical_and(np.abs(xy[:, 0]) <= l, np.abs(xy[:, 1]) <= l)
            xy = xy[mask1]
        return xy

    def _generate_gaussian(X, Y, x, y, sigma=None):
        if sigma is None:
            vmax = np.maximum(np.abs(X).max(), np.abs(Y).max())
            sigma = 0.1 * vmax
        return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))

    def _lattice2image(lattice, a, sigma=None):
        if sigma is None:
            sigma = 2
        r = (int(np.abs(lattice).max()/a)+1)*a
        t = np.arange(-r, r+1)
        X, Y = np.meshgrid(t, t)
        aa = np.zeros(X.shape)
        for (x, y) in lattice:
            aa += _generate_gaussian(X, Y, x, y, sigma)
        return aa

    n = 50
    lattice1 = _generate_hex_lattice(a, n, angle1)
    lattice2 = _generate_hex_lattice(a, n, angle2)
    template1 = _lattice2image(lattice1, a)
    template2 = _lattice2image(lattice2, a)

    return (template1, template2)


class BilayerImage:

    def __init__(self, img):
        self.image = img
        self.shape = img.shape


    def get_angle(self):
        pts1, pts2 = get_strong_pts(self.image)
        angle1 = _get_angle(pts1, pts2)
        angle2 = _get_angle(pts2, pts1)
        return min(angle1, angle2)

    def get_a(self):
        pts1, pts2 = get_strong_pts(self.image, shift=False)
        pts12 = np.vstack([pts1, pts2])
        l = self.shape[0]
        a = (l / np.hypot(pts12[:, 0], pts12[:, 1]) * 2 / np.sqrt(3)).mean()
        return a


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# bilayer fft image
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def find_central_region_xy(img):
    # otsu thresholding
    t = threshold_otsu(img)
    # segmentation
    lbs = label(img>t)
    regions = regionprops(lbs, intensity_image=img)
    a = np.array([e.area for e in regions])
    region = regions[np.argmax(a)]
    y, x = region.weighted_centroid
    return (x, y)


def get_ring_pts(img, j=3, debug=False, shift=True):
    x0, y0 = find_central_region_xy(img)
    b = warp_polar(img, center=(y0, x0))
    l = b.mean(axis=0)
    l_log = np.log(l + 1)
    base_line = baseline_correction(l_log, 20)
    l_log_corrected = l_log - base_line
    l_log_corrected[0:j] = 0
    l_log_corrected[img.shape[0] // 3:] = 0

    if debug:
        plt.plot(l_log_corrected)
        plt.plot(l_log)
        plt.plot(l_log * l_log_corrected)

    # first ring
    r1, i1, j1, l1 = get_r_from_line(l_log_corrected, j=3)
    # second ring
    r2, i2, j2, l2 = get_r_from_line(l1, j=j)

    ring_mask1 = ring(i1, j1, center=(x0, y0), shape=img.shape)
    mask1 = gaussian(img * ring_mask1, 1)
    mask1 = mask1 / mask1.max()

    ring_mask2 = ring(i2, j2, center=(x0, y0), shape=img.shape)
    mask2 = gaussian(img * ring_mask2, 1)
    mask2 = mask2 / mask2.max()

    yx = peak_local_max(mask1, min_distance=3, threshold_abs=0.1)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    xy = xy[np.argsort(angles)]

    pts1 = xy[range(0, len(xy), 2)]
    pts2 = xy[range(1, len(xy), 2)]

    yx = peak_local_max(mask2, min_distance=3, threshold_abs=0.1)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]
    pts1_ = np.roll(pts1, 1, axis=0)
    pts2_ = np.roll(pts2, 1, axis=0)
    pts3 = pts1 + pts1_
    pts4 = pts2 + pts2_

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xy)
    ind1 = nbrs.kneighbors(pts3, return_distance=False)[:, 0]
    ind2 = nbrs.kneighbors(pts4, return_distance=False)[:, 0]
    pp1 = np.vstack([pts1, xy[ind1]])
    pp2 = np.vstack([pts2, xy[ind2]])
    if shift:
        pp1 = pp1 + [x0, y0]
        pp2 = pp2 + [x0, y0]
    return pp1, pp2


class BilayerFFTImage:

    def __init__(self, img):
        self.image = img
        self.shape = img.shape


    def get_angle(self):
        pts1, pts2 = get_ring_pts(self.image, shift=False)
        angle1 = _get_angle(pts1, pts2)
        angle2 = _get_angle(pts2, pts1)
        return min(angle1, angle2)


def _get_first_ring(img):
    pass

def _get_second_ring(img):
    pass

def _get_r1_r2(img):
    pass
