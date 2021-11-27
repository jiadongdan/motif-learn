import numpy as np
import numbers
import matplotlib.pyplot as plt


from scipy.signal import find_peaks, peak_widths
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

from skimage.filters import gaussian, threshold_otsu, window
from skimage.measure import regionprops, label
from skimage.transform import warp_polar, rotate
from skimage.feature import peak_local_max
from skimage.morphology import diamond

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans



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
    # radius = (d - 1) / 2
    L = np.arange(0, d)
    X, Y = np.meshgrid(L, L)
    mask1 = np.array(((X - x) ** 2 + (Y - y) ** 2) >= r1 ** 2, dtype=np.uint8)
    mask2 = np.array(((X - x) ** 2 + (Y - y) ** 2) <= r2 ** 2, dtype=np.uint8)

    return mask1 * mask2


def find_central_region_xy(img, method='max'):
    if method == 'max':
        img_ = gaussian(img, sigma=img.shape[0] / 1000)
        y0, x0 = np.unravel_index(np.argmax(img_), img_.shape)

    if method == 'seg':
        # otsu thresholding
        t = threshold_otsu(img)
        # segmentation
        lbs = label(img > t)
        regions = regionprops(lbs, intensity_image=img)
        a = np.array([e.area for e in regions])
        region = regions[np.argmax(a)]
        y0, x0 = region.weighted_centroid
    return (x0, y0)


def generate_lattice_image(a, theta=0, x0=0, y0=0, n=None, size=None, shift=False):
    k = 4 * np.pi / (a * np.sqrt(3))
    if n is None:
        n = 2
    if size is None:
        size = int(n * a)
        t = np.arange(-size, size)
    else:
        t = np.arange(0, size)

    cost1 = np.cos(np.deg2rad(theta))
    sint1 = np.sin(np.deg2rad(theta))
    cost2 = np.cos(np.deg2rad(theta + 60))
    sint2 = np.sin(np.deg2rad(theta + 60))

    if shift:
        dx, dy = np.sqrt(3) * a / 3 * cost1, np.sqrt(3) * a / 3 * sint1
    else:
        dx, dy = 0., 0.
    x, y = np.meshgrid(t + dx - x0, t + dy - y0)

    u, v = x * cost1 + y * sint1, x * cost2 + y * sint2
    z = 1 / 9 + 8 / 9 * np.cos(u * 0.5 * k) * np.cos(0.5 * k * v) * np.cos((u - v) * 0.5 * k)
    return z


def generate_layer_image(a, theta=0, x0=0, y0=0, n=None, size=None):
    lattice1 = generate_lattice_image(a, theta, x0, y0, n, size, shift=False)
    lattice2 = generate_lattice_image(a, theta, x0, y0, n, size, shift=True)
    return lattice1 ** 2 + 0.5 * lattice2 ** 3


def locate_two_peaks(l):
    idx = l.argmax()
    l1 = l.copy()
    l2 = l.copy()
    l1[:idx] = l.max()
    l2[idx:] = l.max()
    idx1 = find_peaks(l1, distance=len(l1))[0][0]
    idx2 = find_peaks(l2, distance=len(l2))[0][0]

    v1, v2 = l[idx1], l[idx2]
    if v1 > v2:
        i = idx1
    else:
        i = idx2
    return np.sort([i, idx])


def _get_ls(img_fft, center=None, sigma=2, return_ij=False, debug=False):
    # find center (x0, y0)
    if center is None:
        x0, y0 = find_central_region_xy(img_fft)
    else:
        x0, y0 = center

    # warp image around center
    b = warp_polar(img_fft, center=(y0, x0))

    # get radial intesnity
    l = b.mean(axis=0)
    l_log = np.log(l + 1)[:img_fft.shape[0] // 3]
    # smooth the curve
    l_log = gaussian(l_log, sigma)
    v = (l_log.max() + l_log.min()) / 2
    idx = (np.abs(l_log - v)).argmin()
    base_line = baseline_correction(l_log, 20)
    l_log_ = l_log - base_line
    l_log_[0:idx] = 0
    return l, l_log, l_log_

def _get_fft_bg(l):
    s1 = int(len(l)//np.sqrt(2))
    s2 = s1 // 2
    t = l[s2: s1].mean()
    return t

def _get_r1(l_log_):
    r1 = l_log_.argmax()
    return r1


def _get_r2(l_log_, r1):
    l = l_log_.copy()
    l[0:r1] = l_log_.max()
    r2 = find_peaks(l, distance=len(l))[0][0]
    return r2


def _get_r0(l_log_, r1):
    l = l_log_.copy()
    l[r1:] = l_log_.max()
    r0 = find_peaks(l, distance=len(l))[0][0]
    return r0


def _get_r3(l_log_, r2):
    l = l_log_.copy()
    l[0:r2] = l_log_.max()
    r3 = find_peaks(l, distance=len(l))[0][0]
    return r3


def _get_ring(img_fft, center, l_log_, r, sigma):
    i, j = peak_widths(l_log_, [r], rel_height=0.7)[2:]
    ring_mask = ring(i, j, center=center, shape=img_fft.shape)

    mask_image = gaussian(img_fft * ring_mask, sigma)
    mask_image = mask_image / mask_image.max() * ring_mask

    return mask_image


def _get_theta_from_pts(pts1, pts2, n=512, angles=None):
    # check centers of pts1 and pts2 are close to (0, 0)
    c = np.vstack([pts1, pts2]).mean(axis=0)

    pts1 = pts1 - c
    pts2 = pts2 - c

    costs = []
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts2)
    if angles is None:
        angles = np.linspace(-60, 60, n)
    for angle in angles:
        pts1_ = rotate_pts(pts1, angle)
        ind = nbrs.kneighbors(pts1_, return_distance=False)[:, 0]
        pts2_ = pts2[ind]
        dd = pts1_ - pts2_
        cost = np.hypot(dd[:, 0], dd[:, 1]).mean()
        costs.append(cost)
    costs = np.array(costs)
    return angles[np.argmin(costs)]


def _get_costs(pts1, pts2, n=512, angles=None):
    c = np.vstack([pts1, pts2]).mean(axis=0)

    pts1 = pts1 - c
    pts2 = pts2 - c

    costs = []
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts2)
    if angles is None:
        angles = np.linspace(-180, 180, n)
    for angle in angles:
        pts1_ = rotate_pts(pts1, angle)
        ind = nbrs.kneighbors(pts1_, return_distance=False)[:, 0]
        pts2_ = pts2[ind]
        dd = pts1_ - pts2_
        cost = np.hypot(dd[:, 0], dd[:, 1]).mean()
        costs.append(cost)
    costs = np.array(costs)
    return costs


def _get_theta_from_costs(costs1, costs2):
    A = np.fft.fft(costs1)
    B = np.fft.fft(costs2)
    Ar = -A.conjugate()
    Br = -B.conjugate()
    s1 = np.argmax(np.abs(np.fft.ifft(Ar * B)))
    s2 = np.argmax(np.abs(np.fft.ifft(A * Br)))
    s = min(s1, s2)
    n = len(costs1)
    theta = np.linspace(-180, 180, n)[s] - np.linspace(-180, 180, n)[0]
    return theta


def _get_fft_abs(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.
    img_ = img*win
    return np.abs(np.fft.fftshift(np.fft.fft2(img_)))


def _get_pts12_from_ring2(ring, center):
    x0, y0 = center
    t = threshold_otsu(ring)
    yx = peak_local_max(ring, min_distance=3, threshold_abs=t)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    xy = xy[np.argsort(angles)]

    pts1 = xy[range(0, len(xy), 2)] + [x0, y0]
    pts2 = xy[range(1, len(xy), 2)] + [x0, y0]
    return pts1, pts2


def _extend_pts_to_ring1(ring1, center, pts1, pts2):
    x0, y0 = center
    pts1 = pts1 - center
    pts2 = pts2 - center
    # obtain first ring points from second ring points
    pts1_ = np.roll(pts1, 1, axis=0)
    pts2_ = np.roll(pts2, 1, axis=0)
    pts3 = (pts1 + pts1_) / 3
    pts4 = (pts2 + pts2_) / 3

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts3)
    d, ind = nbrs.kneighbors(pts4)
    dmin = max(int((d.min() - 1) // 2), 1)

    yx = peak_local_max(ring1, min_distance=dmin, threshold_abs=0.1)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xy)
    ind1 = nbrs.kneighbors(pts3, return_distance=False)[:, 0]
    ind2 = nbrs.kneighbors(pts4, return_distance=False)[:, 0]

    pp1 = xy[ind1] + [x0, y0]
    pp2 = xy[ind2] + [x0, y0]

    return pp1, pp2

def _extend_pts_to_ring3(ring3, center, pts1, pts2):
    x0, y0 = center
    pts1 = pts1 - center
    pts2 = pts2 - center
    # obtain third ring points from second ring points
    pts1_ = np.roll(pts1, 1, axis=0)
    pts2_ = np.roll(pts2, 1, axis=0)
    pts3 = (pts1 + pts1_) / 3 * 2
    pts4 = (pts2 + pts2_) / 3 * 2

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts3)
    d, ind = nbrs.kneighbors(pts4)
    dmin = max(int((d.min() - 1) // 2), 1)

    yx = peak_local_max(ring3, min_distance=dmin, threshold_abs=0.1)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xy)
    ind1 = nbrs.kneighbors(pts3, return_distance=False)[:, 0]
    ind2 = nbrs.kneighbors(pts4, return_distance=False)[:, 0]

    pp1 = xy[ind1] + [x0, y0]
    pp2 = xy[ind2] + [x0, y0]

    return pp1, pp2


def _get_theta(pts1, center):
    pts1 = pts1 - center
    angles = np.linspace(0, 360, 7)[0:-1]

    pts1_rot = np.vstack([rotate_pts(pts1, angle) for angle in angles])

    # clustering pts1_
    model = KMeans(n_clusters=12).fit(pts1_rot)
    lbs = model.labels_

    pts1_ = np.array([(pts1_rot[lbs == e]).mean(axis=0) for e in np.unique(lbs)])

    # split into two groups according to distances
    dd = np.hypot(pts1_[:, 0], pts1_[:, 1])
    pp1 = pts1_[dd < dd.mean()]
    pp2 = pts1_[dd > dd.mean()]

    angles1 = np.rad2deg(np.arctan2(pp1[:, 0], pp1[:, 1]))
    angles2 = np.rad2deg(np.arctan2(pp2[:, 0], pp2[:, 1]))

    angles1 = angles1[np.argsort(np.abs(angles1))]
    angles2 = angles2[np.argsort(np.abs(angles2))]

    return (angles2[0] + angles1[0:2].mean()) / 2



class BilayerImage:

    def __init__(self, img, sigma=2, ring_sigma=0, use_win=True, include_ring3=False):
        self.image = img
        self.image_fft = _get_fft_abs(self.image, use_win)
        self.shape = img.shape
        self.sigma = sigma
        self.ring_sigma = ring_sigma
        self.use_win = use_win
        self.center = find_central_region_xy(self.image_fft, method='max')

        # radial intensiy
        self.l, self.l_log, self.l_log_ = _get_ls(self.image_fft, self.center, self.sigma)

        # radius
        self.r1 = _get_r1(self.l_log_)
        self.r0 = _get_r0(self.l_log_, self.r1)
        self.r2 = _get_r2(self.l_log_, self.r1)
        self.r3 = _get_r3(self.l_log_, self.r2)

        # rings
        self.ring0 = _get_ring(self.image_fft, self.center, self.l_log_, self.r0, self.ring_sigma)
        self.ring1 = _get_ring(self.image_fft, self.center, self.l_log_, self.r1, self.ring_sigma)
        self.ring2 = _get_ring(self.image_fft, self.center, self.l_log_, self.r2, self.ring_sigma)
        self.ring3 = _get_ring(self.image_fft, self.center, self.l_log_, self.r3, self.ring_sigma)


        # pts1 and pts2
        self.pts1, self.pts2 = self._pts12(include_ring3)

    def _pts12(self, include_ring3=False):
        pts1, pts2 = _get_pts12_from_ring2(self.ring2, self.center)
        pp1, pp2 = _extend_pts_to_ring1(self.ring1, self.center, pts1, pts2)
        if include_ring3:
            pp3, pp4 = _extend_pts_to_ring3(self.ring3, self.center, pts1, pts2)
            pp1 = np.vstack([pp1, pp3])
            pp2 = np.vstack([pp2, pp4])
        pp1 = np.vstack([pts1, pp1])
        pp2 = np.vstack([pts2, pp2])
        return pp1, pp2


    @property
    def a(self):
        l = self.shape[0]
        a = (l / self.r1 * 2 / np.sqrt(3))
        return a


    @property
    def pts0(self):
        yx = peak_local_max(self.ring0, min_distance=int(self.r0 // 2 - 1), threshold_abs=0.1)
        xy = np.fliplr(yx)
        return xy

    def get_layer1(self, mode=1):
        e1 = diamond(0)
        e2 = diamond(1)
        e3 = diamond(2)
        es = [e1, e2, e3]
        e = es[mode]
        s = e.shape[0]//2

        mask = np.zeros(self.shape)
        for (x, y) in self.pts1:
            mask[y-s:y+s+1, x-s:x+s+1] = e

        mask[self.center[1], self.center[0]] = .5
        mask[self.center[1] + 1, self.center[0] + 1] = .5
        mask[self.center[1] - 1, self.center[0] - 1] = .5
        mask[self.center[1] + 1, self.center[0] - 1] = .5
        mask[self.center[1] - 1, self.center[0] + 1] = .5

        layer1 = np.abs(np.fft.ifft2(np.fft.fftshift(mask * np.fft.fftshift(np.fft.fft2(self.image)))))
        return layer1

    def get_layer2(self, mode=1):
        e1 = diamond(0)
        e2 = diamond(1)
        e3 = diamond(2)
        es = [e1, e2, e3]
        e = es[mode]
        s = e.shape[0] // 2

        mask = np.zeros(self.shape)
        for (x, y) in self.pts2:
            mask[y - s:y+s + 1, x - s:x + s + 1] = e

        mask[self.center[1], self.center[0]] = .5
        mask[self.center[1] + 1, self.center[0] + 1] = .5
        mask[self.center[1] - 1, self.center[0] - 1] = .5
        mask[self.center[1] + 1, self.center[0] - 1] = .5
        mask[self.center[1] - 1, self.center[0] + 1] = .5


        layer2 = np.abs(np.fft.ifft2(np.fft.fftshift(mask * np.fft.fftshift(np.fft.fft2(self.image)))))
        return layer2

    def show_fft(self, ax=None, show_pts0=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        aa = self.image_fft
        x, y = self.center
        aa[y-1:y+2, x-1:x+2] = 0
        s = int(self.r2 * 1.5)
        ax.imshow(aa)
        ax.scatter(self.pts1[:, 0], self.pts1[:, 1], color='C0', s=4)
        ax.scatter(self.pts2[:, 0], self.pts2[:, 1], color='C1', s=4)
        if show_pts0:
            ax.scatter(self.pts0[:, 0], self.pts0[:, 1], color='C1', s=4)
        ax.set_xlim(x - s, x + s + 1)
        ax.set_ylim(y - s, y + s + 1)

    def show_rings(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        rings = self.ring1+self.ring2
        ax.imshow(rings)
        ax.axis('off')
        x, y = self.center
        s = int(self.r2 * 1.5)
        ax.set_xlim(x - s, x + s + 1)
        ax.set_ylim(y - s, y + s + 1)

    # this is slow
    def get_thetas_slow(self, n=3601):
        # construct hexagonal lattice
        angles = np.arange(60, 361, 60)
        pts1 = np.array([(self.r1 * np.cos(t), self.r1 * np.sin(t)) for t in np.deg2rad(angles)])
        pts2 = np.array([(self.r2 * np.cos(t), self.r2 * np.sin(t)) for t in np.deg2rad(angles - 30)])
        pts12 = np.vstack([pts1, pts2])

        costs1 = _get_costs(pts12, self.pts1 - self.center, n=n)
        costs2 = _get_costs(pts12, self.pts2 - self.center, n=n)

        t1 = np.mod(costs1.argmin() / 10, 60)
        t2 = np.mod(costs2.argmin() / 10, 60)
        return np.abs(t1 - t2), t1, t2

    def get_thetas(self):
        t1 = _get_theta(self.pts1, self.center)
        t2 = _get_theta(self.pts2, self.center)
        return np.abs(t1 - t2), t1, t2

def get_bond_pairs(pts1, pts2=None, radius=None):
    if pts2 is None:
        pts2 = pts1.copy()
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(pts1)
    g = nbrs.radius_neighbors_graph(pts2).toarray()
    np.fill_diagonal(g, 0)
    inds1, inds2 = np.nonzero(g)
    segs = np.array([[pts1[i], pts2[j]] for (i, j) in zip(inds1, inds2)])
    return segs