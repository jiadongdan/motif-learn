import numpy as np
import matplotlib.pyplot as plt
import numbers
from skimage.transform import warp_polar
from skimage.feature import peak_local_max
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import center_of_mass
from skimage.filters import gaussian, threshold_otsu

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans


def rotation_matrix(angle):
    return R.from_euler('z', angle, degrees=True).as_matrix()[0:2, 0:2]


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


def disk_patch(radius, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


def center_of_mass_refine(data, pts, size=3, mode=None):
    labels = np.zeros_like(data)
    index = []
    for ii, (x, y) in enumerate(pts):
        if mode == 'disk':
            labels[y - size:y + size + 1, x - size:x + size + 1] = (ii + 1) * disk_patch(size)
        else:
            labels[y - size:y + size + 1, x - size:x + size + 1] = ii + 1
        index.append(ii + 1)
    pts_ = center_of_mass(data, labels, index)

    return np.fliplr(np.array(pts_))


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


def _get_ring(img_fft_abs, r, center, sigma, ring_with=7):
    ring_mask = ring(r - ring_with, r + ring_with, center=center, shape=img_fft_abs.shape)

    mask_image = gaussian(img_fft_abs * ring_mask, sigma)
    mask_image = mask_image / mask_image.max() * ring_mask
    return mask_image


def _get_hex(ring, center):
    x0, y0 = center
    t = threshold_otsu(ring)
    # this does not guarantee 12 valid points
    yx = peak_local_max(ring, min_distance=3, threshold_abs=t)
    xy = np.fliplr(yx)
    xy = xy - [x0, y0]
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    xy = xy[np.argsort(angles)]

    pts1 = xy[range(0, len(xy), 2)] + [x0, y0]
    pts2 = xy[range(1, len(xy), 2)] + [x0, y0]

    pts1 = center_of_mass_refine(ring ** 2, pts1, size=5)
    pts2 = center_of_mass_refine(ring ** 2, pts2, size=5)
    return pts1, pts2


def _get_theta_from_hex(pts1, center):
    pts1 = pts1 - center
    angles = np.linspace(0, 360, 7)[0:-1]

    pts1_rot = np.vstack([rotate_pts(pts1, angle) for angle in angles])

    # clustering pts1_rot
    model = KMeans(n_clusters=len(pts1)).fit(pts1_rot)
    lbs = model.labels_

    pts1_ = np.array([(pts1_rot[lbs == e]).mean(axis=0) for e in np.unique(lbs)])
    thetas = np.rad2deg(np.arctan2(pts1_[:, 1], pts1_[:, 0])) + 180
    return np.sort(thetas)


class BilayerFFT:

    def __init__(self, fft_abs, ring_w=7, r0=None, s=None):
        self.fft_abs = fft_abs
        self.shape = self.fft_abs.shape
        self.center = np.array(self.shape) // 2
        self.ring_w = ring_w
        self.s = s

        b = warp_polar(self.fft_abs)
        # get radial intesnity
        l = b.mean(axis=0)
        if r0 is not None:
            l[0:int(r0)] = l.max()

        self.l_log = np.log(l + 1)[:self.fft_abs.shape[0] // 3]

        self.r1 = find_peaks(self.l_log, distance=len(self.l_log))[0][0]
        # estimate r2
        self.r2 = int(self.r1 * np.sqrt(3))

        self.ring1 = _get_ring(self.fft_abs, self.r1, self.center, sigma=2, ring_with=self.ring_w)
        self.ring2 = _get_ring(self.fft_abs, self.r2, self.center, sigma=2, ring_with=self.ring_w)

        self.hex1, self.hex2 = _get_hex(self.ring1, self.center)
        self.hex3, self.hex4 = _get_hex(self.ring2, self.center)

        self.theta1 = _get_theta_from_hex(self.hex1, self.center).min()
        self.theta2 = _get_theta_from_hex(self.hex2, self.center).min()
        self.t1 = _get_theta_from_hex(self.hex3, self.center).min()
        self.t2 = _get_theta_from_hex(self.hex4, self.center).min()

    @property
    def a(self):
        x0, y0 = self.center
        a3 = self.shape[0] / (np.hypot(self.hex3[:, 0] - x0, self.hex3[:, 1] - y0) / np.sqrt(3)).mean() * 2 / np.sqrt(3)
        a4 = self.shape[0] / (np.hypot(self.hex4[:, 0] - x0, self.hex4[:, 1] - y0) / np.sqrt(3)).mean() * 2 / np.sqrt(3)
        return 0.5*(a3+a4)


    def plot_l_log(self, **kwargs):
        plt.plot(self.l_log, '.-', **kwargs)

    def show_fft(self, **kwargs):
        plt.imshow(np.log(self.fft_abs + 1), **kwargs)

    def show_ring(self, **kwargs):
        plt.imshow(self.ring1 + self.ring2, **kwargs)