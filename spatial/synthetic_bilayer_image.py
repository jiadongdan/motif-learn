import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy.spatial.transform import Rotation as R
from skimage.feature import peak_local_max


def generate_lattice_image(a, theta=0, x0=0, y0=0, n=None, size=1024, shift=False):
    k = 4 * np.pi / (a * np.sqrt(3))
    if n is None:
        n = 4
    if size is None:
        size = int(n * a)
        t = np.arange(-size, size)
    else:
        t = np.arange(-size // 2, size // 2)

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


def generate_layer_image(a, theta=0, x0=0, y0=0, n=None, size=1024, ratio=0.5, center='brighter'):
    # get brighter lattice
    lattice1 = generate_lattice_image(a, theta, x0, y0, n, size, shift=False)
    # get another lattice
    lattice2 = generate_lattice_image(a, theta, x0, y0, n, size, shift=True)

    if center == 'brighter':
        lattice12 = lattice1 ** 2 + ratio * lattice2 ** 2
    elif center == 'darker':
        lattice12 = ratio * lattice1 ** 2 + lattice2 ** 2
    return lattice12


def _valid_mode(mode):
    if mode not in ['P', 'AP']:
        mode = 'P'
    return mode


def hex_lattice(a, theta, n=None, size=None, shift=False):
    if n is None:
        n = 4
    if size is None:
        size = int(n * a)

    # get uv
    rot_matrix = R.from_euler('z', -theta, degrees=True).as_matrix()[0:2, 0:2]
    u = (a * np.sqrt(3) / 2, a / 2)
    v = (0, a)
    uv = np.vstack([u, v])
    uv = uv.dot(rot_matrix)

    m = int(size * 4 / np.sqrt(6) / a) + 1
    x, y = np.indices((m, m)) - m // 2
    xy = np.array([x.ravel(), y.ravel()]).T
    pts = xy.dot(uv) + (size // 2, size // 2)

    if shift:
        dx, dy = np.array([(a * 2 / np.sqrt(3), 0)]).dot(rot_matrix)[0]
    else:
        dx, dy = 0, 0
    pts = pts + (dx, dy)

    mask1 = np.logical_and(pts[:, 0] > 0, pts[:, 0] < size - 1)
    mask2 = np.logical_and(pts[:, 1] > 0, pts[:, 1] < size - 1)
    mask = mask1 * mask2
    return pts[mask]


class SyntheticBilayerImage:

    def __init__(self, a, theta, theta1=None, theta2=None, n=None, size=1024, ratio=0.5, mode='P'):
        self.a = a
        self.theta = theta

        if theta1 is None:
            self.theta1 = 0
        else:
            self.theta1 = theta1
        if theta2 is None:
            self.theta2 = theta
        else:
            self.theta2 = theta2

        self.n = n
        self.size = size
        self.ratio = ratio
        self.mode = _valid_mode(mode)

        if mode == 'P':
            self.layer1 = generate_layer_image(self.a, self.theta1, n=self.n, size=self.size, ratio=self.ratio,
                                               center='brighter')
            self.layer2 = generate_layer_image(self.a, self.theta2, n=self.n, size=self.size, ratio=self.ratio,
                                               center='brighter')
        elif mode == 'AP':
            self.layer1 = generate_layer_image(self.a, self.theta1, n=self.n, size=self.size, ratio=self.ratio,
                                               center='brighter')
            self.layer2 = generate_layer_image(self.a, self.theta2, n=self.n, size=self.size, ratio=self.ratio,
                                               center='darker')
        self.data = self.layer1 + self.layer2


    @property
    def pts1(self):
        return hex_lattice(self.a, self.theta1, self.n, self.size)

    @property
    def pts2(self):
        return hex_lattice(self.a, self.theta2, self.n, self.size)
    
    @property
    def pts1_(self):
        return hex_lattice(self.a, self.theta1, self.n, self.size, shift=True)

    @property
    def pts2_(self):
        return hex_lattice(self.a, self.theta2, self.n, self.size, shift=True)

    #@property
    #def pts1_(self):
    #    t = self.ratio * 0.4
    #    yx = peak_local_max(self.layer1, threshold_abs=t)
    #    xy = np.fliplr(yx)

        # remove pts1
    #    mask = self.layer1[yx[:, 0], yx[:, 1]] < (self.ratio + 1.)/2
    #    return xy[mask]

    #@property
    #def pts2_(self):
    #    t = self.ratio * 0.4
    #    yx = peak_local_max(self.layer2, threshold_abs=t)
    #    xy = np.fliplr(yx)

        # remove pts2
    #    mask = self.layer2[yx[:, 0], yx[:, 1]] < (self.ratio + 1.) / 2
    #    return xy[mask]

    @property
    def N(self):
        t = np.deg2rad(self.theta)
        return 1 / (4*np.sin(t/2) ** 2)

    @property
    def l(self):
        t = np.deg2rad(self.theta)
        return  self.a/(2 * np.sin(t/2))

    def to_xyz(self):
        pass

    def get_cell(self):
        p1 = (0, 0)
        p2 = (self.l, 0)
        p3 = (self.l / 2, self.l * np.sqrt(3) / 2)
        p4 = (-self.l / 2, self.l * np.sqrt(3) / 2)
        xy = np.array([p1, p2, p3, p4])
        # rotate xy and shift
        rot_matrix = R.from_euler('z', -self.theta / 2, degrees=True).as_matrix()[0:2, 0:2]
        xy = xy.dot(rot_matrix)
        xy = np.unique(np.vstack([xy, -xy]), axis=0) + np.array(self.data.shape) // 2
        return xy

    def plot(self, ax=None, pts12_only=True, show_cell=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        if pts12_only:
            ax.scatter(self.pts1[:, 0], self.pts1[:, 1], color='C0', s=4, zorder=1)
            ax.scatter(self.pts2[:, 0], self.pts2[:, 1], color='C1', s=4, zorder=2)
        else:
            ax.scatter(self.pts1[:, 0], self.pts1[:, 1], color='C0', s=4, zorder=1)
            ax.scatter(self.pts2[:, 0], self.pts2[:, 1], color='C0', s=4, zorder=2)
            ax.scatter(self.pts1_[:, 0], self.pts1_[:, 1], color='C1', s=4, zorder=1)
            ax.scatter(self.pts2_[:, 0], self.pts2_[:, 1], color='C1', s=4, zorder=2)
        ax.set_xlim(-0.5, self.data.shape[0]-0.5)
        ax.set_ylim(self.data.shape[0]-0.5, -0.5)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if show_cell:
            p1 = (0, 0)
            p2 = (self.l, 0)
            p3 = (self.l/2, self.l*np.sqrt(3)/2)
            p4 = (-self.l/2, self.l*np.sqrt(3)/2)
            xy = np.array([p1, p2, p3, p4])
            # rotate xy and shift
            rot_matrix = R.from_euler('z', -self.theta/2, degrees=True).as_matrix()[0:2, 0:2]
            xy = xy.dot(rot_matrix) + np.array(self.data.shape)//2

            poly = Polygon(xy, fill=False)
            ax.add_patch(poly)