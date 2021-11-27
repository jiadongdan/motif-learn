import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy.spatial.transform import Rotation as R
from skimage.feature import peak_local_max
from skimage.filters import gaussian


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

def generate_lattice_image_conv(data, pts, a):
    for (x, y) in pts:
        data[int(y), int(x)] = 1.
    data = gaussian(data, sigma=a/4)
    return data/data.max()


def hex_lattice(a, theta, n=None, size=None, shift=False, pad=False):
    if n is None:
        n = 4
    if size is None:
        size = int(n * a)
    if pad:
        size = size + 2*int(a)

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

def _valid_mode(mode):
    if mode not in ['P', 'AP']:
        mode = 'P'
    return mode

def _valid_method(method):
    if method not in ['convolution', 'function']:
        method = 'function'
    return method

def _init_data(a, n, size):
    if n is None:
        n = 4
    if size is None:
        size = int(n * a)
    # make it larger
    return np.zeros(shape=(size+2*int(a), size+2*int(a)))


class SyntheticBilayerImage:

    def __init__(self, a, theta, theta1=None, theta2=None, n=None, size=1024, ratio=0.5, mode='P', method='convolution'):
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
        self.method = _valid_method((method))


        if self.method == 'function':

            # get image first
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

        elif self.method == 'convolution':

            # get points first
            self.pts1 = hex_lattice(self.a, self.theta1, self.n, self.size, pad=True)
            self.pts2 = hex_lattice(self.a, self.theta2, self.n, self.size, pad=True)
            self.pts1_ = hex_lattice(self.a, self.theta1, self.n, self.size, shift=True, pad=True)
            self.pts2_ = hex_lattice(self.a, self.theta2, self.n, self.size, shift=True, pad=True)

            init_a = _init_data(self.a, self.n, self.size)
            l1 = generate_lattice_image_conv(init_a, self.pts1, self.a)
            l1_ = generate_lattice_image_conv(init_a, self.pts1_, self.a)
            l2 = generate_lattice_image_conv(init_a, self.pts2, self.a)
            l2_ = generate_lattice_image_conv(init_a, self.pts2_, self.a)
            # now layer1 and layer2 is cropped
            self.layer1 = (l1 + self.ratio * l1_)[int(self.a):-int(self.a), int(self.a):-int(self.a)]
            self.layer2 = (l2 + self.ratio * l2_)[int(self.a):-int(self.a), int(self.a):-int(self.a)]
            self.data = (self.layer1 + self.layer2)


class Layer:

    def __init__(self, img, pts, pts_):
        self.img = img
        self.pts = pts
        self.pts_ = pts_

    def crop(self, size):
        y, x = np.array(self.img.shape)//2
        img = self.img[y-size//2:y+size//2, x-size//2:x+size//2]

        pts = self.pts - [x, y]
        pts_ = self.pts_ - [x, y]

        m1 = np.logical_and(pts[:, 0] > size//2, pts[:, 0] < size//2)
        m2 = np.logical_and(pts[:, 1] > size//2, pts[:, 1] < size//2)
        m3 = np.logical_and(pts_[:, 0] > size//2, pts_[:, 0] < size//2)
        m4 = np.logical_and(pts_[:, 1] > size//2, pts_[:, 1] < size//2)
        pts = pts[np.logical_and(m1, m2)] + [size//2, size//2]
        pts_ = pts_[np.logical_and(m3, m4)] + [size//2, size//2]
        return Layer(img, pts, pts_)

















