import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection


def generate_lattice_image(a, theta=0, x0=0., y0=0., n=None, size=None, shift=False, f1=None, f2=None):
    k = 4 * np.pi / (a * np.sqrt(3))
    if n is None:
        n = 2
    if size is None:
        size = int(n * a)
        t = np.arange(-size, size)
    else:
        t = np.arange(-size // 2, size // 2)

    cost1 = np.cos(np.deg2rad(theta))
    sint1 = np.sin(np.deg2rad(theta))
    cost2 = np.cos(np.deg2rad(theta + 60))
    sint2 = np.sin(np.deg2rad(theta + 60))

    cost3 = np.cos(np.deg2rad(theta + 30))
    sint3 = np.sin(np.deg2rad(theta + 30))

    if shift:
        dx, dy = np.sqrt(3) * a / 3 * cost1, np.sqrt(3) * a / 3 * sint1
    else:
        dx, dy = 0., 0.

    # glide along diagonal
    if f1 is not None:
        x0, y0 = f1 * np.sqrt(3) * a / 3 * cost1, f1 * np.sqrt(3) * a / 3 * sint1

    # glide along a
    if f2 is not None:
        x0 += f2 * a * cost3
        y0 += f2 * a * sint3

    x, y = np.meshgrid(t + dx - x0, t + dy - y0)

    u, v = x * cost1 + y * sint1, x * cost2 + y * sint2
    z = 1 / 9 + 8 / 9 * np.cos(u * 0.5 * k) * np.cos(0.5 * k * v) * np.cos((u - v) * 0.5 * k)
    return z




def generate_layer_image(a=15, theta=0, size=256, A=0.3, center='Mo', f1=None, f2=None):
    if f1 is None:
        f1 = 0.
    if center in ['Mo', 'mo']:
        f1 = f1 + 0
    elif center in ['S', 's']:
        f1 = f1 + 1
    elif center in ['empty', '']:
        f1 = f1 + 2
    print(f1)
    lattice1 = generate_lattice_image(a, theta, 0., 0., None, size, shift=False, f1=f1, f2=f2)
    lattice2 = generate_lattice_image(a, theta, 0., 0., None, size, shift=True, f1=f1, f2=f2)
    return lattice1 ** 2 + A * lattice2 ** 2


def hex_lattice(a, theta=0, n=None, size=None, pad=True, shift=False, f1=0, f2=0):
    if n is None:
        n = 4
    if size is None:
        size = int(n * a)
    if pad:
        size_ = size + 2 * int(a)
    else:
        size_ = size

    theta = theta - 30
    cost1 = np.cos(np.deg2rad(theta))
    sint1 = np.sin(np.deg2rad(theta))
    cost2 = np.cos(np.deg2rad(theta + 60))
    sint2 = np.sin(np.deg2rad(theta + 60))

    cost3 = np.cos(np.deg2rad(theta + 30))
    sint3 = np.sin(np.deg2rad(theta + 30))

    # get uv
    u = (a * cost1, a * sint1)
    v = (a * cost2, a * sint2)
    uv = np.vstack([u, v])

    m = int(size_ * 4 / np.sqrt(6) / a) + 1
    x, y = np.indices((m, m)) - m // 2
    xy = np.array([x.ravel(), y.ravel()]).T
    pts = xy.dot(uv) + (size // 2, size // 2)

    if shift:
        dx, dy = -np.sqrt(3) * a / 3 * cost3, -np.sqrt(3) * a / 3 * sint3
    else:
        dx, dy = 0, 0

    # glide along diagonal
    if f1 is not None:
        dx += f1 * np.sqrt(3) * a / 3 * cost3
        dy += f1 * np.sqrt(3) * a / 3 * sint3

    # glide along a
    if f2 is not None:
        dx += f2 * a * cost2
        dy += f2 * a * sint2

    pts = pts + (dx, dy)
    return crop_xy(pts, size)


def crop_xy(pts, size):
    mask1 = np.logical_and(pts[:, 0] > 0, pts[:, 0] < size - 1)
    mask2 = np.logical_and(pts[:, 1] > 0, pts[:, 1] < size - 1)
    mask = mask1 * mask2
    return pts[mask]


def graphene_lattice(a, theta=0, size=512, center='empty', f1=0, f2=0):
    if center in ['a', 'A']:
        f1 = f1 + 0
    elif center in ['b', 'B']:
        f1 = f1 + 1
    elif center in ['empty', '']:
        f1 = f1 + 2
    latticeA = hex_lattice(a, theta, size=size, shift=False, f1=f1, f2=f2)
    latticeB = hex_lattice(a, theta, size=size, shift=True, f1=f1, f2=f2)
    return np.vstack([latticeA, latticeB])

class HexLattice:


    def __init__(self, a, theta=0, size=512, z=0, element='X'):
        self.a = a
        self.theta = theta
        self.size = size
        self.z = z
        self.element = element

        self.xy_ = hex_lattice(self.a, self.theta, None, self.size)
        self.xy = crop_xy(self.xy_, self.size)
        self.xyz = np.array([(x, y, self.z) for (x, y) in self.xy])

    def slide_a(self, s=0.5):
        pass

    def slide_diag(self, s=1):
        pass

class GrapheneLattice:

    def __init__(self, a=15, theta=0, size=512, center='empty', z=0, f1=0, f2=0):
        self.a = a
        self.theta = theta
        self.size = size
        self.center = center
        self.z = z
        self.element = 'C'
        self.f1 = f1
        self.f2 = f2

        self.xy = graphene_lattice(self.a, self.theta, self.size, self.center, self.f1, self.f2)
        self.xyz = np.array([(x, y, self.z) for (x, y) in self.xy])

    def plot(self, ax=None, lw=0.5, c='C0'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(self.xy)
        ds, inds = nbrs.kneighbors(self.xy)

        inds1 = np.array(np.arange(len(self.xy)).tolist() * 4).reshape(4, -1).T
        mask = np.logical_and(ds > 0, ds < self.a / np.sqrt(3) + 1)
        segs = np.array([[self.xy[ind1], self.xy[ind2]] for ind1, ind2 in zip(inds1[mask], inds[mask])])
        ls = LineCollection(segs, color=c, lw=lw)
        ax.add_collection(ls)
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size, 0)
