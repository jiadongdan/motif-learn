import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors


def get_lattice_image(a=15, theta=0., size=128, slide=(0, 0)):
    k = 4 * np.pi / (a * np.sqrt(3))
    t = np.arange(-size // 2, size // 2)

    # cost1, sint1, cost2, sint2 are used for lattice
    theta1 = theta + 30
    cost1 = np.cos(np.deg2rad(theta1))
    sint1 = np.sin(np.deg2rad(theta1))
    cost2 = np.cos(np.deg2rad(theta1 + 60))
    sint2 = np.sin(np.deg2rad(theta1 + 60))
    # cost3, sint3, cost4, sint4 are used for unit vectors
    cost3 = np.cos(np.deg2rad(theta))
    sint3 = np.sin(np.deg2rad(theta))
    cost4 = np.cos(np.deg2rad(theta + 60))
    sint4 = np.sin(np.deg2rad(theta + 60))

    u0 = np.array([a * cost3, a * sint3])
    v0 = np.array([a * cost4, a * sint4])
    dx, dy = slide[0] * u0 + slide[1] * v0

    # xy grids
    x, y = np.meshgrid(t + dx, t + dy)

    u, v = x * cost1 + y * sint1, x * cost2 + y * sint2
    z = 1 / 9 + 8 / 9 * np.cos(u * 0.5 * k) * np.cos(0.5 * k * v) * np.cos((u - v) * 0.5 * k)
    return z


def hex_lattice(a=15, theta=0., size=512, slide=(0, 0)):
    size_ = size + 2 * int(a)

    cost3 = np.cos(np.deg2rad(theta))
    sint3 = np.sin(np.deg2rad(theta))
    cost4 = np.cos(np.deg2rad(theta + 60))
    sint4 = np.sin(np.deg2rad(theta + 60))

    # get uv
    u = np.array([a * cost3, a * sint3])
    v = np.array([a * cost4, a * sint4])
    uv = np.vstack([u, v])

    m = int(size_ * 4 / np.sqrt(6) / a) + 1
    x, y = np.indices((m, m)) - m // 2
    xy = np.array([x.ravel(), y.ravel()]).T
    pts = xy.dot(uv) + (size // 2, size // 2)

    dx, dy = slide[0] * u + slide[1] * v
    pts = pts - (dx, dy)

    return crop_xy(pts, size)


def crop_xy(pts, size):
    mask1 = np.logical_and(pts[:, 0] > 0, pts[:, 0] < size - 1)
    mask2 = np.logical_and(pts[:, 1] > 0, pts[:, 1] < size - 1)
    mask = mask1 * mask2
    return pts[mask]


def get_layer_image(a=15, theta=0., size=512, A=0.5, center='mo', slide=(0, 0)):
    # slide1 and slide2 control the center
    if center in ['mo', 'Mo']:
        slide1 = np.array([0, 0])
        slide2 = np.array([1 / 3, 1 / 3])
    elif center in ['s', 'S']:
        slide1 = np.array([0, 0]) + [2 / 3, 2 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [2 / 3, 2 / 3]
    elif center in ['empty', '']:
        slide1 = np.array([0, 0]) + [1 / 3, 1 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [1 / 3, 1 / 3]
    else:
        raise ValueError("Invalid value for center.")
    # slide controls the global shift
    slide1 = slide1 + np.array(slide)
    slide2 = slide2 + np.array(slide)
    lattice1 = get_lattice_image(a=a, theta=theta, size=size, slide=slide1)
    lattice2 = get_lattice_image(a=a, theta=theta, size=size, slide=slide2)
    return lattice1 ** 2 + A * lattice2 ** 2


def graphene_lattice(a=15, theta=0., size=512, center='A', slide=(0, 0)):
    if center in ['a', 'A']:
        slide1 = np.array([0, 0])
        slide2 = np.array([1 / 3, 1 / 3])
    elif center in ['b', 'B']:
        slide1 = np.array([0, 0]) + [2 / 3, 2 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [2 / 3, 2 / 3]
    elif center in ['empty', '']:
        slide1 = np.array([0, 0]) + [1 / 3, 1 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [1 / 3, 1 / 3]
    else:
        raise ValueError("Invalid value for center.")
    # slide controls the global shift
    slide1 = slide1 + np.array(slide)
    slide2 = slide2 + np.array(slide)
    lattice1 = hex_lattice(a=a, theta=theta, size=size, slide=slide1)
    lattice2 = hex_lattice(a=a, theta=theta, size=size, slide=slide2)
    return np.vstack([lattice1, lattice2])


def mos2_lattice(a=15, theta=0., size=512, center='mo', slide=(0, 0)):
    if center in ['mo', 'Mo']:
        slide1 = np.array([0, 0])
        slide2 = np.array([1 / 3, 1 / 3])
    elif center in ['s', 'S']:
        slide1 = np.array([0, 0]) + [2 / 3, 2 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [2 / 3, 2 / 3]
    elif center in ['empty', '']:
        slide1 = np.array([0, 0]) + [1 / 3, 1 / 3]
        slide2 = np.array([1 / 3, 1 / 3]) + [1 / 3, 1 / 3]
    else:
        raise ValueError("Invalid value for center.")
    # slide controls the global shift
    slide1 = slide1 + np.array(slide)
    slide2 = slide2 + np.array(slide)
    lattice1 = hex_lattice(a=a, theta=theta, size=size, slide=slide1)
    lattice2 = hex_lattice(a=a, theta=theta, size=size, slide=slide2)
    return lattice1, lattice2


class GrapheneLattice:

    def __init__(self, a=15, theta=0, size=512, center='empty', z=0, slide=(0, 0)):
        self.a = a
        self.theta = theta
        self.size = size
        self.center = center
        self.z = z
        self.element = 'C'
        self.slide = slide

        self.xy = graphene_lattice(self.a, self.theta, self.size, self.center, self.slide)
        self.xyz = np.array([(x, y, self.z) for (x, y) in self.xy])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(self.xy)
        ds, inds = nbrs.kneighbors(self.xy)

        inds1 = np.array(np.arange(len(self.xy)).tolist() * 4).reshape(4, -1).T
        mask = np.logical_and(ds > 0, ds < self.a / np.sqrt(3) + 1)
        segs = np.array([[self.xy[ind1], self.xy[ind2]] for ind1, ind2 in zip(inds1[mask], inds[mask])])

        if 'lw' not in kwargs:
            kwargs['lw'] = 0.5
        if 'color' not in kwargs:
            kwargs['color'] = 'C0'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.7

        ls = LineCollection(segs, **kwargs)
        ax.add_collection(ls)
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size, 0)

    def to_image(self):
        # for graphene, A=1.0
        return get_layer_image(self.a, self.theta, self.size, 1., self.center, self.slide)

    # to do
    def to_xyz(self, filename=None):
        pass


def plot_mos2_bonds(ax, mo, s, a, color='gray', lw=3):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(mo)
    ds, inds = nbrs.kneighbors(s)

    mask = ds < a / np.sqrt(3) + 1
    inds1 = np.array(np.arange(len(s)).tolist() * 3).reshape(3, -1).T
    segs = np.array([[s[ind1], mo[ind2]] for ind1, ind2 in zip(inds1[mask], inds[mask])])

    ls = LineCollection(segs, color=color, lw=lw, zorder=-5)
    ax.add_collection(ls)


class MoS2Lattice:

    def __init__(self, a=15, theta=0, size=512, center='empty', z=0, slide=(0, 0)):
        self.a = a
        self.theta = theta
        self.size = size
        self.center = center
        self.z = z
        self.slide = slide

        # no z
        self.mo, self.s = mos2_lattice(self.a, self.theta, self.size, self.center, self.slide)

    def plot(self, ax=None, c1='C0', c2='C1', c3='gray', s=10, show_bonds=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        ax.scatter(self.mo[:, 0], self.mo[:, 1], c=c1, s=s)
        ax.scatter(self.s[:, 0], self.s[:, 1], c=c2, s=0.8 * s)

        if show_bonds:
            plot_mos2_bonds(ax, self.mo, self.s, self.a, color=c3, lw=np.sqrt(s) * 0.618)

        ax.axis('equal')

    def to_image(self, A=0.5):
        return get_layer_image(self.a, self.theta, self.size, A, self.center, self.slide)

    # to do
    def to_xyz(self, filename=None):
        pass
