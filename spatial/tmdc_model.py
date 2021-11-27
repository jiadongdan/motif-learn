import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors



def hex_lattice(a, theta=0, n=None, size=None, shift=False, pad=False):
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

def get_monolayer(a, theta, n=None, size=None, pad=False, center='M', z=0):
    if center in ['M', 'Mo', 'mo']:
        mo = hex_lattice(a, theta, n, size, shift=False, pad=pad)
        s1 = hex_lattice(a, theta, n, size, shift=True, pad=pad)
        s2 = hex_lattice(a, theta, n, size, shift=True, pad=pad)
    elif center in ['X', 'S', 's']:
        mo = hex_lattice(a, theta, n, size, shift=True, pad=pad)
        s1 = hex_lattice(a, theta, n, size, shift=False, pad=pad)
        s2 = hex_lattice(a, theta, n, size, shift=False, pad=pad)

    z0 = np.atleast_2d([z]*len(mo)).T
    z1 = np.atleast_2d([z]*len(s1)).T - a/2
    z2 = np.atleast_2d([z]*len(s2)).T + a/2

    mo = np.hstack([mo, z0])
    s1 = np.hstack([s1, z1])
    s2 = np.hstack([s2, z2])

    return (mo, s1, s2)


def plot_bonds(ax, a, atoms1, atoms2, direction='xy', color='gray', lw=3.):
    if direction == 'xy':
        ind1, ind2 = 0, 1
    elif direction == 'xz':
        ind1, ind2 = 0, 2
    elif direction == 'yz':
        ind1, ind2 = 1, 2

    nbrs = NearestNeighbors(radius=a, algorithm='ball_tree').fit(atoms2)
    d, inds = nbrs.radius_neighbors(atoms1, a)

    segs = [[atoms1[i][[ind1, ind2]], atoms2[ind][[ind1, ind2]]] for i, row in enumerate(inds) for ind in row]

    line_segments = LineCollection(segs, zorder=-2, color=color, lw=lw)
    ax.add_collection(line_segments)


class monolayer:

    def __init__(self, n=None, size=None, a=3.16, theta=0, center='M', pad=False):
        self.a = a
        self.theta = theta
        self.n = n
        self.size = size
        self.center = center
        self.pad = pad

        self.mo, self.s1, self.s2 = get_monolayer(self.a, self.theta, self.n, self.size, self.pad)

    def to_xyz(self, file_name):
        pass

    def plot(self, ax=None, c1='C0', c2='C1', c3='gray', add_bonds=True, direction='xy'):
        if direction == 'xy':
            ind1, ind2 = 0, 1
        elif direction == 'xz':
            ind1, ind2 = 0, 2
        elif direction == 'yz':
            ind1, ind2 = 1, 2

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        else:
            fig = ax.figure

        sc1 = ax.scatter(self.mo[:, 0], self.mo[:, 1], c=c1, s=1)
        sc2 = ax.scatter(self.s1[:, 0], self.s1[:, 1], c=c2, s=1)
        # ax.scatter(self.s2[:, 0], self.s2[:, 1], c=c2, s=1)

        # automatically determine s
        # transform: data --> points
        p1, p2 = ax.transData.transform([(0, 0), (0, self.a)])
        p12 = p1 - p2
        pt = np.hypot(p12[0], p12[1]) / fig.dpi * 72
        s1 = (pt / 4) ** 2
        s2 = (pt / 5) ** 2


        if direction == 'xy':
            sc1.set_sizes([s1] * len(self.mo), dpi=fig.dpi)
            sc2.set_sizes([s2] * len(self.s1), dpi=fig.dpi)
        else:
            sc1.set_alpha(0)
            sc2.set_alpha(0)
            sc3 = ax.scatter(self.mo[:, ind1], self.mo[:, ind2], c=c1, s=s1)
            sc4 = ax.scatter(self.s1[:, ind1], self.s1[:, ind2], c=c2, s=s2)
            sc5 = ax.scatter(self.s2[:, ind1], self.s2[:, ind2], c=c2, s=s2)

        if add_bonds:
            if direction == 'xy':
                plot_bonds(ax, self.a, self.mo, self.s1, direction=direction, color=c3, lw=(pt/11))
            else:
                plot_bonds(ax, self.a, self.mo, self.s1, direction=direction, color=c3, lw=(pt/11))
                plot_bonds(ax, self.a, self.mo, self.s2, direction=direction, color=c3, lw=(pt/11))
