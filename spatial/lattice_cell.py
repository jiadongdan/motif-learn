import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from sklearn.neighbors import NearestNeighbors


def _get_p0_u_v(pts):
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pts)
    ind = nbrs.kneighbors(pts.mean(axis=0)[np.newaxis, :], return_distance=False)[0][0]

    p0 = pts[ind]
    inds = nbrs.kneighbors(p0[np.newaxis, :], return_distance=False)[0][1:]
    pp6 = pts[inds] - p0
    angles = np.abs(np.rad2deg(np.arctan2(pp6[:, 1], pp6[:, 0])))
    ind2 = inds[np.argsort(angles)][0:2]
    u = pts[ind2[0]] - p0
    v = pts[ind2[1]] - p0
    return p0, u, v


def points_to_regions(pts, threshold=None):
    if threshold is None:
        threshold = _estimate_r0(pts) * 0.2
    p0, u, v = _get_p0_u_v(pts)
    pts1 = pts + u
    pts2 = pts + v
    pts3 = pts + u + v
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)
    d1, ind1 = nbrs.kneighbors(pts1)
    d2, ind2 = nbrs.kneighbors(pts2)
    d3, ind3 = nbrs.kneighbors(pts3)
    d123 = np.hstack([d1, d2, d3])
    ind1234 = np.hstack([np.arange(len(pts))[:, np.newaxis], ind1, ind3, ind2])
    mask = d123.mean(axis=1) < threshold
    inds = ind1234[mask]
    return inds


# estimate nearest neighbor radius
def _estimate_r0(pts):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pts)
    d = d[:, 1]
    return d.mean()


# propagate single step
def _propagate_pts(pts, pts_, u, v, threshold=None):
    pts1 = pts_ + u
    pts2 = pts_ + v
    pts3 = pts_ + u + v
    pts4 = pts_ - u
    pts5 = pts_ - v
    pts6 = pts_ - u - v
    pts7 = pts_ - u + v
    pts8 = pts_ + u - v
    pp = np.vstack([pts_, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pp)
    pp_ = pts[ind[d < threshold]]
    return np.unique(pp_, axis=0)


def propagate_pts(pts, pts_, u, v, threshold=None):
    if threshold is None:
        r0 = _estimate_r0(pts)
        threshold = 0.2 * r0

    if len(pts_.shape) == 1:
        num_pts = 1
    else:
        num_pts = pts_.shape[0]
    do_propagate = True
    while do_propagate:
        pts_ = _propagate_pts(pts, pts_, u, v, threshold)
        if pts_.shape[0] > num_pts:
            num_pts = pts_.shape[0]
        else:
            do_propagate = False
    return pts_


def get_ordered_lbs(pts, lbs, inds):
    nmax = np.max([len(e) for e in inds])
    lbs_ordered = []
    for ii, row in enumerate(inds):
        if len(row) == 0:
            lbs_select = [-1] * nmax
        elif len(row) == 1:
            lbs_select = [lbs[row[0]]] + [-1] * (nmax - 1)
        else:
            pts_select = pts[row] - (pts[row]).mean(axis=0)
            angles = np.arctan2(pts_select[:, 1], pts_select[:, 0]) + np.pi
            # sort by angle
            lbs_row = lbs[row][np.argsort(angles)]
            # rolling min to first
            idx = np.argmin(lbs_row)
            lbs_select = np.roll(lbs_row, -idx).tolist() + [-1] * (nmax - len(angles))
        lbs_ordered.append(lbs_select)
    return np.array(lbs_ordered)


def reorder_lbs(lbs):
    unique, counts = np.unique(lbs, return_counts=True)
    lbs_order = np.argsort(counts)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)
    return lbs


def get_next_level_lbs():
    pass


class HexCells:

    def __init__(self, pts, dx=0, dy=0, lbs=None):
        dxdy = np.array([dx, dy])
        self.pts = pts + dxdy
        self.p0, self.u, self.v = _get_p0_u_v(self.pts)

        # indices of pts, shape (num_cells, 4), counterclockwise
        self.regions = points_to_regions(self.pts)

        self.lbs = lbs

    # merge small cells, this is a key method
    def merge(self, n=2):
        U = n * self.u
        V = n * self.v
        pts_new = propagate_pts(self.pts, self.p0, U, V, threshold=None)
        if self.lbs is None:
            new_hexcells = HexCells(pts_new)
        else:
            new_hexcells = HexCells(pts_new)
            pos = self.pts[self.regions].mean(axis=1)
            new_hexcells.set_cell_lbs(pos, self.lbs)
        return new_hexcells

    def set_cell_lbs(self, pos, lbs):
        polys = self.pts[self.regions]
        # indices of points inside each cell
        inds = []
        for poly in polys:
            ind = np.where(Path(poly).contains_points(pos) == 1)[0]
            inds.append(ind)
        lbs_ordered = get_ordered_lbs(pos, lbs, inds)
        lbs_ordered_unique, self.lbs = np.unique(lbs_ordered, axis=0, return_counts=False, return_inverse=True)
        self.lbs = reorder_lbs(self.lbs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        if 'ec' not in kwargs:
            kwargs['ec'] = '#2d3742'

        # ax.scatter(self.pts[:, 0], self.pts[:, 1], color='g', s=10)
        cs = ['C{}'.format(e) for e in range(10)]
        for i, e in enumerate(self.regions):
            if self.lbs is None:
                fc = 'C0'
            else:
                fc = cs[self.lbs[i] % 10]
            poly = Polygon(self.pts[e], fc=fc, **kwargs)
            ax.add_patch(poly)
        ax.axis('equal')