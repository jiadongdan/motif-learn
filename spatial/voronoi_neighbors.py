import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial import Voronoi, Delaunay

from .alpha_shape import alphashape_edges


# add 4 corner points
def add_corner_points(pts, pad=0.05):
    center = pts.mean(axis=0)
    pts_ = pts - center
    vmax = np.abs(pts_).max() * (1 + pad)
    p1 = (-vmax, -vmax)
    p2 = (+vmax, -vmax)
    p3 = (+vmax, +vmax)
    p4 = (-vmax, +vmax)
    p1234 = np.array([p1, p2, p3, p4])
    pts_new = np.vstack([pts, p1234 + center])
    return pts_new


def locate_vnn_idx(ridge_points_pairs, ii):
    return np.where((ridge_points_pairs[:, 0] == ii) | (ridge_points_pairs[:, 1] == ii))[0]

def sort_inds(inds, pts):
    inds_ = []
    for i, row in enumerate(inds):
        if row[0] != -1:
            pp = pts[row] - pts[i]
            angles = np.arctan2(pp[:, 1], pp[:, 0]) + np.pi
            row_ = row[np.argsort(angles)]
            inds_.append(row_)
        else:
            inds_.append([-1])
    return inds_

class VorNeighbors:

    def __init__(self, pts, lbs=None, threshold=0.05, alpha=None):
        # input points
        self.pts = pts
        # input points add 4 corners
        self.pts_ = add_corner_points(self.pts)


        # lbs
        if lbs is None:
            self.lbs = np.ones(len(self.pts)) * (-1)
        else:
            self.lbs = lbs

        self.lbs_ = np.hstack([self.lbs, [-1] * 4])


        # boundary indices from alpha shape
        self.boundary = alphashape_edges(pts, alpha)


        # Voronoi model, generated from pts_, NOT pts
        vor = Voronoi(self.pts_)

        # ridges: number of ridges = len(vor.ridge_vertices)
        ridge_vertices_pairs = np.array(vor.ridge_vertices)
        ridge_points_pairs = np.array(vor.ridge_points)

        # calculate all ridge lengths
        p12 = vor.vertices[ridge_vertices_pairs[:, 0]] - vor.vertices[ridge_vertices_pairs[:, 1]]
        L = np.hypot(p12[:, 0], p12[:, 1])

        # L = np.hypot(p12[:, 0], p12[:, 1])[:, np.newaxis]
        # pairs = np.hstack([ridge_points_pairs, ridge_vertices_pairs, L])

        self.inds = []
        for ii in range(len(self.pts)):
            if ii not in self.boundary:
                idx = locate_vnn_idx(ridge_points_pairs, ii)
                # remove neighbors with a small edge
                aa = ridge_points_pairs[idx, 0:2]
                aa[aa == ii] = 0
                bb = aa.sum(axis=1).astype(int)
                l = L[idx]
                s = l / l.sum()
                self.inds.append(bb[s > threshold])
            else:
                self.inds.append([-1])

        # sort by angle
        self.inds = sort_inds(self.inds, self.pts_)

        self.ks = np.array([len(e) for e in self.inds])
        self.k = mode(self.ks).mode[0]

    def get_mols(self, t=0.01):
        mols = []

        k_max = np.max([len(row) for row in self.inds]) + 1
        for i, row in enumerate(self.inds):
            a = np.array([-1] * k_max)
            if row[0] == -1:
                mols.append(a)
            else:
                ind = [i] + [e for e in row]
                mol = self.lbs[ind]
                a[0:len(row) + 1] = mol
                mols.append(a)

        mols = np.array(mols)

        mols_, cnts = np.unique(mols, axis=0, return_counts=True)
        mols_ = mols_[np.argsort(cnts)[::-1]]
        cnts_ = cnts[np.argsort(cnts)[::-1]]
        # remove boundary
        idx = np.where(np.all(mols_ == [-1] * mols_.shape[1], axis=1))[0][0]
        mols_ = np.delete(mols_, idx, axis=0)
        cnts_ = np.delete(cnts_, idx)
        mask = cnts_/(cnts_.sum()) > t
        return mols_[mask], cnts_[mask]

    def get_all_mols(self):
        mols = []

        k_max = np.max([len(row) for row in self.inds]) + 1
        for i, row in enumerate(self.inds):
            a = np.array([-1] * k_max)
            if row[0] == -1:
                mols.append(a)
            else:
                ind = [i] + [e for e in row]
                mol = self.lbs[ind]
                a[0:len(row) + 1] = mol
                mols.append(a)

        mols = np.array(mols)
        return mols

    def show_k(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        ks = self.ks.copy()
        ks[ks == 1] = 100
        ksmin = ks.min()
        ks = ks - (ksmin - 1)
        ks[ks > 100 - ksmin - 1] = 0
        cs = np.array(['#2d3742'] + ['C{}'.format(i) for i in range(9)])
        ax.scatter(self.pts[:, 0], self.pts[:, 1], color=cs[ks], **kwargs)







# use Delaunay to get vnn, but difficult to get L
def get_voronoi_nbrs(pts):
    tri = Delaunay(pts)
    indptr, inds = tri.vertex_neighbor_vertices
    vnn = []
    # indptr has a shape (npoints + 1,)
    for i in range(len(indptr) - 1):
        vnn.append(list(inds[indptr[i]:indptr[i + 1]]))
    return vnn