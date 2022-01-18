import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix


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


class VoronoiNeighbors:

    def __init__(self, pts, alpha=None):
        # input points
        self.pts = pts
        # input points add 4 corners
        self.pts_ = add_corner_points(self.pts)

        # Voronoi model, generated from pts_, NOT pts
        self.vor = Voronoi(self.pts_)

    def vnn(self, query=None, return_distance=False):
        pass

    def construct_graph(self, threshold=0.15, dmax=25, return_sparse=False):
        # ridges: number of ridges = len(vor.ridge_vertices)
        ridge_vertices = np.array(self.vor.ridge_vertices)
        ridge_points = np.array(self.vor.ridge_points)
        vertices = self.vor.vertices

        # duplicate
        ridge_points = np.vstack([ridge_points, np.fliplr(ridge_points)])
        ridge_vertices = np.vstack([ridge_vertices, np.fliplr(ridge_vertices)])

        # sort
        sort_inds = np.argsort(ridge_points[:, 0])
        ridge_points = ridge_points[sort_inds]
        ridge_vertices = ridge_vertices[sort_inds]
        # ridge lengths
        p12 = vertices[ridge_vertices[:, 0]] - vertices[ridge_vertices[:, 1]]
        L = np.hypot(p12[:, 0], p12[:, 1])
        # split
        split_ind = np.unique(ridge_points[:, 0], return_index=True)[1][1:]
        groups = np.split(L, split_ind)[0:-4]
        L_groups = [group / group.sum() for group in groups]
        ridge_points_groups = np.split(ridge_points, split_ind, axis=0)[0:-4]

        ijs = [g1[g2 > threshold] for g1, g2, in zip(ridge_points_groups, L_groups)]
        ijs = [ij[ij[:, 1] < len(self.pts)] for ij in ijs]
        ijs = np.vstack(ijs)
        d = np.array([np.hypot((self.pts[i] - self.pts[j])[0], (self.pts[i] - self.pts[j])[1]) for (i, j) in ijs])
        ijs = ijs[d < dmax]

        # construct dense matrix for graph
        shape = (len(self.pts), len(self.pts))
        matrix = np.zeros(shape)
        matrix[ijs[:, 0], ijs[:, 1]] = 1
        # make it symmetric
        matrix = np.maximum(matrix, matrix.transpose())

        if return_sparse:
            return csr_matrix(matrix)
        else:
            return matrix

    def show(self, ax=None, threshold=0.1, dmax=25, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='C0', s=10)

        # matrix is symmetric
        matrix = self.construct_graph(threshold=threshold, dmax=dmax)
        ijs = np.vstack(np.nonzero(matrix)).T

        lines = np.array([(self.pts[i], self.pts[j]) for (i, j) in ijs])
        segs = LineCollection(lines, color='#2d3742')
        ax.add_collection(segs)
