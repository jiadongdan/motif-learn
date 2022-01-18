import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.stats import mode
from scipy.spatial import Voronoi, Delaunay



def get_vnn_point_i(i, ridge_points, ridges_lengths, threshold):
    # len(ridges_lengths) == len(ridge_points)

    # idx is the pairs positions
    idx = np.where((ridge_points[:, 0] == i) | (ridge_points[:, 1] == i))[0]
    # get all voronoi neighbors of point i
    # each row in pairs must contain i
    pairs = ridge_points[idx, 0:2]
    # make all i to 0 in pairs
    pairs[pairs == i] = 0
    inds = pairs.sum(axis=1).astype(int)
    # calculate lengths
    l = ridges_lengths[idx]
    s = l / l.sum()
    # filter inds according threshold
    return inds[s>threshold]



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


def plot_vor_edges(ax, vor, **kwargs):
    if 'color' not in kwargs:
        kwargs['color'] = '#2d3742'
    if 'lw' not in kwargs:
        kwargs['lw'] = 1

    # edges points
    pts = vor.vertices
    # indices of the Voronoi vertices forming each Voronoi ridge
    inds = vor.ridge_vertices

    segs = np.array([(pts[row[0]], pts[row[1]]) for row in inds if -1 not in row])
    lc = LineCollection(segs, **kwargs)
    ax.add_collection(lc)


def plot_vor_polygons(ax, vor, k, **kwargs):
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5
    if 'fc' not in kwargs:
        kwargs['fc'] = 'C0'

    regions = [vor.regions[vor.point_region[i]] for i in range(len(vor.point_region))]
    for region in regions:
        if -1 not in region and len(region) == k:
            polygon = vor.vertices[region]
            ax.fill(*zip(*polygon), **kwargs)


class VoronoiNeighbors:

    def __init__(self, pts, threshold=0.05, alpha=None):
        # input points
        self.pts = pts
        # input points add 4 corners
        self.pts_ = add_corner_points(self.pts)

        # bounds related
        self.w = np.ptp(self.pts_[:, 0])
        self.h = np.ptp(self.pts_[:, 1])
        self.xmin = self.pts_[:, 0].min()
        self.ymin = self.pts_[:, 1].min()


        # boundary indices from alpha shape
        self.boundary = alphashape_edges(pts, alpha)

        # Voronoi model, generated from pts_, NOT pts
        self.vor = Voronoi(self.pts_)

        # ridges: number of ridges = len(vor.ridge_vertices)
        ridge_vertices_pairs = np.array(self.vor.ridge_vertices)
        ridge_points_pairs = np.array(self.vor.ridge_points)
        vertices = self.vor.vertices

        # calculate all ridge lengths
        p12 = vertices[ridge_vertices_pairs[:, 0]] - vertices[ridge_vertices_pairs[:, 1]]
        L = np.hypot(p12[:, 0], p12[:, 1])

        # L = np.hypot(p12[:, 0], p12[:, 1])[:, np.newaxis]
        # pairs = np.hstack([ridge_points_pairs, ridge_vertices_pairs, L])

        self.inds = []
        for ii in range(len(self.pts)):
            idx = locate_vnn_idx(ridge_points_pairs, ii)
            # remove neighbors with a small edge length
            aa = ridge_points_pairs[idx, 0:2]
            aa[aa == ii] = 0
            bb = aa.sum(axis=1).astype(int)
            if ii not in self.boundary:
                l = L[idx]
                s = l / l.sum()
                self.inds.append(bb[s > threshold])
            else:
                self.inds.append(bb[bb<len(self.pts)])

        # sort by angle
        self.inds = sort_inds(self.inds, self.pts_)

        # distances related
        ds = []
        for i , js in enumerate(self.inds):
            ds.append(np.array([np.hypot((self.pts[i]-self.pts[j])[0], (self.pts[i]-self.pts[j])[1]) for j in js]))

        ds_ = np.hstack(ds)
        d0 = np.sqrt(2)*ds_.mean() - 4
        print(d0)
        self.ds = [d[d<d0] for d in ds]

        self.inds = [ind[d < d0] for (d, ind) in zip(ds, self.inds)]

        self.ks = np.array([len(e) for e in self.inds])
        self.k = mode(self.ks).mode[0]



    def show(self, ax=None, fill=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # plot edges
        plot_vor_edges(ax, self.vor)
        ax.set_xlim(self.xmin, self.xmin + self.w)
        ax.set_ylim(self.ymin, self.ymin + self.h)

        # plot major polygons
        if fill:
            plot_vor_polygons(ax, self.vor, self.k)

    def show1(self, ax=None):
        mpl_10 = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # plot edges
        plot_vor_edges(ax, self.vor)
        ax.set_xlim(self.xmin, self.xmin + self.w)
        ax.set_ylim(self.ymin, self.ymin + self.h)

        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='r', s=10)
        regions = [self.vor.regions[self.vor.point_region[i]] for i in range(len(self.vor.point_region))]
        for ii, region in enumerate(regions):
            if -1 not in region:
                polygon = self.vor.vertices[region]
                ax.fill(*zip(*polygon), alpha=0.5, fc=mpl_10[self.ks[ii] - 4])