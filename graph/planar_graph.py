import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numpy.lib.stride_tricks import sliding_window_view


# conversion
def matrix2edges(matrix):
    i, j = np.where(matrix == 1)
    ij = np.vstack([i, j]).T
    return ij


def edges2matrix(ij):
    smax = np.max(ij)
    shape = (smax, smax)
    matrix = np.zeros(shape)
    for i, j in ij:
        matrix[i, j] = 1
    return matrix


def matrix2inds(matrix):
    ij = matrix2edges(matrix)
    # get split indices from i
    split_ind = np.unique(ij[:, 0], return_index=True)[1][1:]
    # split j according split_ind
    return np.split(ij[:, 1], split_ind)


def inds2matrix(inds):
    # number of nodes/vertices
    n = len(inds)
    matrix = np.zeros((n, n))
    for i, ind in enumerate(inds):
        for j in ind:
            matrix[i, j] = 1
    return matrix


# get polygons function

def search_next_wedge(wedges, current, groups_start_end=None):
    # guaranteed to success
    if groups_start_end is None:
        # get start and end indices of each group
        split_ind = np.unique(wedges[:, 0], return_index=True)[1]
        split_ind = np.append(split_ind, len(wedges))
        # len(groups_start_end) = len(pts)
        groups_start_end = sliding_window_view(split_ind, 2)
    # return next wedge which connected to current
    i, j, k = current
    # locate group j
    start, end = groups_start_end[j]
    group_j = wedges[start:end]
    # within group j, find first occurrence of k (guaranteed to success)
    idx = np.where(group_j[:, 1] == k)[0][0]
    wedge_idx = start + idx
    return group_j[idx], wedge_idx


def find_polygon(wedges, start_wedge_idx, used=None, groups_start_end=None):
    # return polygon indices AND update used

    if groups_start_end is None:
        # get start and end indices of each group
        split_ind = np.unique(wedges[:, 0], return_index=True)[1]
        split_ind = np.append(split_ind, len(wedges))
        # len(groups_start_end) = len(pts)
        groups_start_end = sliding_window_view(split_ind, 2)

    if used is None:
        used = np.zeros(len(wedges))

    start_wedge = wedges[start_wedge_idx]
    used[start_wedge_idx] = 1

    polygon = [e for e in start_wedge]
    start_wedge_copy = start_wedge.copy()
    keep_search = True
    while keep_search:
        next_wedge, wedge_idx = search_next_wedge(wedges, start_wedge, groups_start_end)
        # update used
        used[wedge_idx] = 1
        if np.all(next_wedge[-2:] == start_wedge_copy[0:2]):
            # no need to search anymore
            keep_search = False
        else:
            keep_search = True
            # update start_wedge and search
            start_wedge = next_wedge
        polygon.append(next_wedge[-1])
    return np.array(polygon[:-1]), used


def locate_next_start(used):
    # return next start from unused
    starts = np.where(used == 0)[0]
    if starts.size == 0:
        return None
    else:
        return starts[0]


def sort_wedges(wedges):
    # sort according first column
    idx = np.argsort(wedges[:, 0])
    wedges = wedges[idx]
    # get split indices
    split_ind = np.unique(wedges[:, 0], return_index=True)[1][1:]
    # split wedges into wedge_groups
    wedge_groups = np.split(wedges, split_ind)
    # sort according second column with each group
    wedge_groups_ = [group[np.argsort(group[:, 1])] for group in wedge_groups]
    return np.vstack(wedge_groups_)


def get_wedges(pts, inds, sort=True):
    # number of pts == len(inds)
    # every row in inds stores the neighbors indices
    all_wedges = []
    for i, js in enumerate(inds):
        # thetas
        thetas = np.array([np.arctan2((pts[j] - pts[i])[1], (pts[j] - pts[i])[0]) for j in js])
        # sort js according to thetas
        js = js[np.argsort(thetas)]
        ijs = np.array([(i, j) for j in js])
        # form wedges from ijs
        wedges = [(j2, i1, j1) for (i1, j1), (i1, j2) in zip(ijs, np.roll(ijs, 1, axis=0))]
        all_wedges.append(wedges)
    all_wedges = np.vstack(all_wedges)
    # sort all_wedges by first and second columns
    if sort:
        all_wedges = sort_wedges(all_wedges)
    return all_wedges


def group_wedges(wedges, start=0):
    # step 1. sort wedges
    wedges = sort_wedges(wedges)
    # step 2. mark all wedges as unused, 0 means unused
    used = np.zeros(len(wedges))

    # step 3
    # get start and end indices of each group
    split_ind = np.unique(wedges[:, 0], return_index=True)[1]
    split_ind = np.append(split_ind, len(wedges))
    # len(groups_start_end) = len(pts)
    groups_start_end = sliding_window_view(split_ind, 2)

    # find polygons
    polys = []
    while start is not None:
        # i marks polygon index
        i = 0
        polygon, used = find_polygon(wedges, start, used, groups_start_end)
        polys.append(polygon)
        # get a new start
        start = locate_next_start(used)
    return polys


class PlanarGraph:

    def __init__(self, pts, matrix):

        self.pts = pts
        self.matrix = matrix
        self.matrix_sparse = None
        self.edges = matrix2edges(self.matrix)
        # a list of neighbor indices
        self.inds = matrix2inds(self.matrix)

        # this is important concept -- wedges
        self.wedges = get_wedges(self.pts, self.inds)

        self.num_of_nodes = len(pts)
        self.num_of_vertices = len(pts)
        self.num_of_edges = len(self.edges)
        self.num_of_wedges = len(self.wedges)

        # polygons are formed by grouping wedges
        polygons = group_wedges(self.wedges)
        # THE polygon with max number of nodes is the boundary
        self.k = np.array([len(polygon) for polygon in polygons])
        idx = np.argmax(self.k)

        self.boundary = polygons[idx]
        del polygons[idx]
        self.polys = polygons

    def show(self, ax=None):
        mpl_10 = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='r', s=10)
        for inds in self.polys:
            poly = Polygon(self.pts[inds], ec='k', fc=mpl_10[len(inds) - self.k.min()], alpha=0.5)
            ax.add_patch(poly)