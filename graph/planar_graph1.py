import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import radius_neighbors_graph

from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy.sparse import csr_matrix


# conversion
def matrix2edges(matrix):
    i, j = np.where(matrix == 1)
    ij = np.vstack([i, j]).T
    return ij

def matrix2inds(matrix):
    num_of_nodes = matrix.shape[0]
    ij = matrix2edges(matrix)

    # some issue with the split method, unconnected nodes will cause indexing problem,
    # in which num_of_nodes > split groups
    # we should make num_of_nodes == split groups
    # get split indices from i
    vals_, split_ind = np.unique(ij[:, 0], return_index=True)
    split_ind = split_ind[1:]
    # len(vals_) <= len(vals), we have to check which are missing
    vals = np.arange(num_of_nodes)
    if len(vals_) == len(vals):
        # split j according split_ind
        groups = np.split(ij[:, 1], split_ind)
    else:
        missing_inds = np.where(np.in1d(vals, vals_)==False)[0]
        groups = np.split(ij[:, 1], split_ind)
        for ind in missing_inds:
            groups.insert(ind, np.array([]))
    return groups


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


def wiki_arctan2(y, x):
    angles = np.arctan2(y, x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return angles

def get_wedges(pts, inds, sort=True):
    # number of pts == len(inds)
    # every row in inds stores the neighbors indices
    all_wedges = []
    for i, js in enumerate(inds):
        if len(js) == 0:
            all_wedges.append([i, -1, -1])
        elif len(js) == 1:
            all_wedges.append([js[0], -1, js[0]])
        else:
            # thetas
            thetas = np.array([wiki_arctan2((pts[j] - pts[i])[1], (pts[j] - pts[i])[0]) for j in js])
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
    # within group j, find first occurrence of k
    inds = np.where(group_j[:, 1] == k)[0]
    if inds.size == 0:
        # next wedge cannot be found
        next_wedge = None
        wedge_idx = None
    else:
        idx = np.where(group_j[:, 1] == k)[0][0]
        next_wedge = group_j[idx]
        wedge_idx = start + idx
    return next_wedge, wedge_idx


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

    if start_wedge[1] == -1:
        polygon = None
    else:
        polygon = [e for e in start_wedge]
        start_wedge_copy = start_wedge.copy()
        keep_search = True
        while keep_search:
            # should consider the case where next wedge cannot be found
            next_wedge, wedge_idx = search_next_wedge(wedges, start_wedge, groups_start_end)
            if next_wedge is None:
                # no polygon will be found
                keep_search = False
                # reset polygon to None
                polygon = None
            else:
                # update used
                used[wedge_idx] = 1
                if np.all(next_wedge[-2:] == start_wedge_copy[0:2]):
                    # no need to search anymore
                    keep_search = False
                    # return polygon indices
                    polygon = np.array(polygon[:-1])
                else:
                    keep_search = True
                    # update start_wedge and search
                    start_wedge = next_wedge
                    polygon.append(next_wedge[-1])
    return polygon, used


def locate_next_start(used):
    # return next start from unused
    starts = np.where(used == 0)[0]
    if starts.size == 0:
        return None
    else:
        return starts[0]


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
        if polygon is not None:
            polys.append(polygon)
        # get a new start
        start = locate_next_start(used)
    return polys


def get_rmin_rmax(pts, ij):
    i, j = ij.T
    p = pts[i] - pts[j]
    d = np.hypot(p[0], p[1])
    return d.min(), d.max()

class PlanarGraph:

    def __init__(self, pts, matrix):

        self.pts = pts
        self.matrix = matrix
        self.matrix_sparse = csr_matrix(matrix)
        self.edges = matrix2edges(self.matrix)
        # a list of neighbor indices
        self.inds = matrix2inds(self.matrix)

        self.degrees = self.matrix.sum(axis=0)

        self.rmin, self.max = get_rmin_rmax(self.pts, self.edges)


        # this is important concept -- wedges
        self.wedges = get_wedges(self.pts, self.inds)

        self.num_of_nodes = len(pts)
        self.num_of_vertices = len(pts)
        self.num_of_edges = len(self.edges)
        self.num_of_wedges = len(self.wedges)

    # polygons are formed by grouping wedges
    @property
    def polys(self):
        polys = group_wedges(self.wedges)

        # remove polygons
        polys = np.array([poly for poly in polys if len(poly)<=10], dtype=object)
        return polys

    @property
    def polys_(self):
        kmax = self.k.max()
        return np.array([poly.tolist() + [-1] * (kmax - len(poly)) for poly in self.polys])

    @property
    def k(self):
        return np.array([len(polygon) for polygon in self.polys])

    @property
    def centers(self):
        return np.array([self.pts[poly].mean(axis=0) for poly in self.polys])

    @classmethod
    def construct(cls, pts, rmin, rmax):
        matrix = radius_neighbors_graph(pts, rmax, mode='distance', p=2).toarray()
        matrix[matrix < rmin] = 0
        matrix[matrix > 0] = 1
        return cls(pts, matrix)


    def to_image(self):
        pass

    def to_polygon_graph(self):
        polys = self.polys
        polys_ = self.polys_
        ll = [np.where(np.isin(polys_, poly).sum(axis=1) == 2)[0] for poly in polys]
        ijs = [(i, j) for i, l in enumerate(ll) for j in l]
        shape = (len(polys), len(polys))
        matrix = np.zeros(shape)
        for (i,j) in ijs:
            matrix[i, j] = 1
        return PlanarGraph(self.centers, matrix)

    def show_edges(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='C0', s=10)

        lines = np.array([(self.pts[i], self.pts[j]) for (i, j) in self.edges])
        segs = LineCollection(lines, color='#2d3742')
        ax.add_collection(segs)

    def show(self, ax=None):
        mpl_10 = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='r', s=10)
        for inds in self.polys:
            if len(inds) in [3, 4, 5, 6, 7, 8]:
                poly = Polygon(self.pts[inds], ec='#2d3742', fc=mpl_10[len(inds)-3], alpha=0.5)
                ax.add_patch(poly)

    # only a bit faster
    def show_(self, ax=None):
        mpl_10 = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(self.pts[:, 0], self.pts[:, 1], color='r', s=10)

        for e in np.unique(self.k):
            polys_selected = self.pts[np.vstack(self.polys[self.k == e])]
            polys_path = Path.make_compound_path_from_polys(polys_selected)
            polys_patch = PathPatch(polys_path, ec='#2d3742', fc=mpl_10[polys_selected.shape[1]-3], alpha=0.5)
            ax.add_patch(polys_patch)
