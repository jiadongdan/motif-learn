import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse import lil_matrix, csr_matrix, issparse, hstack, vstack

from .utils import matrix2inds, edges2matrix


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
            js = np.array(js)[np.argsort(thetas)]
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


def grow_one_edge(pts, ijs):
    i = np.argmin(pts[:, 0])
    p = (pts[i, 0] - 1, pts[i, 1])
    pts_ = np.vstack([pts, p])
    j = pts.shape[0]
    extra_ij = np.array([(i, j), (j, i)])
    ijs_ = np.vstack([ijs, extra_ij])
    return pts_, ijs_

def find_regions(pts, ijs, return_dict=False):
    # grow one edge to avoid a Large polygon from contour points
    pts_, ijs_ = grow_one_edge(pts, ijs)
    shape = (pts_.shape[0], pts_.shape[0])
    matrix_ = edges2matrix(ijs_, shape)
    inds = matrix2inds(matrix_)
    # form wedges
    wedges = get_wedges(pts_, inds)
    # group wedges
    polys = group_wedges(wedges)
    polys = np.array([poly for poly in polys], dtype=object)
    if return_dict:
        ks = np.array([len(e) for e in polys])
        polys = {str(e):np.vstack(polys[ks==e]) for e in np.unique(ks)}
        return polys
    else:
        return polys