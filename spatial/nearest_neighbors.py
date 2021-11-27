import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_pairs(pts, r):
    nbrs = NearestNeighbors(radius=r, algorithm='auto').fit(pts)
    ds, inds = nbrs.radius_neighbors(pts)
    inds = [ind[d > 0] for d, ind in zip(ds, inds)]
    pairs = np.array([(i, e) for i, row in enumerate(inds) for e in row])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0).shape
    return pairs

def get_lines(pts, r):
    nbrs = NearestNeighbors(radius=r, algorithm='auto').fit(pts)
    ds, inds = nbrs.radius_neighbors(pts)
    inds = [ind[d > 0] for d, ind in zip(ds, inds)]
    pairs = np.array([(i, e) for i, row in enumerate(inds) for e in row])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    segs = pts[pairs]
    return segs


def sort_inds_by_angle(pts, inds):
    inds_sort = []
    ns = []
    for ind in inds:
        pts1 = pts[ind]
        center = pts1.mean(axis=0)
        pts2 = pts1 - center
        angles = np.arctan2(pts2[:, 1], pts2[:,  0]) + np.pi
        ind_ = ind[np.argsort(angles)]
        inds_sort.append(ind_)
        ns.append(len(ind))
    if np.unique(ns).shape == (1,):
        return np.array(inds_sort)
    else:
        return inds_sort

