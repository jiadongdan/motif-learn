import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from skimage.filters import threshold_otsu, threshold_li

from .utils import make_symmetric_more, make_symmetric, matrix2ijs


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Estimate radius
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def estimate_d(pts, threshold='otsu', return_k=False):
    l = []
    ts = []
    ks = range(2, 13)
    nbrs = NearestNeighbors(n_neighbors=ks[-1], algorithm='ball_tree').fit(pts)
    dd, ind = nbrs.kneighbors(pts)

    for k in ks:
        d = dd[:, 1:k].ravel()
        if threshold =='otsu':
            t = threshold_otsu(d)
        elif threshold == 'li':
            t = threshold_li(d)
        else:
            t = threshold_li(d)
        s1 = len(np.nonzero(d>t)[0])/len(d)
        s2 = len(np.nonzero(d<=t)[0])/len(d)
        l.append(s1*s2)
        ts.append(t)
    if return_k:
        return ts[np.argmax(l)], ks[np.argmax(l)]
    else:
        return ts[np.argmax(l)]


def vnn_distance(pts, return_all=False):
    vor = Voronoi(pts)
    # ridges: number of ridges = len(vor.ridge_vertices)
    ridge_vertices = np.array(vor.ridge_vertices)
    ridge_points = np.array(vor.ridge_points)
    vertices = vor.vertices
    points = vor.points

    # edge lengths
    p12_ = points[ridge_points[:, 0]] - points[ridge_points[:, 1]]
    L1 = np.hypot(p12_[:, 0], p12_[:, 1])
    if return_all:
        return L1
    else:
        return np.median(L1)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Construct_edges
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

def vnn_graph(pts, threshold=0.1, dmax=None, threshold_method=None, return_ijs=True):
    if dmax is None:
        dmax = estimate_d(pts, threshold=threshold_method)

    pts_ = add_corner_points(pts)
    vor = Voronoi(pts_)
    # ridges: number of ridges = len(vor.ridge_vertices)
    ridge_vertices = np.array(vor.ridge_vertices)
    ridge_points = np.array(vor.ridge_points)
    vertices = vor.vertices
    points = vor.points

    # ridge lengths
    p12 = vertices[ridge_vertices[:, 0]] - vertices[ridge_vertices[:, 1]]
    L = np.hypot(p12[:, 0], p12[:, 1])

    # edge lengths
    p12_ = points[ridge_points[:, 0]] - points[ridge_points[:, 1]]
    L1 = np.hypot(p12_[:, 0], p12_[:, 1])

    ijL = np.hstack([ridge_points, L.reshape(-1, 1)])
    #mask = np.logical_and(L < 15, L1 < dmax)
    mask = L1 < dmax
    # remove too large L
    ijL = ijL[mask]

    s = len(points)
    rows, cols, data = ijL.T
    matrix = coo_matrix((data, (rows, cols)), shape=(s, s))
    # make matrix symmetric, keep values
    matrix = make_symmetric(matrix)

    matrix_ = normalize(matrix, norm='l1')
    matrix_ = (matrix_ >=threshold)*1

    # matrix_ is likely not symmetric now
    matrix_ = make_symmetric_more(matrix_)

    # better to return a sparse matrix as the graph can be very large
    # otherwise it takes too much memory
    matrix_ = matrix_[0:-4, :][:, 0:-4]

    if return_ijs:
        return matrix2ijs(matrix_)
    else:
        return matrix_
