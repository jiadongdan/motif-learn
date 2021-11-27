import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# need more tweaking here
def estimate_alpha(pts):
    tri = Delaunay(pts)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    rs = []
    for ia, ib, ic in tri.simplices:
        pa = pts[ia]
        pb = pts[ib]
        pc = pts[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        rs.append(circum_r)
    return np.array(rs).mean()


# see: https://stackoverflow.com/a/50159452/5855131
def alphashape(pts, alpha=None, only_outer=True):
    if alpha is None:
        alpha = estimate_alpha(pts)

    pts = np.array(pts)

    assert pts.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(pts)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = pts[ia]
        pb = pts[ib]
        pc = pts[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)

    # convert edges to numpy array
    edges = np.array([(i, j) for (i, j) in edges])
    return edges

def alphashape_edges(pts, alpha):
    edges = alphashape(pts, alpha, only_outer=True)
    return np.unique(edges)

class AlphaShape:

    def __init__(self, pts, alpha=None):
        self.pts = pts

        if alpha is None:
            self.alpha = estimate_alpha(pts)
        else:
            self.alpha = alpha

        self.edges = alphashape(self.pts, self.alpha)

        self.inds = np.unique(self.edges)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        ax.scatter(self.pts[:, 0], self.pts[:, 1], c='C0', **kwargs)
        ax.scatter(self.pts[self.inds][:, 0], self.pts[self.inds][:, 1], c='C1', **kwargs)

