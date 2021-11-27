from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode


def estimate_radius(pts, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pts)
    r = d[:, 1:].mean()
    return r


class RNeighbors:

    def __init__(self, pts, radius=None):

        self.pts = pts

        if radius is None:
            self.radius = estimate_radius(pts)
        else:
            self.radius = radius

        nbrs = NearestNeighbors(radius=self.radius, algorithm='auto').fit(pts)
        self.inds = nbrs.radius_neighbors(self.pts, self.radius, return_distance=False)

        self.ks = np.array([len(e) for e in self.inds])
        self.k = mode(self.ks).mode[0]