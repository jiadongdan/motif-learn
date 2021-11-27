import numpy as np
from scipy.ndimage import rotate

from sklearn.utils import check_random_state

def generate_rot_groups(imgs, n=10, angles=None, order=1, reshape=False):
    dim = imgs.ndim
    if angles is None:
        angles = np.linspace(0, 360, n)
    elif not np.iterable(angles):
        angles = np.array([angles])
    ps = np.vstack([rotate(imgs, angle, axes=(dim-2, dim-1), order=order, reshape=reshape) for angle in angles])
    return ps

def get_rand_rows(X, n, seed=0):
    rng = check_random_state(seed=seed)

    if n > 0 and n < 1:
        mask = rng.choice([False, True], len(X), p=[1-n, n])
    elif n >= 1:
        mask = rng.choice(X.shape[0], n, replace=False)
    return X[mask]


from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

def pair_dist(X, metric):
    pass

def knn_graph(X, k, metric='euclidean', mode='connectivity', dense=True):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    G = nbrs.kneighbors_graph(X, mode=mode)
    if dense:
        return G.todense()
    else:
        return G


