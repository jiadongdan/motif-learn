import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.morphology import disk, dilation
from skimage.measure import label


def kmeans_lbs(X, n=None, random_state=0):
    model = KMeans(n_clusters=n, random_state=random_state).fit(X)
    lbs = model.labels_
    # remap lbs according to cluster centroids
    # idx = np.argsort(model.cluster_centers_.sum(axis=1))
    # lut = np.zeros_like(idx)
    # lut[idx] = np.arange(n)
    # order_dict = dict(zip(np.unique(lbs), lut))
    # lbs = np.vectorize(order_dict.get)(lbs)
    # reorder lbs
    unique, counts = np.unique(lbs, return_counts=True)
    lbs_order = np.argsort(counts)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)
    return lbs


def gmm_lbs(X, n, type='full', ramdom_state=0):
    model =  GaussianMixture(n, covariance_type=type, random_state=ramdom_state).fit(X)
    lbs = model.predict(X)
    # reorder lbs
    unique, counts = np.unique(lbs, return_counts=True)
    lbs_order = np.argsort(counts)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)
    return lbs


def sort_lbs(lbs):
    unique_lbs, counts = np.unique(lbs, return_counts=True)
    idx = np.argsort(counts)[::-1]
    unique_lbs = unique_lbs[idx]
    lbs_order = range(len(unique_lbs))
    order_dict = dict(zip(unique_lbs, lbs_order))
    lbs_ = np.vectorize(order_dict.get)(lbs)
    return lbs_


def normalize_xy(xy, low= 0, high = 1):
    vmax = xy.max()
    vmin = xy.min()
    xy_norm = (xy - vmin)/(vmax - vmin)*(high - low) + low
    return xy_norm

def seg_lbs(xy, size=256, t=0.01):
    xy_norm = normalize_xy(xy, -size * 0.9 // 2, size * 0.9 // 2) + size // 2
    xy_ = np.round(xy_norm).astype(int)
    x, y = xy_.T
    aa = np.zeros((size, size))
    s = 3
    aa[y, x] = 1
    aa =  dilation(aa, disk(s))
    lbs_img = label(aa)
    lbs = lbs_img[y, x]
    lbs = lbs - lbs.min()
    lbs = sort_lbs(lbs)
    return lbs
