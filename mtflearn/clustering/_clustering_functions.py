import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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