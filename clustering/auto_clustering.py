import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.measure import label
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import leaves_list, linkage

from mpl_toolkits.axes_grid1 import make_axes_locatable


def normalize_xy(xy, low= 0, high = 1):
    vmax = xy.max()
    vmin = xy.min()
    xy_norm = (xy - vmin)/(vmax - vmin)*(high - low) + low
    return xy_norm

def seg_lbs(xy, size=256, reorder=True):
    xy_norm = normalize_xy(xy, -size * 0.9 // 2, size * 0.9 // 2) + size // 2
    aa = np.zeros((size, size))
    s = 3
    for (y, x) in xy_norm.astype(np.int):
        aa[y - s:y + s + 1, x - s:x + s + 1] += disk(s)

    mask = 1 * (aa > 0)
    seg = label(mask)
    lbs = np.array([seg[y, x] for (y, x) in xy_norm.astype(np.int)])
    if reorder:
        unique, counts = np.unique(lbs, return_counts=True)
        lbs_order = np.argsort(counts)[::-1] + 1
        order_dict = dict(zip(lbs_order, unique))
        lbs = np.vectorize(order_dict.get)(lbs)
    return lbs - 1

def kmeans_lbs(X, n=None, random_state=0):
    model = KMeans(n_clusters=n, random_state=random_state).fit(X)
    lbs = model.labels_
    # remap lbs according to cluster centroids
    #idx = np.argsort(model.cluster_centers_.sum(axis=1))
    #lut = np.zeros_like(idx)
    #lut[idx] = np.arange(n)
    #order_dict = dict(zip(np.unique(lbs), lut))
    #lbs = np.vectorize(order_dict.get)(lbs)

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

def hr_lbs(X, n, k=10, linkage='ward', metric='correlation'):
    connectivity = kneighbors_graph(X, n_neighbors=k, include_self=False, metric=metric)

    model = AgglomerativeClustering(linkage=linkage, n_clusters=n, connectivity=connectivity).fit(X)
    lbs = model.labels_

    unique, counts = np.unique(lbs, return_counts=True)
    lbs_order = np.argsort(counts)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)

    return lbs


def get_graph_from_lbs(lbs):
    s = len(lbs)
    a = np.zeros(shape=(s, s))
    for e in np.unique(lbs):
        mask = (lbs == e)[:, np.newaxis]
        a += mask.dot(mask.T)
    return a

def get_consensus_matrix(klbs):
    a = np.array([get_graph_from_lbs(lbs) for lbs in klbs])
    return a.mean(axis=0)


def estimate_k(X, kmin=2, kmax=10, method='ward', verbose=True):
    if verbose:
        print('do kmeans from k={} to k={}'.format(kmin, kmax), end=':')
    klbs = []
    for k in range(kmin, kmax + 1):
        lbs = kmeans_lbs(X, k)
        klbs.append(lbs)
        if verbose:
            print(k, end=',')
    klbs = np.array(klbs)
    consensus_matrix = get_consensus_matrix(klbs)

    linkage_matrix = linkage(consensus_matrix, method=method)
    ind = leaves_list(linkage_matrix)
    consensus_matrix = consensus_matrix[ind, :]
    consensus_matrix = consensus_matrix[:, ind]
    return consensus_matrix, linkage_matrix, ind

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# heatmap class
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def reorder_heatmap(D, ind=None):
    if ind is None:
        Z = linkage(D, method='ward')
        ind = leaves_list(Z)
    D = D[ind, :]
    D = D[:, ind]
    return D, ind


class Heatmap:

    def __init__(self, data, ind=None):
        self.data_ = data
        self.data, self.ind = reorder_heatmap(self.data_)

    def plot(self, ax=None, grid=None, grid_kw={}, ticks=True, texts=False, reorder=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        if reorder:
            data = self.data
        else:
            data = self.data_
        im = ax.imshow(data, **kwargs)

        if ticks:
            ax.set_xticks(np.arange(self.data.shape[1]))
            ax.set_yticks(np.arange(self.data.shape[0]))
        else:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

        if texts:
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    # here this is data
                    data = np.round(data, 2)
                    text = ax.text(j, i, data[i, j], ha="center", va="center", color="k", fontsize=8)
        else:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

        if grid is None:
            if self.data.shape[0] > 15:
                grid = False
            else:
                grid = True
        else:
            grid = grid
        if grid:
            ax.set_xticks(np.arange(self.data.shape[1] + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(self.data.shape[0] + 1) - 0.5, minor=True)
            if not grid_kw:
                grid_kw = dict(color='white', lw=0.5)
            ax.grid(which='minor', axis='both', **grid_kw)
            ax.tick_params(which="both", bottom=False, left=False)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        return im