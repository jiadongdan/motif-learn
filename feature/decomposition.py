import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D


def ica(X, n_components=2):
    ica_model = FastICA(n_components=n_components)
    X_ica = ica_model.fit_transform(X)
    return X_ica

def pca(X, n_components=2, reconstruct=False):
    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)
    return xarray(X_pca)

def nmf(X, n_components=2):
    nmf_model = NMF(n_components=n_components)
    X_nmf = nmf_model.fit_transform(X)
    return X_nmf

def fa(X, n_components=2):
    fa_model = FactorAnalysis(n_components=n_components, noise_variance_init=None)
    X_fa = fa_model.fit_transform(X)
    return X_fa

def decomposition_ica(X, n_components=2):
    ica_model = FastICA(n_components=n_components)
    X_ica = ica_model.fit_transform(X)
    return X_ica, ica_model.components_


def decomposition_nmf(X, n_components=2):
    nmf_model = NMF(n_components=n_components)
    X_nmf = nmf_model.fit_transform(X)
    return X_nmf, nmf_model.components_

def decomposition_fa(X, n_components=2):
    fa_model = FactorAnalysis(n_components=n_components, noise_variance_init=None)
    X_fa = fa_model.fit_transform(X)
    return X_fa, fa_model.components_

def decomposition_pca(X, n_components=2):
    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)
    return X_pca, pca_model.components_

def plot_layout(X):
    dim = X.shape[1]
    if dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.plot(X[:, 0], X[:, 1], '.')
        ax.axis('equal')
    elif dim == 3:
        fig = plt.figure(figsize=(7.2, 7.2))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], '.')

def _generate_colors(lbs, color_cycle=None):
    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if lbs is None:
        colors = color_cycle[0]
    else:
        colors= [color_cycle[e] for e in lbs]
    return colors

def plot_pca(X, dim=2, lbs=None, colors=None, **kwargs):
    if len(X.shape) == 3:
        data = X.reshape(X.shape[0], -1)
    elif len(X.shape) == 2:
        data = X
    X_pca = PCA(n_components=dim).fit_transform(data)
    colors = _generate_colors(lbs, colors)
    if 's' or 'size' not in kwargs:
        kwargs['s'] = 1
    if dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], color=colors, s=10, **kwargs)
        ax.axis('equal')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color=colors, **kwargs)


class xarray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    def normalize(self, axis=1, norm='l2'):
        return xarray(normalize(self, axis=axis, norm=norm))

    def plot_pca(self, dim=2, lbs=None, colors=None, **kwargs):
        X_pca = PCA(n_components=dim).fit_transform(self)
        colors = _generate_colors(lbs, colors)
        if dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
            ax.scatter(X_pca[:, 0], X_pca[:, 1], color=colors, s=10, **kwargs)
            ax.axis('equal')
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color=colors, **kwargs)