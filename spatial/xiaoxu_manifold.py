import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as mc
import colorsys


from ..clustering.auto_clustering import kmeans_lbs


def get_three_phases_inds(xy):
    lbs = kmeans_lbs(xy, 3)
    d = np.hypot(xy[:, 0], xy[:, 1])
    inds = []
    for e in np.unique(lbs):
        idx = np.argmax(d[lbs==e])
        ind = np.where(lbs==e)[0][idx]
        inds.append(ind)
    return np.array(inds)

def _color_mix(c1, c2, t):
    if mc.is_color_like(c1) and mc.is_color_like(c2):
        rgba1 = np.asarray(mc.to_rgba(c1))
        rgba2 = np.asarray(mc.to_rgba(c2))
    return (1 - t) * rgba1 + t * rgba2

def _change_lightness(name, f):
    if mc.is_color_like(name):
        rgb = mc.to_rgb(name)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        hls = (h, f, s)
        rgb = colorsys.hls_to_rgb(*hls)
        return mc.to_hex(rgb)
    else:
        raise ValueError("name must be a valid color name.")


def get_three_phases_colors_(xy, colors=None):
    inds = get_three_phases_inds(xy)
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi
    dd = np.hypot(xy[:, 0], xy[:, 1])
    inds = inds[[np.argsort(angles[inds])][0]]
    a1, a2, a3 = angles[inds]
    dmax = dd[inds].max()
    dmin = 0.2
    dd = (dd-dd.min())/(dmax-dmin)+dmin
    cs = []
    if colors is None:
        colors = ['C0', 'C1', 'C2']
    c1, c2, c3 = colors
    for a, d in zip(angles, dd):
        if a >= a1 and a <= a2:
            f = (a - a1) / (a2 - a1)
            c = _color_mix(c1, c2, f)
        elif a >= a2 and a <= a3:
            f = (a - a2) / (a3 - a2)
            c = _color_mix(c2, c3, f)
        elif a > a3:
            f = (a - a3) / (np.pi * 2 - a3 + a1)
            c = _color_mix(c3, c1, f)
        elif a < a1:
            f = (2 * np.pi - a3 + a) / (np.pi * 2 - a3 + a1)
            c = _color_mix(c3, c1, f)
        c = _change_lightness(c, d)
        cs.append(c)
    return np.array(cs)


def get_three_phases_motifs(ps, inds, X, k=None):
    if k is None:
        return ps[inds]
    else:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
        idx = nbrs.kneighbors(X[inds], return_distance=False)
        return [ps[row].mean(axis=0) for row in idx]


def get_three_phases_colors(xy, return_rgb=True):
    angles = (np.arctan2(xy[:, 1], xy[:, 0]) + np.pi) / np.pi / 2
    r = np.hypot(xy[:, 0], xy[:, 1])
    r = r / r.max()
    hsv = np.vstack([angles, r, np.ones_like(r) * 0.9]).T
    if return_rgb:
        c = hsv_to_rgb(hsv)
    else:
        return hsv


def get_grid_motifs(ps, xy, tn=32, rmin=0.7, rmax=1.0):
    xy = xy / np.abs(xy).max()
    angles = np.arctan2(xy[:, 1], xy[:, 0])+np.pi
    rs = np.hypot(xy[:, 0], xy[:, 1])
    ms = []
    for i in np.arange(tn):
        m1 = angles >= i * np.pi * 2 / (tn)
        m2 = angles <= (i+1) * np.pi * 2 / (tn)
        m12 = np.vstack([m1, m2]).T.all(axis=1)
        m3 = rs >= (rs[m12].max()*rmin)
        m4 = rs <= (rs[m12].max()*rmax)
        m = np.vstack([m1, m2, m3, m4]).T.all(axis=1)
        motif = ps[m].mean(axis=0)
        ms.append(motif)
    return ms

def imshow_(img, **kwargs):
    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_axes([0, 0, 1, 1], fc=[0, 0, 0, 0])
    ax.imshow(img, **kwargs)
