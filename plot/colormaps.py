import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mc

import os
from ..io.io_image import load_pickle

def color_palette(name, low=0, high=1):
    if mc.is_color_like(name):
        rgb = mc.to_rgb(name)
        hls = colorsys.rgb_to_hls(*rgb)
        palette = np.array([colorsys.hls_to_rgb(*(hls[0], i, hls[2])) for i in np.linspace(low, high, 256)])
        cmap = mc.ListedColormap(palette)
    else:
        cmap = plt.cm.get_cmap(name)
    return cmap


def color_mix(c1, c2, mode='mix', gamma=None):
    assert (mode in ("mix", "blend"))

    if mc.is_color_like(c1) and mc.is_color_like(c2):
        rgba1 = np.asarray(mc.to_rgba(c1))
        rgba2 = np.asarray(mc.to_rgba(c2))

        ts = np.linspace(0, 1, 256)

        if mode == "mix" and gamma in (1., None):
            rgba = np.array([(1 - t) * rgba1 + t * rgba2 for t in ts])
        elif mode == "mix" and gamma > 0:
            rgb = np.array([np.power((1 - t) * rgba1[:3] ** gamma + t * rgba2[:3] ** gamma, 1 / gamma) for t in ts])
            a = np.array([(1 - t) * rgba1[-1] + t * rgba2[-1] for t in ts])
            rgba = np.column_stack([rgb, a])
        elif mode == "blend":
            a = np.array([1 - (1 - (1 - t) * rgba1[-1]) * (1 - rgba2[-1]) for t in ts])
            s = np.array([(1 - (1 - t) * rgba1[-1]) * rgba2[-1] for t in ts]) / a
            if gamma in (1., None):
                rgb = np.array([(1 - t) * rgba1[:3] + t * rgba2[:3] for t in s])
            elif gamma > 0:
                rgb = np.array([np.power((1 - t) * rgba1[:3] ** gamma + t * rgba2[:3] ** gamma, 1 / gamma) for t in s])
            rgba = np.column_stack([rgb, a])
        cmap = mc.ListedColormap(rgba)
        return cmap
    else:
        raise ValueError("c1 and c2 must be valid color names.")


def cubichelix_palette(start=0, rot=0.4, gamma=1.0, hue=0.8):
    def get_color_function(p0, p1):
        def color(x):
            xg = x ** gamma
            a = hue * xg * (1 - xg) / 2
            phi = 2 * np.pi * (start / 3 + rot * x)
            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    cdict = {
        "red": get_color_function(-0.14861, 1.78277),
        "green": get_color_function(-0.29227, -0.90649),
        "blue": get_color_function(1.97294, 0.0),
    }
    cmap = mc.LinearSegmentedColormap(name='cubichelix', segmentdata=cdict)
    return cmap


def show_cmap(cmap):
    x = np.linspace(0, 1, 256)
    rgb = cmap(x)[:, 0:3]
    # convert rgb to hls
    l = np.array([colorsys.rgb_to_hls(*e)[1] for e in rgb])
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2/1.68))
    ax.scatter(x, l, color=rgb, s=50)


def plot_cmap(cmap, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    inds = np.linspace(0, 1, 256)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    rgba = cmap(inds)

    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    col = ['r', 'g', 'b']
    for i in range(3):
        ax.plot(inds, rgba[:, i], color=col[i])

    ax.set_xlabel('index')
    ax.set_ylabel('RGB')



data_dir = os.path.dirname(__file__) + '/colordata/'
parula = load_pickle(data_dir+'LinearSegmentedColormap_parula.pkl')




