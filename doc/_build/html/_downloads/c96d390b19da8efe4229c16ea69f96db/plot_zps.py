# -*- coding: utf-8 -*-
"""
Plotting Zernike Polynomials
============================

plot Zernike Polynomials
"""


import numpy as np
import matplotlib.pyplot as plt
from stempy.feature import ZPs

def generate_h_axes(fig, n=2, y=0.1, left=1 / 15, right=1 / 30, wspace=0.25, ratios=None):
    if ratios == None:
        ratios = np.ones(n)
    elif np.isscalar(ratios):
        ratios = np.array([ratios]*n)
    elif np.iterable(ratios):
        ratios = np.array(ratios)
    width, height = fig.get_size_inches()
    ratio = width / height

    N = ratios.sum() + ratios.sum() / n * wspace * (n - 1)
    dx = (1 - left - right) / N
    ws = dx * ratios.sum() / n * wspace
    W = np.array([e * dx for e in ratios])
    h = W.min()
    axes = [fig.add_axes([left + W[0:i].sum() + i * ws, y, W[i], h * ratio]) for i in range(n)]
    return axes



def axes_from_ax(ax, nrows, ncols, wspace=0.1, hspace=0.1):
    fig = ax.figure
    axes = fig.subplots(nrows, ncols)
    left, bottom, right, top = ax.get_position().extents
    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)
    axes_ = [fig.add_axes(ax_.get_position().bounds) for ax_ in axes.ravel()]
    for ax_ in axes.ravel():
        ax_.remove()
    ax.remove()
    return axes_

def get_axes(axes, dy=0.03):
    axes_ = []
    for i in range(len(axes)-1):
        ax1, ax2 = axes[i], axes[i+1]
        x = (ax1.get_position().x0 + ax2.get_position().x0)/2
        h = ax1.get_position().height
        y = ax1.get_position().y0 + h + dy
        w = ax1.get_position().width
        axes_.append(fig.add_axes([x, y, w, h]))
    return axes_

def plot_image(ax, img, clip=False, **kwargs):
    h, w = img.shape[0:2]
    im = ax.imshow(img, **kwargs)
    ax.axis('off')
    if clip == True:
        patch = plt.Circle((w / 2 - 0.5, h / 2 - 0.5), radius=(h + w) / 4 - 2, transform=ax.transData)
        im.set_clip_path(patch)
        
def zp_j2nm(j):
    if not np.isscalar(j):
        j = np.array(j)
    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(np.int)
    m = 2 * j - n * (n + 2)
    return np.array([n, m]).T

fig = plt.figure(figsize=(3.6, 3.6), facecolor='#fcfcfc')
axes5 = generate_h_axes(fig, n=5, y=0.1, left=1/15, right=1/30, wspace=0.25, ratios=1)

axes4 = get_axes(axes5)
axes3 = get_axes(axes4)
axes2 = get_axes(axes3)
axes1 = get_axes(axes2)

axes = axes1+axes2+axes3+axes4+axes5

zps = ZPs(n_max=5, size=256)
data = zps.data
vmin, vmax = data.min(), data.max()

# https://stackoverflow.com/a/33286367/5855131
ss = [r'$Z_{n}^{{{m}}}$'.format(n=n, m=m) for (n, m) in zp_j2nm(range(21))]

for i, (e, ax) in enumerate(zip(data, axes)):
    plot_image(ax, e, clip=True, vmin=vmin, vmax=vmax)
    ax.text(x=0, y=1, s=ss[i], fontsize=8)