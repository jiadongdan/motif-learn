import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import MultiCursor
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Polygon

from sklearn.utils import check_array

from .style import font_small_normal
from .colors import colors_from_lbs



def plot_image(ax, img=None, clip=False, remove_axis=True, **kwargs):
    """
    A convenience function to display image with or without a circle clip

    Parameters
    ----------
    ax:
    img1 : array-like or PIL image
        The image data. Supported array shapes are:

        - (m, n): an image with scalar data. The values are mapped to
          colors using normalization and a colormap. See parameters *norm*,
          *cmap*, *vmin*, *vmax*.
        - (n, m, 3): an image with RGB values (0-1 float or 0-255 int).
        - (n, m, 4): an image with RGBA values (0-1 float or 0-255 int),
          i.e. including transparency.

        The first two dimensions (n, m) define the rows and columns of
        the image.

        Out-of-range RGB(A) values are clipped.

    clip : bool, default: False
    """
    if img is None:
        img = np.random.random((32, 32))
    h, w = img.shape[0:2]
    im = ax.imshow(img, **kwargs)
    if remove_axis:
        ax.axis('off')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_visible(False)

    if clip == True:
        patch = patches.Circle((w / 2 - 0.5, h / 2 - 0.5), radius=(h + w) / 4 - 1., transform=ax.transData)
        im.set_clip_path(patch)


def plot_xy(xy, lbs, ax=None, colors=None, use_alpha=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    if use_alpha:
        c = colors_from_lbs(lbs, colors=colors, xy=xy)
    else:
        c = colors_from_lbs(lbs, colors=colors)
    ax.scatter(xy[:, 0], xy[:, 1], fc=c, ec='none', **kwargs)
    ax.axis('equal')


def add_scale_bar(ax, l, x=0.05, y=0.05, **kwargs):
    # transfrom x and y to data coordinates
    x, y = ax.transData.inverted().transform(ax.transAxes.transform((x, y)))
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = 2
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = '#e4e1ca'
    ax.plot([x, x+l], [y, y], **kwargs)


def add_anchor_text(ax, s, alpha=0.5, loc='upper left', fc=None, fontsize=5, fontcolor='#cee4cc' ,frameon=True):
    at = AnchoredText(s, prop=dict(size=fontsize, color=fontcolor), frameon=frameon, loc=loc, borderpad=0.0,)
    at.patch.set_alpha(alpha)
    if fc is not None:
        at.patch.set_fc(fc)
    at.patch.set_linewidth(0)
    ax.add_artist(at)

def plot_compare(imgs, size=18, cursor=False, **kwargs):
    # sklearn check_array is not recommended here
    # imgs = check_array(imgs, allow_nd=True)
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(size, size/n))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], **kwargs)
        ax.axis('off')
    fig.tight_layout()
    if cursor:
        multi = MultiCursor(fig.canvas, axes, color='r', lw=1, horizOn=True, vertOn=True)
        return multi


# depreciated, remove latter
def plot_pca_layout(ax, xy, colors, style=None, **kwargs):
    if 's' not in kwargs:
        kwargs['s'] = 0.5
    ax.scatter(xy[:, 0], xy[:, 1], fc=colors, ec='none', **kwargs)
    ax.axis('equal')
    ax.set_xlabel('PC1', labelpad=0, fontdict=font_small_normal)
    ax.set_ylabel('PC2', labelpad=0, fontdict=font_small_normal)
    if style is None:
        style = 'plain'
    ax.ticklabel_format(style=style, scilimits=(0,0), useMathText=True)

# depreciated, remove latter
def plot_FR_layout(ax, xy, colors, style=None, **kwargs):
    if 's' not in kwargs:
        kwargs['s'] = 0.5
    ax.scatter(xy[:, 0], xy[:, 1], fc=colors, ec='none', **kwargs)
    ax.axis('equal')
    ax.set_xlabel('FR Component 1', labelpad=0, fontdict=font_small_normal)
    ax.set_ylabel('FR Component 2', labelpad=0, fontdict=font_small_normal)
    if style is None:
        style = 'plain'
    ax.ticklabel_format(style=style, scilimits=(0,0), useMathText=True)


def plot_stacks(ax, colors=None):
    if colors is None:
        # colors = ['#0072bd', '#2ca02c', '#9467bd', '#edB120']
        colors = ['C0', 'C1', 'C0', 'C1']
    # ax.set_xlim(-176, -176+584)
    # ax.set_ylim(0, 584)
    for i in range(17):
        xy = np.array([(22, 60), (142, 79), (197, 42), (77, 23)])
        xy[:, 1] = xy[:, 1] + i * (94 - 79)
        if i in [2, 4, 12, 14]:
            j = np.where(np.array([2, 4, 12, 14]) == i)[0][0]
            poly = Polygon(xy, alpha=0.4, facecolor=colors[j], edgecolor='k', lw=1.2)
        else:
            poly = Polygon(xy, alpha=0.4, fill=False, facecolor='gray', lw=1.2)
        ax.add_patch(poly)

    p1 = (15, 0)
    ax.annotate("", xy=(197 + p1[0], 42), xycoords='data',
                xytext=(197 + p1[0], 42 + 16 * 15), textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="k"))
    p2 = (5, -10)
    ax.annotate("", xy=(77 + p2[0], 23 + p2[1]), xycoords='data',
                xytext=(197 + p2[0], 42 + p2[1]), textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="k"))
    p3 = (-5, -10)
    ax.annotate("", xy=(77 + p3[0], 23 + p3[1]), xycoords='data',
                xytext=(22 + p3[0], 60 + p3[1]), textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="k"))
    # ax.axis('off')
    ax.axis('equal')

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# share utils
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_ax_position(ax, xy=None):
    fig = ax.figure
    x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds

    if xy is None:
        return (x, y, w, h)
    elif xy == 'left':
        return (x, y + h / 2, w, h)
    elif xy == 'right':
        return (x + w, y + h / 2, w, h)
    elif xy == 'bottom':
        return (x + w / 2, y, w, h)
    elif xy == 'top':
        return (x + w / 2, y + h, w, h)

