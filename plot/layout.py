import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation
import string


def make_ax_3d(fig, ax, remove=False):
    x0, y0, w, h = ax.get_position().bounds
    ax1 = fig.add_axes([x0, y0, w, h], projection='3d', label=0)
    if remove:
        ax.remove()
    else:
        ax.axis('off')
    return ax1


def generate_h_axes(fig, n=2, y=0.1, h='min', left=1 / 15, right=1 / 30, wspace=0.25, ratios=None):
    """
        Create a set of horizontal axes.

        This utility wrapper makes it convenient to create horizontal layouts of
        subplots in a single call.

        Parameter
        ---------
        fig: `figure.Figure`

        n: int, optional, default: 2
            Number of columns of the generated axes.

        y: float, optional, default: 0.1

        left: float, optional, default: 1/15
            The fractional left side of the axes of the figure

        right: float, optional, default: 1/30
            The fractional right side of the axes of the figure

        wspace: float, optional, default: 0.25
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width

        ratios: scalar or array-like, default is None
            - If None or scalar type, the generated axes will have equal
              width ratio

        Returns
        A list of `axes.Axes` objects
        -------
        """
    if ratios == None:
        ratios = np.ones(n)
    elif np.isscalar(ratios):
        ratios = np.array([ratios] * n)
    elif np.iterable(ratios):
        ratios = np.array(ratios)

    width, height = fig.get_size_inches()
    ratio = width / height

    N = ratios.sum() + ratios.sum() / n * wspace * (n - 1)
    dx = (1 - left - right) / N
    ws = dx * ratios.sum() / n * wspace
    W = np.array([e * dx for e in ratios])

    if h == 'min':
        h = W.min()
    elif h == 'max':
        h = W.max()
    elif h == 'mean':
        h = W.mean()
    else:
        h = h*W.mean()

    axes = [fig.add_axes([left + W[0:i].sum() + i * ws, y, W[i], h * ratio]) for i in range(n)]
    return axes


def get_top_from_axes(axes, hspace=0.25):
    y = np.array([ax.get_position().bounds[1] for ax in axes]).mean()
    h = np.array([ax.get_position().bounds[3] for ax in axes]).mean()
    return y - h*hspace


def h_axes(fig, n=2, top=0.95, bottom=None, h='max', left=1 / 15, right=1 / 30, wspace=0.25, ratios=None):
    # process ratios
    if ratios == None:
        ratios = np.ones(n)
    elif np.isscalar(ratios):
        ratios = np.array([ratios] * n)
    elif np.iterable(ratios):
        ratios = np.array(ratios)

    aspect_ratio = get_fig_aspect(fig)

    # only to get lefts and rights
    gs_ = fig.add_gridspec(nrows=1, ncols=n, left=left, right=1 - right, bottom=0.1, top=0.2, wspace=wspace,
                           width_ratios=ratios)
    b, t, lefts, rights = gs_.get_grid_positions(fig)

    widths = rights - lefts
    if h == 'min':
        h = widths.min()
    elif h == 'max':
        h = widths.max()
    elif h == 'mean':
        h = widths.mean()
    elif h > 0 and h < 1:
        h = h * widths.mean()
    else:
        ind = np.minimum(len(widths) - 1, int(h))
        h = widths[ind]

    if bottom is None:
        bottom = top - h * aspect_ratio
    gs = fig.add_gridspec(nrows=1, ncols=n, left=left, right=1 - right, bottom=bottom, top=bottom + h * aspect_ratio,
                          wspace=wspace,
                          width_ratios=ratios)
    # https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.gridspec.GridSpecBase.html#matplotlib.gridspec.GridSpecBase.subplots
    # axes = [fig.add_subplot(gs[0, col]) for col in range(n)]
    axes = gs.subplots()
    return axes

def h_axes_(fig, n=2, y=0.1, h='max', left=1 / 15, right=1 / 30, wspace=0.25, ratios=None):
    # process ratios
    if ratios == None:
        ratios = np.ones(n)
    elif np.isscalar(ratios):
        ratios = np.array([ratios] * n)
    elif np.iterable(ratios):
        ratios = np.array(ratios)

    aspect_ratio = get_fig_aspect(fig)

    # only to get lefts and rights
    gs_ = fig.add_gridspec(nrows=1, ncols=n, left=left, right=1 - right, bottom=y, top=y + 0.1, wspace=wspace,
                           width_ratios=ratios)
    b, t, lefts, rights = gs_.get_grid_positions(fig)

    widths = rights - lefts
    if h == 'min':
        h = widths.min()
    elif h == 'max':
        h = widths.max()
    elif h == 'mean':
        h = widths.mean()
    elif h > 0 and h < 1:
        h = h * widths.mean()
    else:
        ind = np.minimum(len(widths) - 1, int(h))
        h = widths[ind]
    gs = fig.add_gridspec(nrows=1, ncols=n, left=left, right=1 - right, bottom=y, top=y + h * aspect_ratio,
                          wspace=wspace,
                          width_ratios=ratios)
    # https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.gridspec.GridSpecBase.html#matplotlib.gridspec.GridSpecBase.subplots
    #axes = np.array([fig.add_subplot(gs[0, col]) for col in range(n)])
    axes = gs.subplots()
    return axes

def get_y_from_axes(axes, hspace=0.25):
    y = np.array([ax.get_position().bounds[1] for ax in axes]).mean()
    h = np.array([ax.get_position().bounds[3] for ax in axes]).mean()
    return y+h*(1+hspace)


def get_size_inches(ax):
    fig_w, fig_h = ax.figure.get_size_inches()
    w, h = ax.get_position().bounds[2:]
    return (fig_w * w, fig_h * h)


def get_ax_aspect(ax):
    w, h = get_size_inches(ax)
    return w / h


def get_fig_aspect(fig):
    fig_w, fig_h = fig.get_size_inches()
    return fig_w / fig_h


def axes_from_ax(ax, nrows, ncols, wspace=0.1, hspace=0.1, width_ratios=None, height_ratios=None, remove=False):
    fig = ax.figure
    left, bottom, right, top = ax.get_position().extents
    # use gridspec here
    gs = fig.add_gridspec(nrows, ncols, left=left, right=right, bottom=bottom, top=top,
                          wspace=wspace, hspace=hspace,
                          width_ratios=width_ratios, height_ratios=height_ratios)
    # convert gridspec to subplots
    #axes = np.array([fig.add_subplot(gs[row, col]) for row in range(nrows) for col in range(ncols)]).reshape(nrows, ncols)
    axes = gs.subplots()
    if remove:
        ax.remove()
    else:
        ax.axis('off')
    return axes


def make_inset_ax(ax, x, y, zoom=0.2, w=None, h=None):
    if w is None and h is None:
        ratio = get_ax_aspect(ax)
        w = zoom
        h = zoom * ratio
    axins = ax.inset_axes([x - w / 2, y - h / 2, w, h])
    return axins

def make_inset_ax_(ax, x=None, y=None, zoom=0.2, w=None, h=None, loc='lower right'):
    if w is None and h is None:
        ratio = get_ax_aspect(ax)
        w = zoom
        h = zoom * ratio

    if loc == 'lower right':
        x, y = 1-w/2, h/2
    elif loc == 'lower left':
        x, y = w/2, h/2
    elif loc == 'upper right':
        x, y = 1-w/2, 1-h/2
    elif loc == 'upper left':
        x, y = w/2, 1-h/2

    axins = ax.inset_axes([x - w / 2, y - h / 2, w, h])

    return axins


def make_circular_axes(ax, n=8, zoom=0.1, l=0.3, theta=0, x=0.5, y=0.5):
    ratio = get_ax_aspect(ax)
    xyz = np.array([Rotation.from_euler('z', (angle + theta), degrees=True).apply([0, l, 0]) for angle in
                    np.arange(0, 360, 360 / n)])
    xyz[:, 1] *= ratio
    xyz[:, 0] += x
    xyz[:, 1] += y
    axes = [make_inset_ax(ax, xyz[i, 0], xyz[i, 1], zoom) for i in range(n)]
    return axes


def auto_range(ax, scale=1):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax-xmin)*(scale-1)/2
    dy = (ymax-ymin)*(scale-1)/2
    ax.set_xlim(xmin-dx, xmax+dx)
    ax.set_ylim(ymin-dy, ymax+dy)

def auto_resize(axes, f=0.5):
    if not np.iterable(axes):
        axes = [axes]
    fig = axes[0].figure
    for ax in axes:
        x1, y1, w1, h1 = ax.get_position().bounds
        x2, y2, w2, h2 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds

        d1 = (x1 - x2) * f
        d2 = (y1 - y2) * f
        x, y, w, h = x1 + d1, y1 + d2, w1 - d1, h1 - d2
        ax.set_position([x, y, w, h])


def auto_square(axes, ha=None, va=None):
    if not np.iterable(axes):
        axes = [axes]
    for ax in axes:
        ax_w, ax_h = get_size_inches(ax)
        d = np.minimum(ax_w, ax_h)
        fig_w, fig_h = ax.figure.get_size_inches()
        w, h = d / fig_w, d / fig_h

        x, y, w1, h1 = ax.get_position().bounds

        if ha in [None, 'center']:
            x += (w1 - w) / 2
        elif ha == 'right':
            x += (w1 - w)
        elif ha == 'left':
            x += 0

        if va in [None, 'center']:
            y += (h1 - h) / 2
        elif va == 'top':
            y += (h1 - h)
        elif va == 'bottom':
            y += 0
        ax.set_position([x, y, w, h])


def plot_image(ax, img, clip=False, **kwargs):
    h, w = img.shape[0:2]
    im = ax.imshow(img, **kwargs)
    ax.axis('off')
    if clip == True:
        patch = patches.Circle((w / 2 - 0.5, h / 2 - 0.5), radius=(h + w) / 4 - 1.5, transform=ax.transData)
        im.set_clip_path(patch)


def add_scale_bar(ax, l, x=0.05, y=0.05, **kwargs):
    # transfrom x and y to data coordinates
    x, y = ax.transData.inverted().transform(ax.transAxes.transform((x, y)))
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = 2
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = '#e4e1ca'
    ax.plot([x, x+l], [y, y], **kwargs)



def shift_ax(ax, left=0, right=0, upper=0, lower=0):
    bbox = ax.get_position()
    x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
    x = x0 + right - left
    y = y0 + upper - lower
    ax.set_position([x, y, w, h])

def merge_axes(axes, remove=False):
    pos = np.array([ax.get_position().extents for ax in axes])
    x1 = pos[:, 0].min()
    y1 = pos[:, 1].min()
    x2 = pos[:, 2].max()
    y2 = pos[:, 3].max()
    w = x2 - x1
    h = y2 - y1
    fig = axes[0].figure
    ax = fig.add_axes([x1, y1, w, h])
    if remove:
        for ax_ in axes:
            ax_.remove()
    return ax

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# related letters
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def ax_no_xylbs(ax):
    fig = ax.figure
    x1, y1, w1, h1 = ax.get_position().bounds
    x2, y2, w2, h2 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
    eps = 1e-3
    # contain image
    if len(ax.images) != 0:
        return True
    elif np.abs(w2 - w1) <= eps:
        return True
    else:
        return False


def add_letters(axes, dx=0.02, uppercase=True, **kwargs):
    font_letter = {'family': 'Arial',
                   'color': 'black',
                   'weight': 'bold',
                   'size': 10,
                   }
    if 'fontdict' not in kwargs:
        kwargs['fontdict'] = font_letter

    if uppercase:
        letters = list(string.ascii_uppercase)
    else:
        letters = list(string.ascii_lowercase)
    txts = []
    for ax, e in zip(axes, letters):
        if ax_no_xylbs(ax):
            txt = ax.text(x=0-dx, y=1., s=e, transform=ax.transAxes, va='top', ha='right', **kwargs)
        else:
            fig = ax.figure
            x1, y1, w1, h1 = ax.get_position().bounds
            x2, y2, w2, h2 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
            x = -(w2 - w1)/w1
            txt = ax.text(x=x+dx, y=1., s=e, transform=ax.transAxes, va='top', ha='left', **kwargs)
        txts.append(txt)
    return txts

def shift_letter(t, left=0, right=0, upper=0, lower=0):
    x0, y0 = t.get_position()
    x = x0 + right - left
    y = y0 + upper - lower
    t.set_position([x, y])


def connect_by_arrow(ax1, ax2, mode='h', s=0.5, arrow_props=None):
    if arrow_props is None:
        arrow_props = dict(facecolor='#413c39', ec=[0, 0, 0, 0], shrink=0.05, width=3, headwidth=8)

    x1, y1, w1, h1 = get_ax_box(ax1)
    x2, y2, w2, h2 = get_ax_box(ax2)

    if mode == 'h':
        if x1 > x2:
            start = (x1, y1 + s * h1)
            end = (x2 + w2, y1 + s * h1)
        else:
            start = (x1 + w1, y1 + s * h1)
            end = (x2, y1 + s * h1)
    elif mode == 'v':
        if y1 > y2:
            start = (x1 + s * w1, y1)
            end = (x1 + s * w1, y2 + h2)
        else:
            start = (x1 + s * w1, y1 + h1)
            end = (x1 + s * w1, y2)
    plt.annotate('', xytext=start, xy=end, arrowprops=arrow_props, xycoords='figure fraction')


def get_ax_box(ax):
    fig = ax.figure
    # x1, y1, w1, h1 = ax.get_position().bounds
    x2, y2, w2, h2 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
    return (x2, y2, w2, h2)


def connect_by_line(ax1, ax2, mode='right', fraction=0.4):
    x1, y1, w1, h1 = ax1.get_position().bounds
    x2, y2, w2, h2 = ax2.get_position().bounds
    if mode == 'left':
        start = (x1, y1 + h1 / 2)
        end = (x2, y2 + h2 / 2)
        fraction = -np.abs(fraction)
    elif mode == 'right':
        start = (x1 + w1, y1 + h1 / 2)
        end = (x2 + w1, y2 + h2 / 2)
        fraction = -np.abs(fraction)
    elif mode == 'top':
        start = (x1 + w1 / 2, y1 + h1)
        end = (x2 + w1 / 2, y2 + h2)
        fraction = -np.abs(fraction)
    elif mode == 'bottom':
        start = (x1 + w1 / 2, y1)
        end = (x2 + w1 / 2, y2)
        fraction = np.abs(fraction)

    arrow_props = dict(facecolor='#413c39', ec=[0, 0, 0, 0], shrink=0., width=0.8, headwidth=0, headlength=1.1)
    connectionstyle = "bar,fraction=" + str(fraction)
    arrow_props = dict(arrowstyle="-", shrinkA=0, shrinkB=0, connectionstyle=connectionstyle)

    a = plt.annotate("", xytext=start, xy=end, arrowprops=arrow_props, xycoords='figure fraction')
    return a


def align_ax(ax1, ax2, ha=None, va=None):
    x1, y1, w1, h1 = ax1.get_position().bounds
    x2, y2, w2, h2 = ax2.get_position().bounds

    if ha == 'left':
        x2 = x1
    elif ha == 'right':
        x2 = x1 + w1 - w2
    elif ha == 'center':
        x2 = x1 + w1 / 2 - w2 / 2
    if va == 'top':
        y2 = y1 + h1 - h2
    elif va == 'bottom':
        y2 = y1
    elif va == 'center':
        y2 = y1 + h1 / 2 - h2 / 2
    ax2.set_position([x2, y2, w2, h2])

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ticks related
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def auto_tick_length(axes, alpha=1.2):
    if not np.iterable(axes):
        axes = [axes]
    # auto-adjust major and minor tick lengths based on size of ax
    for ax in axes:
        h, w = get_size_inches(ax)
        l1 = max(1.5, min(h, w)*alpha)
        l2 = l1/2
        ax.tick_params(axis='both', which='major', length=l1, direction='in', width=0.5)
        ax.tick_params(axis='both', which='minor', length=l2, direction='in', width=0.5)


def font_size_func(s, vmin=4, slope=10/5):
    if s <=1 :
        return vmin
    else:
        return vmin + slope*(s-1)

def auto_tick_labels_size(axes):
    if not np.iterable(axes):
        axes = [axes]
    for ax in axes:
        h, w = get_size_inches(ax)
        s = (h + w)/2
        font_size = font_size_func(s)
        ax.tick_params(axis='both', labelsize=font_size)

def auto_tick_params(axes, alpha=1.2):
    auto_tick_labels_size(axes)
    auto_tick_length(axes, alpha)

def add_xylbs(ax, xlbs, ylbs):
    # get tick label size
    s = ax.xaxis.get_ticklabels()[0].get_fontsize()
    ax.set_xlabel(xlbs, fontsize=int(s) + 1)
    ax.set_ylabel(ylbs, fontsize=int(s) + 1)


def auto_axes(fig):
    def ax_with_data(ax):
        if len(ax.collections) == 0 and len(ax.images) == 0 and len(ax.collections) == 0:
            return False
        else:
            return True

    def ax_need_resize(ax):
        # check if axison
        return ax.axison


    # get axes that contain data
    axes = np.array([ax for ax in fig.axes if ax_with_data(ax)])

    mask = np.array([ax_need_resize(ax) for ax in axes])

    # auto resize axes without image
    auto_resize(axes[mask])

    # auto adjust tick length
    auto_tick_length(axes[mask])

    # auto adjust tick labels fontsize
    auto_tick_labels_size(axes[mask])


from matplotlib.offsetbox import AnchoredText

def add_anchor_text(ax, s, alpha=0.5, loc='upper left', fc=None, fontsize=5, fontcolor='#cee4cc' ,frameon=True):
    at = AnchoredText(s, prop=dict(size=fontsize, color=fontcolor), frameon=frameon, loc=loc, borderpad=0.0,)
    at.patch.set_alpha(alpha)
    if fc is not None:
        at.patch.set_fc(fc)
    at.patch.set_linewidth(0)
    ax.add_artist(at)


