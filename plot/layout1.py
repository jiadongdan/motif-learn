import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from matplotlib.path import Path
from matplotlib.patches import PathPatch


from scipy.spatial.transform import Rotation


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# function related to size
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Layout
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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


def get_top_from_axes(axes, hspace=0.25):
    y = np.array([ax.get_position().bounds[1] for ax in axes]).mean()
    h = np.array([ax.get_position().bounds[3] for ax in axes]).mean()
    return y - h*hspace


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


def make_inset_ax(ax, x=None, y=None, zoom=0.2, w=None, h=None, transform=None, loc=None):
    if transform is None:
        transform = ax.transAxes

    if w is None and h is None:
        ratio = get_ax_aspect(ax)
        if transform is ax.transAxes:
            w = zoom
            h = zoom * ratio
        elif transform is ax.transData:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            d = (xmax - xmin) / 2 + (ymax - ymin) / 2
            w = zoom * d
            h = zoom * ratio * d

    if loc == 'lower right':
        x, y = 1-w/2, h/2
    elif loc == 'lower left':
        x, y = w/2, h/2
    elif loc == 'upper right':
        x, y = 1-w/2, 1-h/2
    elif loc == 'upper left':
        x, y = w/2, 1-h/2

    axins = ax.inset_axes([x - w / 2, y - h / 2, w, h], transform=transform)

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



def make_ax_3d(fig, ax, remove=False):
    x0, y0, w, h = ax.get_position().bounds
    ax1 = fig.add_axes([x0, y0, w, h], projection='3d', label=0)
    if remove:
        ax.remove()
    else:
        ax.axis('off')
    return ax1

def ax_zoom(ax, scale=1.):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    xmin_ = x - (xmax-xmin)*scale/2
    xmax_ = x + (xmax-xmin)*scale/2
    ymin_ = y - (ymax-ymin)*scale/2
    ymax_ = y + (ymax-ymin)*scale/2
    ax.set_xlim(xmin_, xmax_)
    ax.set_ylim(ymin_, ymax_)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# colorbar
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def make_cbar_ax(ax, position='right', size=0.05, pad=0.05):
    x, y, w, h = ax.get_position().bounds

    if position == 'left':
        h = h
        w = w * size
        x = x - w - w / size * pad
        y = y
    elif position == 'right':
        h = h
        w = w * size
        x = x + w / size + w / size * pad
        y = y
    elif position == 'top':
        h = h * size
        w = w
        x = x
        y = y + h / size + h / size * pad
    elif position == 'bottom':
        h = h * size
        w = w
        x = x
        y = y - h / size * pad - h
    cax = ax.figure.add_axes([x, y, w, h])
    return cax

def make_cbar(ax, img=None, position='right', size=0.05, pad=0.05, cmap='viridis'):

    cax = make_cbar_ax(ax, position, size, pad)
    if img is None:
        mappable = ax.images[0]
    else:
        norm = Normalize(img.min(), img.max())
        mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar= ax.figure.colorbar(cax=cax)
    return cbar

def add_cbar(cax, vmin, vmax, cmap='viridis', **kwargs):
    norm = Normalize(vmin, vmax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar= cax.figure.colorbar(mappable=mappable, cax=cax, **kwargs)
    return cbar


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# functions to control ticks and fontsize
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

tick_length = lambda x: (3.5-1.)/(7.2)*x + 1.
tick_lbs_fontsize = lambda x: np.piecewise(x, [x < 1, x >= 1], [lambda t: 4, lambda t: 3/6.2*(t-1)+4])


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# auto ticks
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def auto_tick_length(axes, alpha=1.2):
    if not np.iterable(axes):
        axes = [axes]
    # auto-adjust major and minor tick lengths based on size of ax
    for ax in axes:
        h, w = get_size_inches(ax)
        s = (h + w) / 2
        l1 = tick_length(s)
        l2 = l1 / 2
        ax.tick_params(axis='both', which='major', length=l1, direction='in', width=0.5)
        ax.tick_params(axis='both', which='minor', length=l2, direction='in', width=0.5)

def auto_tick_labels_size(axes):
    if not np.iterable(axes):
        axes = [axes]
    for ax in axes:
        h, w = get_size_inches(ax)
        s = (h + w) / 2
        font_size = tick_lbs_fontsize(s)
        ax.tick_params(axis='both', labelsize=font_size)
        # set fontsize of xlabel and  ylabel
        ax.xaxis.label.set_size(font_size + 1)
        ax.yaxis.label.set_size(font_size + 1)


def has_ticklbs(ax):
    # get ticklabels
    xlbs = ax.get_xticklabels()
    ylbs = ax.get_yticklabels()
    # check eevery labels
    status1 = [~np.count_nonzero(np.array(e.get_window_extent().bounds[2:])==0)==2 for e in xlbs]
    status2 = [~np.count_nonzero(np.array(e.get_window_extent().bounds[2:])==0)==2 for e in ylbs]
    status = status1 + status2
    return np.array(status).any()

def has_xylbs(ax):
    wh1 = ax.get_xaxis().get_label().get_window_extent().bounds[2:]
    wh2 = ax.get_yaxis().get_label().get_window_extent().bounds[2:]
    status1 = ~np.count_nonzero(np.array(wh1)==0)==2
    status2 = ~np.count_nonzero(np.array(wh2)==0)==2
    return np.logical_or(status1, status2)

def has_lbs(ax):
    return np.logical_or(has_ticklbs(ax), has_xylbs(ax))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# auto resize axes based on the existence of tick labels
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def auto_resize(axes, f=0.5):
    if not np.iterable(axes):
        axes = [axes]
    fig = axes[0].figure
    for ax in axes:
        if has_lbs(ax):
            x1, y1, w1, h1 = ax.get_position().bounds
            x2, y2, w2, h2 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds

            d1 = (x1 - x2) * f
            d2 = (y1 - y2) * f
            x, y, w, h = x1 + d1, y1 + d2, w1 - d1, h1 - d2
            ax.set_position([x, y, w, h])

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# resize, shift, merge axes
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def ax_resize(ax, top=0, bottom=0, left=0, right=0):
    x, y, w, h = ax.get_position().bounds
    w = w + (left + right) * w
    h = h + (top + bottom) * h
    x = x - right * w
    y = y - bottom * h
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

def shift_ax(ax, left=0, right=0, top=0, bottom=0):
    x0, y0, w, h = ax.get_position().bounds
    x = x0 + right - left
    y = y0 + top - bottom
    if ax._label == 'inset_axes':
        ip = InsetPosition(ax._axes, [x, y, w, h])
        ax.set_axes_locator(ip)
    else:
        ax.set_position([x, y, w, h])

def shift_axes(axes, left=0, right=0, bottom=0, top=0):
    axes = np.atleast_1d(axes)
    for ax in axes:
        x0, y0, w, h = ax.get_position().bounds
        x = x0 + right - left
        y = y0 + top - bottom
        if ax._label == 'inset_axes':
            ip = InsetPosition(ax._axes, [x, y, w, h])
            ax.set_axes_locator(ip)
        else:
            ax.set_position([x, y, w, h])


def merge_axes(axes, remove=False):
    bboxes = [ax.get_position() for ax in axes.ravel()]
    bbox = Bbox.union(bboxes)
    ax_union = axes.ravel()[0].figure.add_axes(bbox.bounds)

    if remove:
        for ax_ in axes:
            ax_.remove()
    return ax_union

def auto_range(ax, scale=1):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax-xmin)*(scale-1)/2
    dy = (ymax-ymin)*(scale-1)/2
    ax.set_xlim(xmin-dx, xmax+dx)
    ax.set_ylim(ymin-dy, ymax+dy)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# auto letters
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_ax_position(ax, loc=None):
    fig = ax.figure
    x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds

    if loc == 'left':
        return (x, y + h / 2, w, h)
    elif loc == 'right':
        return (x + w, y + h / 2, w, h)
    elif loc == 'bottom':
        return (x + w / 2, y, w, h)
    elif loc == 'top':
        return (x + w / 2, y + h, w, h)
    else:
        return (x, y, w, h)

def get_position(a, xy=None):
    if hasattr(a, 'figure'):
        fig = a.figure
    else:
        raise AttributeError("The artist should have 'figure' attribute")

    if not hasattr(a, 'get_tightbbox'):
        raise AttributeError("The artist should have 'get_tightbbox' attribute")

    x, y, w, h = a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds

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



def init_letter_position(a):
    x, y, w, h = get_position(a)
    return (x-w/100, y+h)


def auto_letters(artists, uppercase=True, **kwargs):
    n = len(artists)
    if uppercase:
        letters = list(string.ascii_uppercase)[0:n]
    else:
        letters = list(string.ascii_lowercase)[0:n]

    # get initial positions of all letters
    xys = np.array([init_letter_position(a) for a in artists])
    fig = artists[0].figure
    ts = []
    for (x, y), s, ax in zip(xys, letters, artists):
        # here use plt
        t = plt.text(x, y, s, ha='right', va='top', transform=fig.transFigure, **kwargs)
        ts.append(t)
    ts_dict = {e.get_text():e for e in ts}
    return ts_dict


def align_letters(letters, t1, t2, ha=None, va=None, auto=False):
    # make sure either both capital and non-capital letters work
    if list(letters.keys())[0].isupper():
        t1, t2 = t1.upper(), t2.upper()
    else:
        t1, t2 = t1.lower(), t2.lower()
    l1 = letters[t1]
    l2 = letters[t2]
    x1, y1 = l1.get_position()
    x2, y2 = l2.get_position()
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
    x_l, x_r = min(x1, x2), max(x1, x2)
    y_b, y_t = min(y1, y2), max(y1, y2)

    # set ha and va if both are None
    if dx < dy and ha is None:
        if auto:
            if x1 == x_r:
                ha = 'right'
            else:
                ha = 'left'
        else:
            ha = 'center'

    if dx >= dy and va is None:
        if auto:
            if y1 == y_t:
                va = 'top'
            else:
                va = 'bottom'
        else:
            va = 'center'

    if ha == 'center':
        l1.set_position((x_c, y1))
        l2.set_position((x_c, y2))
    elif ha == 'left':
        l1.set_position((x_l, y1))
        l2.set_position((x_l, y2))
    elif ha == 'right':
        l1.set_position((x_r, y1))
        l2.set_position((x_r, y2))

    if va == 'center':
        l1.set_position((x1, y_c))
        l2.set_position((x2, y_c))
    elif va == 'top':
        l1.set_position((x1, y_t))
        l2.set_position((x2, y_t))
    elif va == 'bottom':
        l1.set_position((x1, y_b))
        l2.set_position((x2, y_b))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# auto connect by lines and arrows
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def connect_by_line(ax1, ax2, start='right', end='left', mode=5, fraction=0.5, **kwargs):
    x1, y1, w1, h1 = get_ax_position(ax1, loc=start)
    x2, y2, w2, h2 = get_ax_position(ax2, loc=end)

    d = np.hypot(x1 - x2, y1 - y2) * fraction
    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2

    arrow_props = dict(arrowstyle="-", shrinkA=0, shrinkB=0, **kwargs)

    if mode == 0:
        p1 = (x1, y1)
        p2 = (x2, y2)
    elif mode == 1:
        y = max(y1 + d, y2 + d)
        p1 = (x1, y)
        p2 = (x2, y)
    elif mode == 2:
        y = min(y1 - d, y2 - d)
        p1 = (x1, y)
        p2 = (x2, y)
    elif mode == 3:
        x = max(x1 + d, x2 + d)
        print(x, x1, x2)
        p1 = (x, y1)
        p2 = (x, y2)
    elif mode == 4:
        x = min(x1 - d, x2 - d)
        p1 = (x, y1)
        p2 = (x, y2)
    elif mode == 5:
        p1 = (x0, y1)
        p2 = (x0, y2)
    elif mode == 6:
        p1 = (x1, y0)
        p2 = (x2, y0)

    ax1.annotate("", xytext=(x1, y1), xy=p1, arrowprops=arrow_props, xycoords='figure fraction')
    ax1.annotate("", xytext=p1, xy=p2, arrowprops=arrow_props, xycoords='figure fraction')
    ax1.annotate("", xytext=p2, xy=(x2, y2), arrowprops=arrow_props, xycoords='figure fraction')


def connect_by_arrow(ax1, ax2, mode=None, s=0.5, arrow_props=None):
    if arrow_props is None:
        arrow_props = dict(facecolor='#413c39', ec=[0, 0, 0, 0], shrink=0.05, width=3, headwidth=8)

    x1, y1, w1, h1 = get_ax_position(ax1)
    x2, y2, w2, h2 = get_ax_position(ax2)

    dx = np.abs(x1 + w1 / 2 - x2 - w2 / 2)
    dy = np.abs(y1 + h1 / 2 - y2 - h2 / 2)

    if mode is None:
        if dx > dy:
            mode = 'h'
        else:
            mode = 'v'

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
    ax1.annotate('', xytext=start, xy=end, arrowprops=arrow_props, xycoords='figure fraction')


def make_bararrow(start, end, r=0.2, aspect_ratio=1., add_head=True, **kwargs):
    start = np.asarray(start)
    end = np.asarray(end)
    l = np.abs(end[0] - start[0]) / 2
    s = np.array([(start[0] + end[0]) / 2, start[1]])
    e = np.array([(start[0] + end[0]) / 2, end[1]])
    sgn1 = np.sign(start[0] - s[0])
    sgn2 = np.sign(e[1] - s[1])
    sgn3 = -sgn2
    sgn4 = np.sign(end[0] - e[0])

    if add_head:
        end[0] = end[0] + sgn1 * l / 8

    p1 = np.array([s[0] + sgn1 * r * l, start[1]])
    p2 = np.array([s[0], s[1] + sgn2 * r * l * aspect_ratio])
    p3 = np.array([s[0], e[1] + sgn3 * r * l * aspect_ratio])
    p4 = np.array([e[0] + sgn4 * r * l, end[1]])

    points = np.vstack([start, p1, s, p2, p3, e, p4, end, p4, e, p3, p2, s, p1, (0, 0)])
    codes = [1, 2, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 3, 79]

    if add_head:
        q1 = np.array([end[0], end[1] - l / 10])
        q2 = np.array([end[0] - sgn1 * l / 8, end[1]])
        q3 = np.array([end[0], end[1] + l / 10])
        points1 = np.vstack([q1, q2, q3, (0, 0)])
        points = np.vstack([points, points1])
        codes += [1, 2, 2, 79]
    pp = PathPatch(Path(points, codes), fill=True, **kwargs)

    return pp

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# debug
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def draw_box(ax):
    fig = ax.figure
    x, y, w, h = get_ax_position(ax)
    rect = Rectangle(xy=(x, y), width=w, height=h, fill=False, lw=2, color='red', transform=fig.transFigure)
    fig.add_artist(rect)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# arrows
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def fig_add_arrow(fig, start, end, color='#2d3742', width=1, ratio=2, connectionstyle=None, arrowprops=None):
    x1, y1 = np.atleast_1d(start)
    x2, y2 = np.atleast_1d(end)
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    w, h = fig.get_size_inches()
    d_inch = np.sqrt(((x1 - x2) * w) ** 2 + ((y1 - y2) * h) ** 2)
    d = d_inch * 72  # in points
    if arrowprops is None:
        # automatically determine arrow properties
        arrowprops = dict(width=width, headwidth=width * ratio, headlength=width * (ratio + 1), ec='none', fc=color,
                          connectionstyle=connectionstyle)

    arrow = plt.annotate('', end, start, xycoords='figure fraction', arrowprops=arrowprops)


def ax_add_arrow(ax, start, end, color='#2d3742', width=1, ratio=2, connectionstyle=None, **kwargs):
    x1, y1 = np.atleast_1d(start)
    x2, y2 = np.atleast_1d(end)
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    fig = ax.figure
    w, h = fig.get_size_inches()
    d_inch = np.sqrt(((x1 - x2) * w) ** 2 + ((y1 - y2) * h) ** 2)
    d = d_inch * 72  # in points
    # automatically determine arrow properties
    arrowprops = dict(width=width, headwidth=width * ratio, headlength=width * (ratio + 0.5), ec='none', fc=color,
                      connectionstyle=connectionstyle, shrink=0)
    arrow = ax.annotate('', end, start, xycoords='data', arrowprops=arrowprops, **kwargs)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# errorbar
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def ax_errorbar(x, y, ax=None, capsize=4, capthick=0.5, ecolor='gray', fmt='o--', **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    y_err = np.std(y, axis=0)
    y_mean = np.mean(y, axis=0)
    ax.errorbar(x, y_mean, yerr=y_err, capsize=capsize, capthick=capthick, ecolor=ecolor, fmt=fmt, **kwargs)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# errorbar
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def ax_boxplot(x, y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    ax.boxplot(x=y, positions=x, **kwargs)
