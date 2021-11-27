import numpy as np
from matplotlib.patches import FancyArrowPatch

from matplotlib.path import Path
from matplotlib.patches import PathPatch


#-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FancyArrowPatch is used here: it draws an arrow using the ArrowStyle.
#-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def fig_add_arrow(fig, start, end, **kwargs):
    # default
    if 'arrowstyle' not in kwargs:
        kwargs['arrowstyle'] = 'simple'
    if 'mutation_scale' not in kwargs:
        kwargs['mutation_scale'] = 20
    if 'color' not in kwargs:
        kwargs['color'] = '#413c39'

    arrow = FancyArrowPatch(start, end, **kwargs)
    fig.add_artist(arrow)
    return arrow


def ax_add_arrow(ax, start, end, **kwargs):
    # default
    if 'arrowstyle' not in kwargs:
        kwargs['arrowstyle'] = 'simple'
    if 'mutation_scale' not in kwargs:
        kwargs['mutation_scale'] = 20
    if 'color' not in kwargs:
        kwargs['color'] = '#413c39'

    arrow = FancyArrowPatch(start, end, **kwargs)
    ax.add_patch(arrow)
    return arrow

def align_arrows(arrow1, arrow2, mode=None):

    (x1, y1), (x2, y2) = arrow1._posA_posB
    (x3, y3), (x4, y4) = arrow2._posA_posB

    dx = np.abs(x1 + x2 - (x3 + x4))
    dy = np.abs(y1 + y2 - (y3 + y4))

    if mode is None:
        if dx > dy:
            mode = 'v'
        else:
            mode = 'h'

    if mode == 'h':
        posA = ((x1 + x2) / 2, y3)
        posB = ((x1 + x2) / 2, y4)
    elif mode == 'v':
        posA = (x3, (y1 + y2) / 2)
        posB = (x4, (y1 + y2) / 2)
    arrow2.set_positions(posA, posB)

def shift_arrow(arrow, dx=0, dy=0):
    (x1, y1), (x2, y2) = arrow._posA_posB

    posA = (x1 + dx, y1 + dy)
    posB = (x2 + dx, y2 + dy)
    arrow.set_positions(posA, posB)



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

def connect_by_fancyarrow(ax1, ax2, mode=None, s=0.5, **kwargs):

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

    fig = ax1.figure
    arrow = fig_add_arrow(fig, start, end, **kwargs)
    return arrow


def rotate_around(pts, p0=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(p0)
    p = np.atleast_2d(pts)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_arrow_points(posA, posB, num_control_pts, mode, r, s=0.5):
    def _add_extra_pts(p1, p2, p3, r):
        sign1 = np.sign(p2 - p1)
        sign2 = np.sign(p2 - p3)
        q1 = p2 - sign1 * [r, r]
        q2 = p2 - sign2 * [r, r]
        return np.vstack([p1, q1, p2, q2, p3])

    if num_control_pts == 3:
        if mode == 'v':
            p = np.array([posA[0], posB[1]])
        else:
            p = np.array([posB[0], posA[1]])
        sign1 = np.sign(p - posA)
        sign2 = np.sign(p - posB)
        p1 = p - sign1 * [r, r]
        p2 = p - sign2 * [r, r]
        # five points in total
        pts = np.vstack([posA, p1, p, p2, posB])
    elif num_control_pts == 4:
        if mode == 'v':
            p1 = np.array([posA[0], posA[1] * (1 - s) + s * posB[1]])
            p2 = np.array([posB[0], posA[1] * (1 - s) + s * posB[1]])
        else:
            p1 = np.array([posA[0] * (1 - s) + s * posB[0], posA[1]])
            p2 = np.array([posA[0] * (1 - s) + s * posB[0], posB[1]])
        pts1 = _add_extra_pts(posA, p1, p2, r)
        pts2 = _add_extra_pts(p1, p2, posB, r)
        # eight points in total
        pts = np.vstack([pts1[0:-1], pts2[1:]])
    else:
        pts = np.vstack([posA, posB])
    return pts


def get_head_points(posA, posB, mode, head_length):
    sign_xy = -np.sign(posB - posA)
    if mode == 'v':
        sign = sign_xy[1]
        q1 = np.array([posB[0], posB[1] + sign * head_length])
        q2 = np.array([q1[0] - head_length / 2, q1[1]])
        q3 = np.array([q1[0] + head_length / 2, q1[1]])
    elif mode == 'h':
        sign = sign_xy[0]
        q1 = np.array([posB[0] + sign * head_length, posB[1]])
        q2 = np.array([q1[0], q1[1] - head_length / 2])
        q3 = np.array([q1[0], q1[1] + head_length / 2])
    else:
        posBA = posB - posA
        angle = np.rad2deg(np.arctan2(posBA[1], posBA[0]))
        q1 = np.array([posB[0] - head_length, posB[1]])
        q2 = np.array([q1[0], q1[1] - head_length / 2])
        q3 = np.array([q1[0], q1[1] + head_length / 2])
        q1234 = np.vstack([q1, q2, q3, posB])
        q1, q2, q3, q4 = rotate_around(q1234, q1, angle)
        q1 = q1 - (q4 - posB)
        q2 = q2 - (q4 - posB)
        q3 = q3 - (q4 - posB)
    return np.vstack([q1, q2, posB, q3, q1])


# get codes according to number of control points and head
def get_codes(num_control_pts, add_head=True):
    if num_control_pts == 3:
        if add_head:
            codes = [1, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 79]
        else:
            codes = [1, 2, 3, 3, 2, 2, 3, 3, 79]
    elif num_control_pts == 4:
        if add_head:
            codes1 = [2, 3, 3, 2, 3, 3, 2]
            head_codes = [2, 2, 2, 2]
            codes = [1] + codes1 + head_codes + codes1[::-1][0:-1] + [79]
        else:
            codes1 = [2, 3, 3, 2, 3, 3, 2]
            codes = [1] + codes1 + codes1[::-1][0:-1] + [79]
    else:
        if add_head:
            codes = [1, 2, 2, 2, 2, 2, 79]
        else:
            codes = [1, 2, 79]
    return codes


def make_arrow(fig, start, end, num_control_pts=2, mode=None, r=0.05, add_head=True, s=0.5, **kwargs):
    def _arrow_mode(num_control_pts, mode):
        if num_control_pts == 3:
            if mode == 'v':
                arrow_mode = 'h'
            else:
                arrow_mode = 'v'
        elif num_control_pts == 4:
            arrow_mode = mode
        else:
            arrow_mode = None
        return arrow_mode

    if 'color' not in kwargs:
        kwargs['color'] = '#2d3742'
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = 3

    # convert start and end to pixel space
    fig2pixel = fig.transFigure
    posA = fig2pixel.transform(start)
    posB = fig2pixel.transform(end)
    l = np.max(np.abs(posA - posB))
    radius = l * r
    # head_length need to change according to fig size
    #head_length = radius / 2
    head_length = 10
    arrow_mode = _arrow_mode(num_control_pts, mode)

    pts = get_arrow_points(posA, posB, num_control_pts, mode, radius, s)
    head = get_head_points(posA, posB, arrow_mode, head_length)
    codes = get_codes(num_control_pts, add_head)

    if add_head:
        points = np.vstack([pts[0:-1], head, pts[0:-1][::-1]])
    else:
        points = np.vstack([pts, pts[0:-1][::-1]])

    # convert from pixel space to figure fraction space
    pixel2fig = fig.transFigure.inverted()
    points_ = pixel2fig.transform(points)

    pp = PathPatch(Path(points_, codes), fill=True, **kwargs)
    return pp



def ax_get_position(ax, loc=None, pad=0.005, transform='fig'):
    fig = ax.figure
    if transform == 'fig':
        x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
    elif transform == 'display':
        x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).bounds
        # convert to pixel space
        fig2pixel = ax.figure.transFigure
        pad = fig2pixel.transform([pad, pad])[0]

    if loc == 'left':
        return (x-pad, y + h / 2, w, h)
    elif loc == 'right':
        return (x + w + pad, y + h / 2, w, h)
    elif loc == 'bottom':
        return (x + w / 2, y-pad, w, h)
    elif loc == 'top':
        return (x + w / 2, y + h+pad, w, h)
    elif loc == 'center':
        return (x + w / 2, y + h / 2, w, h)
    else:
        return (x, y, w, h)


def estimate_loc1_loc2(ax1, ax2):
    x1, y1, w1, h1 = ax_get_position(ax1, loc='center', transform='display')
    x2, y2, w2, h2 = ax_get_position(ax2, loc='center', transform='display')
    dx = np.abs(x1-x2)
    dy = np.abs(y1-y2)
    if dx > dy:
        if x1 > x2:
            loc1, loc2 = 'left', 'right'
        else:
            loc1, loc2 = 'right', 'left'
    else:
        if y1 > y2:
            loc1, loc2 = 'bottom', 'top'
        else:
            loc1, loc2 = 'top', 'bottom'
    return (loc1, loc2)

def estimate_mode(loc1):
    if loc1 in ['top', 'bottom']:
        mode = 'v'
    elif loc1 in ['left', 'right']:
        mode = 'h'
    else:
        mode = None
    return mode

def connect_by_arrow(ax1, ax2, loc1=None, loc2=None, n=2, mode=None, r=0.05, add_head=True, s=0.5, pad=0.005, **kwargs):
    if loc1 is None and loc2 is None:
        loc1, loc2 = estimate_loc1_loc2(ax1, ax2)

    if mode is None:
        mode = estimate_mode(loc1)

    x1, y1, w1, h1 = ax_get_position(ax1, loc=loc1, pad=pad)
    x2, y2, w2, h2 = ax_get_position(ax2, loc=loc2, pad=pad)

    fig = ax1.figure
    pp = make_arrow(fig,(x1, y1), (x2, y2), n, mode, r, add_head, s, **kwargs)
    fig.add_artist(pp)
    return pp
