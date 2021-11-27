from matplotlib.patches import FancyBboxPatch


def fig_add_box(fig, x, y, zoom=1 / 10, ratio=0.5, **kwargs):
    w = zoom
    h = w * ratio
    pad = min(w, h) / 3

    if 'boxstyle' not in kwargs:
        kwargs['boxstyle'] = "round, pad={}".format(pad)
    if 'fill' not in kwargs:
        kwargs['fill'] = False
    if 'ec' not in kwargs:
        kwargs['ec'] = '#2d3742'

    box = FancyBboxPatch(xy=(x - w / 2, y - h / 2), width=w, height=h, transform=fig.transFigure, **kwargs)
    fig.add_artist(box)
    return box


def fig_add_boxes(fig, xys, zoom=1/10, ratio=0.5, **kwargs):
    boxes = [fig_add_box(fig, x, y, zoom, ratio, **kwargs)   for (x, y) in xys]
    return boxes

def shift_box(box, dx=0, dy=0):
    x = box.get_x() + dx
    y = box.get_y() + dy

    box.set_x(x)
    box.set_y(y)




