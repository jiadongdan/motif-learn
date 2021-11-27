import numpy as np
import matplotlib.pyplot as plt
from ..plot.arrows import fig_add_arrow



def get_axes_wh(ccells):
    ws = [ccell.w for ccell in ccells]
    hs = [ccell.h for ccell in ccells]
    return ws, hs

def _connect_axes(ax1, ax2):
    x1, y1, w1, h1 = ax1.get_position().bounds
    x2, y2, w2, h2 = ax2.get_position().bounds

    if y1 > y2:
        # from y2 to y1
        posA = (x2 + w2 / 2, y2 + h2)
        posB = (x1 + w1 / 2, y1)
    else:
        # from y1 to y2
        posA = (x1 + w1 / 2, y1 + h1)
        posB = (x2 + w2 / 2, y2)
    fig = ax1.figure
    fig_add_arrow(fig, posA, posB, arrowstyle='-')


def fig_add_axes(fig, xy=(0.5, 0.5), w=2, h=3, zoom=0.2):
    fig_w, fig_h = fig.get_size_inches()
    fig_aspect = fig_w / fig_h
    ratio = w / h
    w, h = w * zoom, h * fig_aspect * zoom
    x, y = xy
    ax = fig.add_axes([x - w / 2, y - h / 2, w, h])
    return ax


def _construct_full_graph(motifs):
    g = np.array([e1.contains(e2) * 1 for e1 in motifs for e2 in motifs])
    g = g.reshape(len(motifs), len(motifs))
    l = np.array([e2.num for e1 in motifs for e2 in motifs])
    l = l.reshape(len(motifs), len(motifs)) * g
    np.fill_diagonal(g, 0)
    np.fill_diagonal(l, 0)
    return g, l


def _truncate_graph(full_graph, n_array):
    # np.fill_diagonal(n_array, 0)
    vmax_row = np.max(n_array, axis=1)
    g = np.array([1 if e == vmax else 0 for vmax, row in zip(vmax_row, n_array) for e in row])
    g = g.reshape(full_graph.shape) * full_graph
    # np.fill_diagonal(g, 0)
    return g

def _nodes_pos(levels, cnts):
    pos = []
    for y in range(len(levels)):
        cnt = cnts[y]
        for x in range(cnt):
            pos.append([(x - (cnt - 1) / 2) * 3, y])
    return np.array(pos)


class HGraph:

    def __init__(self, motifs: list):
        self.root = None

        self.levels, cnts = np.unique([motif.num for motif in motifs], return_counts=True)
        self.full_graph, self.n_array = _construct_full_graph(motifs)
        self.graph = _truncate_graph(self.full_graph, self.n_array)
        self.pos = _nodes_pos(self.levels, cnts)
        self.motifs = [motif.pixels for motif in motifs]
        self.cells = motifs

    def update_pos(self, index, xy=None, shift=0):
        if xy is None:
            x, y = self.pos[index]
            xy = [x + shift, y]
        self.pos[index] = xy

    def plot(self, ax=None, zoom=1):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        else:
            fig = ax.figure
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c='white', alpha=0)
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()

        trans1 = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        axes = []
        ws, hs = get_axes_wh(self.cells)
        for ii, (x, y) in enumerate(self.pos):
            w = ws[ii]
            h = hs[ii]
            xx, yy = trans1([x, y])  # data --> pixel
            xa, ya = trans2((xx, yy))  # pixel --> figure fraction
            a = fig_add_axes(ax.figure, xy=(xa, ya), w=w, h=h, zoom=zoom)
            axes.append(a)
            a.imshow(self.motifs[ii])
            a.axis('off')
        # ax.axis('off')
        for axis in ['top', 'bottom', 'right']:
            ax.spines[axis].set_visible(False)
        ax.xaxis.set_ticklabels([])
        ax.set_xticks([])

        # connect
        i, j = np.nonzero(self.graph)
        ij = np.vstack([i, j]).T
        for (i, j) in ij:
            _connect_axes(axes[i], axes[j])

    def plot1(self, ax=None, zoom=1., **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        else:
            fig = ax.figure
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c='white')
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()
        ss = np.array([np.max(e.shape) for e in self.motifs])
        ss = ss / ss.min()

        trans1 = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        axes = []
        ws, hs = get_axes_wh(self.cells)
        for ii, (x, y) in enumerate(self.pos):
            piesize = 0.04 * zoom * ss[ii]  # this is the image size
            w = 0.04 * zoom * ws[ii]
            h = 0.04 * zoom * hs[ii]
            p2 = (w + h) / 2.0
            xx, yy = trans1([x, y])  # data --> pixel
            xa, ya = trans2((xx, yy))  # pixel --> figure fraction
            a = fig.add_axes([xa - w / 2, ya - h / 2, w, h])
            axes.append(a)
            a.set_aspect('equal')
            self.cells[ii].plot(a, colors=['C0', 'C1', 'C4', 'C2'], **kwargs)
            a.axis('off')

        # connect
        i, j = np.nonzero(self.graph)
        ij = np.vstack([i, j]).T
        for (i, j) in ij:
            _connect_axes(axes[i], axes[j])
