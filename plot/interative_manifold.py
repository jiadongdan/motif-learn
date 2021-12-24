import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


def _update_mean_patch(ax, p):
    if ax.images:  # ax.images not empty
        ax.images[0].set_data(p)
    else:
        ax.imshow(p)


class InterativeManifold:

    def __init__(self, fig, xy, ps, **kwargs):
        self.fig = fig
        self.ax_xy = fig.axes[0]
        self.ax_patch = fig.axes[1]

        self.xy = xy
        self.ps = ps

        self.path_collection = self.ax_xy.scatter(xy[:, 0], xy[:, 1], **kwargs)

        self.lasso = LassoSelector(self.ax_xy, onselect=self.onselect)

    def onselect(self, event):
        path = Path(event)
        self.ind = np.nonzero(path.contains_points(self.xy))[0]
        if self.ind.size != 0:
            self.ps_selected = self.ps[self.ind]
            self.xy_selected = self.xy[self.ind]

            p = self.ps_selected[0]
            _update_mean_patch(self.ax_patch, p)
            self.fig.canvas.draw_idle()