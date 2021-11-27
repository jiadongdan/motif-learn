import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import to_rgb
from matplotlib.colors import to_rgba_array


from collections import namedtuple


def _update_color(ax, ind, color, color_selected):
    # get path collection
    path_collection = ax.collections[0]
    fc = color.copy()
    fc[ind, 0:3] = to_rgb(color_selected)
    path_collection.set_color(fc)

def _update_pts(ax, pts, **kwargs):
    if ax.collections:
        ax.collections = []
    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

def _update_mean_patch(ax, p):
    if ax.images:  # ax.images not empty
        ax.images[0].set_data(p)
    else:
        ax.imshow(p)



Cluster = namedtuple('Cluster', ['X', 'pts', 'ind'])


class InteractiveCluster:

    def __init__(self, fig, X, img, pts, ps, color=None, color_selected='C9', color_pts='red', s=5):
        self.fig = fig
        self.ax_img = fig.axes[0]
        self.ax_cluster = fig.axes[1]
        self.ax_patch = fig.axes[2]

        if color is None:
            self.color = np.array(['C0']*len(X))
        elif not np.iterable(color):
            self.color = np.array([color] * len(X))
        else:
            self.color = color
        self.color = np.array([to_rgb(e) for e in self.color])

        self.color_selected = color_selected
        self.color_pts = color_pts

        self.s = s

        self.path_collection = self.ax_cluster.scatter(X[:, 0], X[:, 1], s=self.s, c=self.color)
        self.ax_cluster.axis('equal')
        self.ax_img.imshow(img)
        self.ax_img.axis('off')

        self.img = img
        self.pts = pts
        self.X = X
        self.ps = ps

        self.ind = None
        self.X_selected = None
        self.pts_selected = None

        self.lbs = np.array(len(self.pts)*[-1])

        self.num_clusters = 0

        self.lasso = LassoSelector(self.ax_cluster, onselect=self.onselect)
        self.press = self.fig.canvas.mpl_connect("key_press_event", self.press_key)


    def onselect(self, event):
        path = Path(event)
        self.ind = np.nonzero(path.contains_points(self.X))[0]
        if self.ind.size != 0:
            self.pts_selected = self.pts[self.ind]
            self.X_selected = self.X[self.ind]

            # update alpha
            _update_color(self.ax_cluster, self.ind, self.color, self.color_selected)
            # update pts
            _update_pts(self.ax_img, self.pts_selected, color=self.color_pts, s=self.s)
            # update mean patch
            _update_mean_patch(self.ax_patch, self.ps[self.ind].mean(axis=0))
            self.fig.canvas.draw_idle()

    def press_key(self, event):
        if event.key == "enter":
            if self.ind.any():
                self.lbs[self.ind] = self.num_clusters
                self.num_clusters += 1
                print("One cluster has been selected.")

def interactive_clusters(X, img, pts, ps, color=None, color_selected='C9', color_pts='red'):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    app = InteractiveCluster(fig, X, img, pts, ps, color, color_selected, color_pts)
    return app