import numpy as np
#from statistics import mode
import matplotlib.pyplot as plt
from scipy.stats import mode
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from .colors import generate_colors_from_lbs
from .colormaps import color_palette
from ..clustering.auto_clustering import seg_lbs


#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from skimage.morphology import disk
from skimage.transform import rotate, warp_polar
from sklearn.utils import check_random_state

def get_angle(img1, img2):
    mask = disk(img1.shape[0]//2)
    img1_rot = warp_polar(img1*mask)
    img2_rot = warp_polar(img2*mask)
    shifts, error, phasediff = phase_cross_correlation(img1_rot, img2_rot, upsample_factor=20)
    return shifts[0]

def _register_imgs(imgs):
    ref = imgs[0]
    for i, img in enumerate(imgs):
        a = get_angle(img, ref)
        img_rot = rotate(img, angle=a, order=1)
        ref = (ref*(i+1) + img_rot)/(i+2)
    return ref

def register_imgs(imgs, max_samples=15, seed=48):
    if len(imgs) > max_samples:
        rng = check_random_state(seed=seed)
        mask = rng.choice(len(imgs), max_samples, replace=False)
        imgs = imgs[mask]
    return _register_imgs(imgs)


def _update_pts(ax, pts, **kwargs):
    if ax.collections:
        ax.collections = []
    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)


def _update_mean_patch(ax, p, cmap, clip=True):
    if ax.images:  # ax.images not empty
        ax.images[0].set_data(p)
        ax.images[0].set_cmap(cmap)
    else:
        ax.imshow(p, cmap=cmap)
    if clip:
        c = plt.Circle((p.shape[0] / 2 - 0.25, p.shape[1] / 2 - 0.25), radius=p.shape[0] / 2, transform=ax.transData)
        ax.images[0].set_clip_path(c)


class InteractiveCluster:

    def __init__(self, fig, X, img, pts, ps, lbs=None, clip=True, max_samples=15, rotate=False, **kwargs):
        self.fig = fig
        self.ax_img = fig.axes[0]
        self.ax_cluster = fig.axes[1]
        self.ax_patch = fig.axes[2]

        if lbs is None:
            self.lbs_ = seg_lbs(X)
        else:
            self.lbs_ = lbs
        self.colors = generate_colors_from_lbs(self.lbs_)

        self.path_collection = self.ax_cluster.scatter(X[:, 0], X[:, 1], c=self.colors, **kwargs)
        for e in np.unique(self.lbs_):
            x, y = X[self.lbs_ == e].mean(axis=0)
            self.ax_cluster.text(x, y, s=e, transform=self.ax_cluster.transData)
        self.ax_cluster.axis('equal')
        self.ax_img.imshow(img)
        self.ax_img.axis('off')
        self.ax_patch.set_xlim(0 - 0.5, ps.shape[2] - 0.5)
        self.ax_patch.set_ylim(ps.shape[1] - 0.5, 0 - 0.5)

        self.img = img
        self.pts = pts
        self.X = X
        self.ps = ps
        self.clip = clip
        self.max_samples = max_samples
        self.rotate = rotate

        self.ind = None
        self.X_selected = None
        self.pts_selected = None

        self.lbs = np.array(len(self.pts) * [-1])

        self.num_clusters = 0

        self.lasso = LassoSelector(self.ax_cluster, onselect=self.onselect)
        self.press = self.fig.canvas.mpl_connect("key_press_event", self.press_key)

    def onselect(self, event):
        path = Path(event)
        self.ind = np.nonzero(path.contains_points(self.X))[0]
        if self.ind.size != 0:
            self.pts_selected = self.pts[self.ind]
            self.X_selected = self.X[self.ind]

            c = mode(self.colors[self.ind])[0][0]
            # update pts
            _update_pts(self.ax_img, self.pts_selected, color='r', s=3)
            # update mean patch
            if self.rotate:
                p = register_imgs(self.ps[self.ind], self.max_samples)
            else:
                p = self.ps[self.ind].mean(axis=0)
            _update_mean_patch(self.ax_patch, p, cmap=color_palette(c), clip=self.clip)
            self.fig.canvas.draw_idle()

    def press_key(self, event):
        if event.key == "enter":
            if self.ind.any():
                self.lbs[self.ind] = self.num_clusters
                self.num_clusters += 1
                print("One cluster has been selected.")


def interactive_clusters(X, img, pts, ps, lbs=None, clip=True, max_samples=15, rotate=False, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    app = InteractiveCluster(fig, X, img, pts, ps, lbs, clip, max_samples, rotate, **kwargs)
    return app


class InteractivePentagon:

    def __init__(self, fig, xy, pts):
        self.fig = fig
        self.ax_xy = self.axes[0]
        self.ax_pentagon = self.axes[1]

        self.ax_xy.scatter(xy[:, 0], xy[: 1])


        self.ind = None
        self.xy = xy
        self.pts = pts

        self.lasso = LassoSelector(self.ax_xy, onselect=self.onselect)


    def onselect(self, event):
        path = Path(event)
        self.ind = np.nonzero(path.contains_points(self.xy))[0]
        if self.ind.size != 0:
            pc = self.pts[self.ind].mean(axis=0)
            _update_pts(self.ax_pentagon, pc, color='r', s=3)

def display_pentagons(xy, pts):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.2/2))
    app = InteractivePentagon(fig, xy, pts)
    return app



