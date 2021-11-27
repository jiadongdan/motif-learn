import numpy as np
from sklearn.decomposition import PCA
from .auto_clustering import seg_lbs
from .auto_clustering import kmeans_lbs

from matplotlib.transforms import Bbox
from matplotlib.path import Path
from skimage.transform import rotate, warp_polar
#from skimage.feature import register_translation
from skimage.morphology import disk

from skimage.registration import phase_cross_correlation


def get_grid_inds(xy, tn=16, r=0.7):
    xy = xy / np.abs(xy).max()
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    rs = np.hypot(xy[:, 0], xy[:, 1])
    for i in np.arange(tn):
        m1 = angles > i * np.pi * 2 / (tn)
        m2 = angles > (i+1) * np.pi * 2 / (tn)
        m3 = rs > r
        m = np.hstack([m1, m2, m3])
        break




def get_xy_box(xy, margin=0.1):
    dx = xy[:, 0].max() - xy[:, 0].min()
    dy = xy[:, 1].max() - xy[:, 1].min()
    x0, y0 = xy.mean(axis=0)
    d = max(dx, dy) * (1 + margin) / 2

    return Bbox([[x0 - d, x0 + d], [y0 - d, y0 + d]])

def grids(bbox, n1=10, n2=10):
    xmin, xmax, ymin, ymax = bbox.extents
    xs = np.linspace(xmin, xmax, n1 + 1)
    xx = np.stack([xs, np.roll(xs, -1)]).T[0:-1]
    ys = np.linspace(ymin, ymax, n2 + 1)
    yy = np.stack([ys, np.roll(ys, -1)]).T[0:-1]
    paths = [Path([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) for (x1, x2) in xx for (y1, y2) in yy]
    return paths


def register_rotation(img1, img2, upsample_factor=20):
    mask = disk((img1.shape[0]-1)/2)
    img1_polar = warp_polar(img1 * mask)
    img2_polar = warp_polar(img2 * mask)
    shifts, error, phasediff = phase_cross_correlation(img1_polar, img2_polar, upsample_factor=upsample_factor)
    return shifts[0]

def register_imgs(imgs, upsample_factor=20, return_angles=False):
    ref = imgs[0]
    angles = []
    for i, img in enumerate(imgs):
        a = register_rotation(img, ref, upsample_factor)
        angles.append(a)
        img_rot = rotate(img, angle=a, order=1)
        ref = (ref * (i + 1) + img_rot) / (i + 2)
    if return_angles:
        return ref, np.array(angles)
    else:
        return ref



class ClusterDataSource:

    def __init__(self, img, pts, X, xy, ps, lbs=None, level=1, parent=None):
        self.img = img
        self.pts = pts
        self.X = X
        self.xy = xy
        self.ps = ps
        self.level = level
        self.parent = parent
        if lbs is None:
            self.lbs = seg_lbs(xy)
        else:
            self.lbs = lbs
        if self.lbs is None:
            self.num_clusters = None
        else:
            self.num_clusters = len(np.unique(self.lbs))

    def get_mean_motifs(self):
        motifs = np.array([self.ps[self.lbs==e].mean(axis=0) for e in np.unique(self.lbs)])
        return motifs

    def select(self, cls):
        if self.lbs is None:
            raise ValueError('Labels of clusters have not been set.')
        else:
            if not np.iterable(cls):
                cls = [cls]
            ind = np.where(np.in1d(self.lbs, cls))[0]
        return ClusterDataSource(self.img, self.pts[ind],
                                 self.X[ind], self.xy[ind],
                                 self.ps[ind], self.lbs[ind],
                                 self.level+1, cls)

    # this is the key component
    def update_xy(self, X=None, mode='pca', states=None):
        if X is None:
            X = self.X
        X = X.select(states)
        if mode == 'pca':
            self.xy = PCA(n_components=2).fit_transform(X)
        elif mode == 'force':
            pass
        return self


    def update_seg_lbs(self):
        self.lbs = seg_lbs(self.xy)

    def update_kmeans_lbs(self):
        self.lbs = kmeans_lbs(self.X)

    # This is abit slow
    def grid_imgs(self, n1=10, n2=10, rot=True, diff=False):
        bbox = get_xy_box(self.xy)
        gs = grids(bbox, n1=n1, n2=n2)
        inds = [np.nonzero(g.contains_points(self.xy))[0] for g in gs]
        imgs = []
        valid = []
        for i, ind in enumerate(inds):
            if ind.size == 0:
                im = np.zeros_like(self.ps[0])
            else:
                print(i, end=',')
                valid.append(i)
                if rot:
                    im = register_imgs(self.ps[ind])
                else:
                    im = self.ps[ind].mean(axis=0)
            imgs.append(im - im.mean())
        valid = np.array(valid)
        imgs = np.array(imgs)

        if rot:
            _, angles = register_imgs(imgs[valid], return_angles=True)
            imgs_ = np.array([rotate(e, angle) for angle, e in zip(angles, imgs[valid])])
            if diff:
                imgs[valid] = (imgs_ - imgs_.mean(axis=0))
            else:
                imgs[valid] = imgs_

        tt = np.concatenate(imgs, axis=1)
        kk = np.split(tt, n1, axis=1)
        gg = np.vstack(kk).T
        return gg
