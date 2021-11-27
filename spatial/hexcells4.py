import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.path import Path
from matplotlib.patches import Polygon
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import inv
from skimage.measure import label, regionprops
from skimage.transform import estimate_transform

from matplotlib.collections import LineCollection

import base64
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from skimage.transform import rescale
from scipy.sparse.csgraph import connected_components

import numbers
from numpy.lib.stride_tricks import as_strided

from sklearn.cluster import KMeans, DBSCAN

from ..plot.colors import colors_from_lbs


def estimate_r0(pts, k=7):
    nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
    d, ind = nbrs.kneighbors(pts)
    d = d[:, 1:].ravel()[:, np.newaxis]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(d)
    lbs = kmeans.labels_
    d1 = (d[lbs == 0]).mean()
    d2 = (d[lbs == 1]).mean()
    return (d1 + d2) / 2


def get_hex_p0_uv(pts, r=None):
    if r is None:
        r = estimate_r0(pts)
    nbrs = NearestNeighbors(n_neighbors=7).fit(pts)
    inds = nbrs.kneighbors(pts, return_distance=False)

    pts1 = (pts[inds] - pts[:, np.newaxis, :]).reshape(-1, 2)
    r1 = np.hypot(pts1[:, 0], pts1[:, 1])
    mask = np.logical_and(r1 <= r, r1 > 0)
    pts2 = pts1[mask]

    kmeans = KMeans(n_clusters=6, random_state=0).fit(pts2)
    lbs = kmeans.labels_
    p6 = np.array([pts2[lbs == e].mean(axis=0) for e in range(6)])
    angles = np.abs(np.rad2deg(np.arctan2(p6[:, 1], p6[:, 0])))
    i, j = np.argsort(angles)[0:2]
    u, v = p6[i], p6[j]

    # get p0
    ind = nbrs.kneighbors(pts.mean(axis=0)[np.newaxis, :], return_distance=False)[0][0]
    p0 = pts[ind]
    return (p0, u, v)


def _get_p0_u_v(pts):
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pts)
    ind = nbrs.kneighbors(pts.mean(axis=0)[np.newaxis, :], return_distance=False)[0][0]

    p0 = pts[ind]
    inds = nbrs.kneighbors(p0[np.newaxis, :], return_distance=False)[0][1:]
    pp6 = pts[inds] - p0
    angles = np.abs(np.rad2deg(np.arctan2(pp6[:, 1], pp6[:, 0])))
    ind2 = inds[np.argsort(angles)][0:2]
    u = pts[ind2[0]] - p0
    v = pts[ind2[1]] - p0
    return p0, u, v


def _remove_duplicate_pts(pts):
    return np.unique(pts, axis=0)


def points_to_cell_inds(pts, threshold=None):
    if threshold is None:
        threshold = _estimate_r0(pts) * 0.2
    p0, u, v = _get_p0_u_v(pts)
    pts1 = pts + u
    pts2 = pts + v
    pts3 = pts + u + v
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)
    d1, ind1 = nbrs.kneighbors(pts1)
    d2, ind2 = nbrs.kneighbors(pts2)
    d3, ind3 = nbrs.kneighbors(pts3)
    d123 = np.hstack([d1, d2, d3])
    ind1234 = np.hstack([np.arange(len(pts))[:, np.newaxis], ind1, ind3, ind2])
    mask = d123.mean(axis=1) < threshold
    inds = ind1234[mask]
    return inds


# estimate nearest neighbor radius
def _estimate_r0(pts):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pts)
    d = d[:, 1]
    return d.mean()


def get_ordered_lbs(pts, lbs, inds):
    nmax = np.max([len(e) for e in inds])
    lbs_ordered = []
    for ii, row in enumerate(inds):
        if len(row) == 0:
            lbs_select = [-1] * nmax
        elif len(row) == 1:
            lbs_select = [lbs[row[0]]] + [-1] * (nmax - 1)
        else:
            pts_select = pts[row] - (pts[row]).mean(axis=0)
            angles = np.arctan2(pts_select[:, 1], pts_select[:, 0]) + np.pi
            # sort by angle
            lbs_row = lbs[row][np.argsort(angles)]
            # rolling min to first
            idx = np.argmin(lbs_row)
            lbs_select = np.roll(lbs_row, -idx).tolist() + [-1] * (nmax - len(angles))
        lbs_ordered.append(lbs_select)
    return np.array(lbs_ordered)


def reorder_lbs(lbs, vals=None, mode='min'):
    unique, counts = np.unique(lbs, return_counts=True)
    if vals is None:
        vals = counts
    if mode == 'min':
        lbs_order = np.argsort(vals)
    else:
        lbs_order = np.argsort(vals)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)
    return lbs


def get_rowcol_lbs(X, eps=0.3):
    # clustering
    dbscan = DBSCAN(eps=eps, min_samples=1).fit(X)
    lbs = dbscan.labels_
    # sort lbs
    vals = np.array([X[lbs == e].mean() for e in np.unique(lbs)])
    lbs = reorder_lbs(lbs, vals)
    return lbs


def pts2grid(pts):
    lbs1 = get_rowcol_lbs(pts[:, 0:1])
    lbs2 = get_rowcol_lbs(pts[:, 1:2])
    pts1 = pts.copy()
    for i, e in enumerate(np.unique(lbs2)):
        kk = pts1[lbs2 == e]
        kk[:, 1] = i
        pts1[lbs2 == e] = kk

    for i, e in enumerate(np.unique(lbs1)):
        kk = pts1[lbs1 == e]
        kk[:, 0] = i
        pts1[lbs1 == e] = kk
    return pts1.astype(int)


class HexCells:

    def __init__(self, pts):
        self.pts = _remove_duplicate_pts(pts)
        p0, self.u, self.v = _get_p0_u_v(self.pts)
        self.inds = points_to_cell_inds(self.pts)
        self.cells = self.pts[self.inds]

        # region(cell) centers
        self.centers = np.array([self.pts[ind].mean(axis=0) for ind in self.inds])

        p0, u, v = get_hex_p0_uv(self.centers)
        uv = np.vstack([u, v])
        ij = (self.centers - p0).dot(inv(uv))
        self.centers_ = pts2grid(ij)

        # get transform
        self.tform = estimate_transform('affine', self.centers_, self.centers)
        self.pts_ = self.tform.inverse(self.pts)

        self.pos = None
        self.lbs = None
        self.cell_lbs = None

    @property
    def image(self):
        if self.cell_lbs is None:
            raise ValueError('cell_lbs is not set.')
        vmax = max(np.ptp(self.centers_[:, 0]), np.ptp(self.centers_[:, 1]))
        cells_matrix = np.ones(shape=(vmax + 1, vmax + 1)) * (-1)
        # need cell_lbs
        for idx, (i, j) in enumerate(self.centers_):
            cells_matrix[j, i] = self.cell_lbs[idx]
        return cells_matrix

    @property
    def id_image(self):
        id_matrix = np.ones_like(self.image) * (-1)
        for idx, (i, j) in enumerate(self.centers_):
            id_matrix[j, i] = idx
        return id_matrix.astype(int)

    # this is important
    def set_cell_lbs(self, pos, lbs):
        polys = self.pts[self.inds]
        # indices of points inside each cell
        inds = []
        # len(self.pts) is equal to number of cells
        self.pos = []
        self.lbs = []
        for poly in polys:
            ind = np.where(Path(poly).contains_points(pos) == 1)[0]
            inds.append(ind)
            self.pos.append(pos[ind])
            self.lbs.append(lbs[ind])
        lbs_ordered = get_ordered_lbs(pos, lbs, inds)
        lbs_ordered_unique, self.cell_lbs = np.unique(lbs_ordered, axis=0, return_counts=False, return_inverse=True)
        self.cell_lbs = reorder_lbs(self.cell_lbs, mode='max')

    @property
    def connected_cells(self):
        threshold = 0
        connectivity = 1
        # label image
        lbs_image = label(self.image > threshold, connectivity=connectivity)
        # regions
        region_props = regionprops(lbs_image)

        # areas and components
        areas = np.array([e.area for e in region_props if e.area > 1])
        components = [self.image[e.slice] for e in region_props if e.area > 1]
        ids = [self.id_image[e.slice] for e in region_props if e.area > 1]
        masks = [(e > 0) * 1 for e in components]
        components = [components[i] * masks[i] for i in range(len(masks))]
        ids = [ids[i] * masks[i] for i in range(len(masks))]
        # sort components
        ind = np.argsort(areas)
        self.components = np.array([components[e] for e in ind], dtype=object)
        self.ids = np.array([ids[e] for e in ind], dtype=object)

        # unique areas and components
        self.components, lbs = unique_components(self.components)
        self.ids = np.array([self.ids[np.where(lbs == e)[0][0]] for e in np.unique(lbs)], dtype=object)

        cs = []
        for i in range(len(self.ids)):
            idx = self.ids[i].ravel()[self.ids[i].ravel() != 0]
            pos = np.vstack([self.pos[e] for e in idx])
            lbs = np.hstack([self.lbs[e] for e in idx])
            cs.append(ConnectedCells(self.components[i], self.cells[idx], pos, lbs))
        return cs

    def get_connected_cells(self, min_area=3, max_area=10):
        threshold = 0
        connectivity = 1
        # label image
        lbs_image = label(self.image > threshold, connectivity=connectivity)
        # regions
        region_props = regionprops(lbs_image)

        # areas and components
        areas = np.array([e.area for e in region_props if np.logical_and(e.area >= min_area, e.area <= max_area)])
        components = [self.image[e.slice] for e in region_props if
                      np.logical_and(e.area >= min_area, e.area <= max_area)]
        ids = [self.id_image[e.slice] for e in region_props if np.logical_and(e.area >= min_area, e.area <= max_area)]
        masks = [(e > 0) * 1 for e in components]
        components = [components[i] * masks[i] for i in range(len(masks))]
        ids = [ids[i] * masks[i] for i in range(len(masks))]
        # sort components
        ind = np.argsort(areas)
        self.components = np.array([components[e] for e in ind], dtype=object)
        self.ids = np.array([ids[e] for e in ind], dtype=object)

        # unique areas and components
        self.components, lbs = unique_components(self.components)
        self.ids = np.array([self.ids[np.where(lbs == e)[0][0]] for e in np.unique(lbs)], dtype=object)

        cs = []
        for i in range(len(self.ids)):
            idx = self.ids[i].ravel()[self.ids[i].ravel() != 0]
            pos = np.vstack([self.pos[e] for e in idx])
            lbs = np.hstack([self.lbs[e] for e in idx])
            cs.append(ConnectedCells(self.components[i], self.cells[idx], pos, lbs))
        return cs

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # ax.scatter(self.pts[:, 0], self.pts[:, 1], color='g', s=10)
        cs = ['C{}'.format(e) for e in range(10)]
        for i, e in enumerate(self.inds):
            if self.cell_lbs is None:
                fc = 'C0'
            else:
                fc = cs[self.cell_lbs[i] % 10]
            poly = Polygon(self.pts[e], alpha=0.5, ec='k', fc=fc)
            ax.add_patch(poly)
        ax.axis('equal')


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CellsImages
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def unique_components(components):
    def is_same(e1, item):
        e2 = np.rot90(e1, k=1)
        e3 = np.rot90(e1, k=2)
        e4 = np.rot90(e1, k=3)
        s1 = np.array_equal(e1, item)
        s2 = np.array_equal(e2, item)
        s3 = np.array_equal(e3, item)
        s4 = np.array_equal(e4, item)
        return np.any([s1, s2, s3, s4])

    # now this is a graph
    g = np.array([is_same(e1, e2) for e1 in components for e2 in components])
    g = g.reshape(len(components), len(components))
    num, lbs = connected_components(g)
    components_ = np.array([components[np.where(lbs == e)[0][0]] for e in np.unique(lbs)], dtype=object)
    return components_, lbs


mpl_21 = ['1', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#3397dc', '#ff993e', '#3fca3f', '#df5152', '#a985ca',
          '#ad7165', '#e992ce', '#999999', '#dbdc3c', '#35d8e9']


def add_grids(rgba, s=20):
    h, w = rgba.shape[0:2]

    ys = np.arange(0, h, 20)[1:]
    xs = np.arange(0, w, 20)[1:]

    d = 1
    for y in ys:
        rgba[y - d:y + d, ::] = [255, 255, 255, 255]

    for x in xs:
        rgba[:, x - d:x + d, :] = [255, 255, 255, 255]

    return rgba


def cells2points(cells, return_centers=False):
    xs, ys = np.nonzero(cells)
    centers = (np.vstack([xs, ys]).T).astype(float)
    p1 = centers.copy()
    p2 = centers.copy()
    p3 = centers.copy()
    p4 = centers.copy()
    p1[:, 0] += 0.5
    p1[:, 1] += 0.5
    p2[:, 0] -= 0.5
    p2[:, 1] -= 0.5
    p3[:, 0] += 0.5
    p3[:, 1] -= 0.5
    p4[:, 0] -= 0.5
    p4[:, 1] += 0.5
    pts = np.vstack([p1, p2, p3, p4])
    if return_centers:
        return np.unique(pts, axis=0), centers
    else:
        return np.unique(pts, axis=0)


def sort_inds_by_angle(pts, inds):
    inds_sort = []
    ns = []
    for ind in inds:
        pts1 = pts[ind]
        center = pts1.mean(axis=0)
        pts2 = pts1 - center
        angles = np.arctan2(pts2[:, 1], pts2[:, 0]) + np.pi
        ind_ = ind[np.argsort(angles)]
        inds_sort.append(ind_)
        ns.append(len(ind))
    if np.unique(ns).shape == (1,):
        return np.array(inds_sort)
    else:
        return inds_sort


def get_pairs(pts, centers):
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(pts)
    inds = nbrs.kneighbors(centers, return_distance=False)
    inds = sort_inds_by_angle(pts, inds)

    pairs = np.dstack([inds, np.roll(inds, 1, axis=1)]).reshape(-1, 2)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pairs, inds


class ConnectedCells:

    def __init__(self, data, cells=None, pos=None, lbs=None, tform=None, colors=None):
        self.mask = data > 0
        self.data = data * self.mask

        self.shape = data.shape
        self.h, self.w = data.shape
        self.num = np.sum(self.mask)

        if colors is None:
            self.colors = np.array([mc.to_rgba(e) for e in mpl_21])
        else:
            self.colors = colors

        # self.pxiels is rgba representation
        data_256 = rescale(self.data, scale=20, order=0).astype(int)
        data_256[data_256 < 0] = 0

        self.pixels = (self.colors[data_256] * 255).astype(np.uint8)
        # add grids
        self.pixels = add_grids(self.pixels)
        # make white pixel transparent
        alpha = ~np.all(self.pixels == 255, axis=2) * 255
        self.pixels[:, :, 3] = alpha

        self.cells = cells
        self.pos = pos
        self.lbs = lbs

    def _repr_png_(self):
        """Generate a PNG representation of the ConnectedComponents."""
        # data_256 = rescale(self.data, scale=20, order=0).astype(int)
        # data_256[data_256 < 0] = 0

        # pixels = (self.colors[data_256] * 255).astype(np.uint8)
        # code from matplotlib.colors
        png_bytes = io.BytesIO()
        title = 'level-{}'.format(self.num)
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        Image.fromarray(self.pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the ConnectedComponent."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')

        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.num}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.num} colormap" '
                f'title="{self.num}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {258}px; '
                'display: flex; justify-content: space-between;">')

    def rot90(self, k=1):
        return ConnectedCells(data=np.rot90(self.data, k=k))

    def _contains(self, c):
        def extract_patches(data, patch_shape=64, extraction_step=1):
            data_ndim = data.ndim
            # if patch_shape is a number, turn it into tuple
            if isinstance(patch_shape, numbers.Number):
                patch_shape = tuple([patch_shape] * data_ndim)

            patch_strides = data.strides

            # Extract all patches setting extraction_step to 1
            slices = tuple([slice(None, None, st) for st in (1, 1)])
            indexing_strides = data[slices].strides

            patch_indices_shape = (np.array(data.shape) - np.array(patch_shape)) + 1

            shape = tuple(list(patch_indices_shape) + list(patch_shape))
            strides = tuple(list(indexing_strides) + list(patch_strides))
            # Using strides and shape to get a 4d numpy array
            patches = as_strided(data, shape=shape, strides=strides)
            return patches

        if min(self.w, self.h) >= max(c.w, c.h):
            patches = extract_patches(self.data, c.shape)
            patches_ = patches * c.mask[np.newaxis, np.newaxis, :]
            s1 = ((patches_ == c.data).all(axis=(2, 3))).any()
            return s1
        else:
            return False

    def contains(self, c):
        c1 = c.rot90(k=1)
        c2 = c.rot90(k=2)
        c3 = c.rot90(k=3)

        s = self._contains(c)
        s1 = self._contains(c1)
        s2 = self._contains(c2)
        s3 = self._contains(c3)
        return np.any([s, s1, s2, s3])

    def plot(self, ax=None, colors=None, **kwargs):
        colors = colors_from_lbs(self.lbs, colors)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        for cell in self.cells:
            poly = Polygon(cell, ec='k', fc='none')
            ax.add_patch(poly)
        ax.scatter(self.pos[:, 0], self.pos[:, 1], c=colors, **kwargs)
        ax.axis('equal')