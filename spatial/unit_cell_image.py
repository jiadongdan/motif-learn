import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.path import Path
from matplotlib.patches import Polygon
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import inv
from skimage.measure import label, regionprops

import base64
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from skimage.transform import rescale
from scipy.sparse.csgraph import connected_components

import numbers
from numpy.lib.stride_tricks import as_strided



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


def points_to_regions(pts, threshold=None):
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


# propagate single step
def _propagate_pts(pts, pts_, u, v, threshold=None):
    pts1 = pts_ + u
    pts2 = pts_ + v
    pts3 = pts_ + u + v
    pts4 = pts_ - u
    pts5 = pts_ - v
    pts6 = pts_ - u - v
    pts7 = pts_ - u + v
    pts8 = pts_ + u - v
    pp = np.vstack([pts_, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pp)
    pp_ = pts[ind[d < threshold]]
    return np.unique(pp_, axis=0)


def propagate_pts(pts, pts_, u, v, threshold=None):
    if threshold is None:
        r0 = _estimate_r0(pts)
        threshold = 0.2 * r0

    if len(pts_.shape) == 1:
        num_pts = 1
    else:
        num_pts = pts_.shape[0]
    do_propagate = True
    while do_propagate:
        pts_ = _propagate_pts(pts, pts_, u, v, threshold)
        if pts_.shape[0] > num_pts:
            num_pts = pts_.shape[0]
        else:
            do_propagate = False
    return pts_


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


def reorder_lbs(lbs):
    unique, counts = np.unique(lbs, return_counts=True)
    lbs_order = np.argsort(counts)[::-1]
    order_dict = dict(zip(lbs_order, unique))
    lbs = np.vectorize(order_dict.get)(lbs)
    return lbs


class HexCells:

    def __init__(self, pts, dx=0, dy=0, lbs=None):
        dxdy = np.array([dx, dy])
        self.pts = pts + dxdy
        self.p0, self.u, self.v = _get_p0_u_v(self.pts)

        # indices of pts, shape (num_cells, 4), counterclockwise
        self.regions = points_to_regions(self.pts)

        # region(cell) centers
        self.centers = np.array([self.pts[region].mean(axis=0) for region in self.regions])

        self.lbs = lbs

    def to_image(self):
        p0, u, v = _get_p0_u_v(self.centers)
        uv = np.vstack([u, v])
        ij = (self.centers - p0).dot(inv(uv))
        ij = np.round(ij).astype(int)
        vmax = max(np.ptp(ij[:, 0]), np.ptp(ij[:, 1]))
        ij += [vmax // 2, vmax // 2]
        cells_matrix = np.ones(shape=(vmax + 1, vmax + 1)) * (-1)
        for idx, (i, j) in enumerate(ij):
            cells_matrix[j, i] = self.lbs[idx]
        return CellsImage(cells_matrix)

    # merge small cells, this is a key method
    def merge(self, n=2):
        U = n * self.u
        V = n * self.v
        pts_new = propagate_pts(self.pts, self.p0, U, V, threshold=None)
        if self.lbs is None:
            new_hexcells = HexCells(pts_new)
        else:
            new_hexcells = HexCells(pts_new)
            pos = self.pts[self.regions].mean(axis=1)
            new_hexcells.set_cell_lbs(pos, self.lbs)
        return new_hexcells

    def set_cell_lbs(self, pos, lbs):
        polys = self.pts[self.regions]
        # indices of points inside each cell
        inds = []
        for poly in polys:
            ind = np.where(Path(poly).contains_points(pos) == 1)[0]
            inds.append(ind)
        lbs_ordered = get_ordered_lbs(pos, lbs, inds)
        lbs_ordered_unique, self.lbs = np.unique(lbs_ordered, axis=0, return_counts=False, return_inverse=True)
        self.lbs = reorder_lbs(self.lbs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # ax.scatter(self.pts[:, 0], self.pts[:, 1], color='g', s=10)
        cs = ['C{}'.format(e) for e in range(10)]
        for i, e in enumerate(self.regions):
            if self.lbs is None:
                fc = 'C0'
            else:
                fc = cs[self.lbs[i] % 10]
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


class CellsImage:

    def __init__(self, image, threshold=0, connectivity=1):
        # image data representing the cells
        self.image = image

        # label image
        self.lbs_image = label(self.image > threshold, connectivity=connectivity)
        # regions
        self.region_props = regionprops(self.lbs_image)

        # areas and components
        areas = np.array([e.area for e in self.region_props if e.area > 1])
        components = [self.image[e.slice] for e in self.region_props if e.area > 1]
        ind = np.argsort(areas)
        self.areas = areas[ind]
        self.components = np.array([components[e] for e in ind], dtype=object)
        # unique areas and components
        components_, self.lbs = unique_components(self.components)
        self.areas_ = np.array([self.areas[np.where(self.lbs == e)[0][0]] for e in np.unique(self.lbs)])
        self.components_ = [ConnectedCells(data=e) for e in components_]




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

class ConnectedCells:

    def __init__(self, data, colors=None):
        self.mask = data > 0
        self.data = data * self.mask

        self.shape = data.shape
        self.w, self.h = data.shape
        self.num = np.sum(self.mask)

        if colors is None:
            self.colors = np.array([mc.to_rgba(e) for e in mpl_21])
        else:
            self.colors = colors

        data_256 = rescale(self.data, scale=20, order=0).astype(int)
        data_256[data_256 < 0] = 0

        self.pixels = (self.colors[data_256] * 255).astype(np.uint8)
        # add grids
        self.pixels = add_grids(self.pixels)
        # make white pixel transparent
        alpha = ~np.all(self.pixels == 255, axis=2) * 255
        self.pixels[:, :, 3] = alpha

    def _repr_png_(self):
        """Generate a PNG representation of the ConnectedComponents."""
        #data_256 = rescale(self.data, scale=20, order=0).astype(int)
        #data_256[data_256 < 0] = 0

        #pixels = (self.colors[data_256] * 255).astype(np.uint8)
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


class HTree:

    def __init__(self, motifs: list):
        self.root = None

        self.levels, cnts = np.unique([motif.num for motif in motifs], return_counts=True)
        self.full_graph, self.n_array = _construct_full_graph(motifs)
        self.graph = _truncate_graph(self.full_graph, self.n_array)
        self.pos = _nodes_pos(self.levels, cnts)
        self.motifs = [motif.pixels for motif in motifs]

    def plot(self, ax=None, zoom=1):
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
        for ii, (x, y) in enumerate(self.pos):
            piesize = 0.04 * ss[ii]  # this is the image size
            p2 = piesize / 2.0
            xx, yy = trans1([x, y])  # data --> pixel
            xa, ya = trans2((xx, yy))  # pixel --> figure fraction
            a = fig.add_axes([xa - p2, ya - p2, piesize, piesize])
            axes.append(a)
            a.set_aspect('equal')
            a.imshow(self.motifs[ii])
            a.xaxis.set_ticklabels([])
            a.yaxis.set_ticklabels([])
            a.set_xticks([])
            a.set_yticks([])
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