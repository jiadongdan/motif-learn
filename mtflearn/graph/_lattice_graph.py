import numpy as np
from typing import List
from abc import ABC, abstractmethod
from scipy.stats import mode
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from skimage.draw import polygon2mask
from .find_regions import find_regions
from .mixin_class import SaveGraphMixin, ShowGraphMixin
from .utils import edges2matrix, matrix2edges, matrix2lil

def sort_lbs(lbs):
    unique_lbs, counts = np.unique(lbs, return_counts=True)
    idx = np.argsort(counts)[::-1]
    unique_lbs = unique_lbs[idx]
    lbs_order = range(len(unique_lbs))
    order_dict = dict(zip(unique_lbs, lbs_order))
    lbs_ = np.vectorize(order_dict.get)(lbs)
    return lbs_

def symmetric_edges(edges):
    ijs = np.vstack([edges, np.fliplr(edges)])
    return np.unique(ijs, axis=0)


def cantor_pairing(ij, symmetric=True):
    ij = np.array(ij).reshape(-1, 2)
    if symmetric:
        i = ij.min(axis=1)
        j = ij.max(axis=1)
    else:
        i = ij[:, 0]
        j = ij[:, 1]
    # p = (i+j-2)*(i+j-1)//2 + i
    p = (i + j) * (i + j + 1) // 2 + j
    return p

def _get_regions_graph_edges(regions):
    ijq = np.vstack([np.array([region.astype(int), np.roll(region.astype(int), -1), [i] * len(region)]).T for i, region in enumerate(regions)])
    p = cantor_pairing(ijq[:, 0:2])
    _, lbs, cnts = np.unique(p, return_counts=True, return_inverse=True)
    cnt2 = np.where(cnts == 2)[0]
    mask = np.isin(lbs, cnt2)
    lbsq = np.array([lbs, ijq[:, -1]]).T
    lbsq = lbsq[mask]
    ind = np.argsort(lbsq[:, 0])
    lbsq = lbsq[ind]
    ij1 = np.array(np.split(lbsq[:, -1], len(lbsq) // 2))
    return ij1

def construct_motif(pts):
    def get_edges(n):
        js = np.roll(np.arange(n), 1)
        ijs = [(i, j) for i, j in zip(range(n), js)] + [(j, i) for i, j in zip(range(n), js)]
        return np.array(ijs)

    return Motif(pts, get_edges(len(pts)))


class PlanarGraphBase(ABC):

    # alias for some property names
    aliases = {
                  'polys': 'regions',
                  'faces': 'regions',
                  'polygons': 'regions',
                  'pts': 'nodes',
                  'vertices': 'nodes',
                  'ijs': 'edges',}


    """An abstract class with required attributes.

        Attributes:
            nodes: 2D points of node positions.
            edges: An (unweighted) edge defined by its start and end point indices.
            matrix: Adjacency matrix in sparse format.
    """

    @abstractmethod
    def __init__(self):
        """Forces you to implement __init__ in its subclass.
        Make sure to call __post_init__() from inside its subclass."""

    # https://stackoverflow.com/a/68794377/5855131
    def __post_init__(self):
        self._has_required_attributes()
        # You can also type check here if you want.

    def _has_required_attributes(self):
        req_attrs: List[str] = ['nodes', 'edges',]
        for attr in req_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing attribute: '{attr}'")

    def __setattr__(self, name, value):
        name = self.aliases.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.aliases.get(name, name)
        return object.__getattribute__(self, name)


class PlanarGraph(SaveGraphMixin, PlanarGraphBase):

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = symmetric_edges(edges)
        self._matrix = None
        self._lil = None
        self._degs = None
        super().__post_init__()  # Enforces the attribute requirement.

    @property
    def matrix(self):
        if self._matrix is None:
            shape = (len(self.nodes), len(self.nodes))
            self._matrix = edges2matrix(self.edges, shape=shape)
        return self._matrix

    @property
    def lil(self):
        if self._lil is None:
            self._lil = matrix2lil(self.matrix)
        return self._lil

    @property
    def degs(self):
        if self._degs is None:
            self._degs = np.array(np.sum(self.matrix, axis=1)).ravel()
        return self._degs

    def is_symmetric(self, tol=1e-8):
        return np.all(np.abs(self.matrix - self.matrix.T) < tol)


class LatticeGraph1(ShowGraphMixin, PlanarGraph):

    def __init__(self, nodes, edges, img=None, lbs=None):
        super().__init__(nodes, edges)

        self.img = img
        if lbs is None:
            self.lbs = self.degs
        else:
            self.lbs = lbs

        # polygons
        self._regions = None
        self._centers = None
        self._ks = None
        super().__post_init__()  # Enforces the attribute requirement.


    @property
    def regions(self):
        if self._regions is None:
            self._regions = find_regions(self.nodes, self.edges, return_dict=False)
        return self._regions

    @property
    def centers(self):
        if self._centers is None:
            self._centers = np.array([self.nodes[region.astype(int)].mean(axis=0) for region in self.regions])
        return self._centers

    @property
    def ks(self):
        if self._ks is None:
            self._ks = np.array([len(regions) for regions in self.regions])
        return self._ks

    def get_node_motifs(self):
        ll = []
        for i in range(len(self.nodes)):
            l = []
            for region in self.regions:
                if i in region:
                    l.append(len(region))
            ll.append(l)
        return ll

    def to_motifs_graph(self):
        motifs = np.array([construct_motif(self.pts[region.astype(int)]) for region in self.regions], dtype=object)
        ijs = _get_regions_graph_edges(self.regions)
        return MotifsGraph(motifs, self.centers, ijs)

    def get_level1(self):
        return self.lbs

    def get_level2(self):
        return [self.lbs[row] for row in self.lil]

    def decompose(self, min_nodes=4):
        n_components, lbs = connected_components(self.matrix, directed=False)
        mask = np.array([(lbs == i).sum() for i in range(n_components)]) >= min_nodes
        lbs_valid = np.arange(n_components)[mask]
        gs = []
        for e in lbs_valid:
            mask = (lbs == e)
            nodes_ = self.nodes[mask]
            matrix_ = lil_matrix(self.matrix)[mask, :][:, mask]
            ijs_ = matrix2edges(matrix_)
            if self.lbs is not None:
                lbs_ = self.lbs[mask]
            else:
                lbs_ = self.lbs
            g = LatticeGraph(nodes_, ijs_, lbs_)
            gs.append(g)
        return gs

    def remove_nodes(self, mask):
        pts_ = self.pts[mask]
        matrix_ = lil_matrix(self.matrix)[mask, :][:, mask]
        ijs_ = matrix2edges(matrix_)
        if self.lbs is not None:
            lbs_ = self.lbs[mask]
        else:
            lbs_ = None
        return LatticeGraph(pts_, ijs_, lbs_)

    def get_polygon_masks(self, img_shape, polygon=5):
        masks = []
        for region in self.regions:
            if len(region) == polygon:
                polygon_xy = self.nodes[region]
                polygon_yx = np.fliplr(polygon_xy)
                mask = polygon2mask(image_shape=img_shape, polygon=polygon_yx)
                masks.append(mask)
        return np.array(masks)
