import numpy as np
from typing import List
from abc import ABC, abstractmethod
from scipy.stats import mode
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
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


class LatticeGraph(ShowGraphMixin, PlanarGraph):

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

    @property
    def is_loop(self):
        return np.all(self.degs==2)

    @property
    def is_chain(self):
        return sum(self.degs) == 2*(len(self.degs)-1)

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

class Motif(ShowGraphMixin, PlanarGraphBase):

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = symmetric_edges(edges)
        super().__post_init__()  # Enforces the attribute requirement.

    def __add__(self, other):
        # stack edge points
        edge_pts1 = self.pts[self.edges].reshape(-1, 2)
        edge_pts2 = other.pts[other.edges].reshape(-1, 2)

        edge_pts = np.vstack([edge_pts1, edge_pts2])
        # stack points and get unique nodes
        pts_ = np.unique(np.vstack([self.pts, other.pts]), axis=0)
        # matching edge points with the unique nodes
        ijs = np.empty(len(edge_pts))
        for i, e in enumerate(pts_):
            idx = (edge_pts == e).all(axis=1).nonzero()[0]
            ijs[idx] = i
        ijs = ijs.reshape(-1, 2).astype(int)
        ijs = np.unique(ijs, axis=0)
        return Motif(pts_, ijs)

    def to_grakel(self):
        pass

    def to_networkx(self):
        pass

    def to_image(self):
        pass


def find_n_nodes(ijs, n=2):
    matrix = edges2matrix(ijs)
    ijs = matrix2edges(matrix)
    lil = matrix2lil(matrix)
    for i in range(n-2):
        ijs = np.vstack([expand_nodes(row, lil) for row in ijs])
        mask = ijs[:, -1] != -1
        ijs = ijs[mask]
    # sort and unique in the final step
    # sometimes the order matters
    # ijs_ = np.sort(ijs, axis=1)
    _, idx = np.unique(ijs, axis=0, return_index=True)
    return ijs[idx]

def expand_nodes(nodes, lil):
    last_e = nodes[-1]
    nbrs_last_e = lil[last_e]
    valid_nodes = [e for e in nbrs_last_e if e not in nodes]
    if len(valid_nodes) > 0:
        nodes_ = np.array([nodes.tolist()+[e] for e in valid_nodes])
    else:
        nodes_ = np.array([nodes.tolist()+[-1]])
    return nodes_

def get_connected_components(matrix):
    n_components, lbs = connected_components(matrix, directed=False, return_labels=True)
    lbs = sort_lbs(lbs)
    return n_components, lbs


class MotifsGraph(ShowGraphMixin, PlanarGraphBase):

    def __init__(self, motifs, nodes, edges, lbs=None):
        self.nodes = nodes # nodes are positions of region centers
        self.edges = symmetric_edges(edges)
        self.motifs = motifs
        shape = (len(self.nodes), len(self.nodes))
        self.matrix = edges2matrix(self.edges, shape=shape)
        self.degs = np.array(np.sum(self.matrix, axis=1)).ravel()
        self.ks = np.array([len(motif.pts) for motif in self.motifs])
        # change [0][0] --> [0]
        self.major_k = mode(self.ks)[0]
        self.n_components, self.component_lbs = get_connected_components(self.matrix)
        self.lbs = lbs
        # Enforces the attribute requirement.
        super().__post_init__()

    def select(self, k=None):
        if k is None:
            k = [self.major_k, ]
        elif not np.iterable(k):
            k = [k,]

        mask = np.isin(self.ks, k)
        motifs_selected = self.motifs[mask]
        pts_ = self.pts[mask]
        matrix_ = lil_matrix(self.matrix)[mask, :][:, mask]
        ijs_ = matrix2edges(matrix_)
        return MotifsGraph(motifs_selected, pts_, ijs_)

    def find_n_nodes(self, n=3):
        return find_n_nodes(self.edges, n=n)

    def select_connections(self, k1k2):
        j = cantor_pairing(self.ks[self.ijs])
        q = cantor_pairing(k1k2)
        # this mask is for edges
        mask = np.isin(j, q)
        ijs_ = self.ijs[mask]
        return MotifsGraph(self.motifs, self.pts, ijs_)

    def remove_edges(self, mask):
        ijs_ = self.ijs[mask]
        return MotifsGraph(self.motifs, self.pts, ijs_)

    def select_nodes(self, mask=None, k=None):
        if mask is None:
            if k is None:
                k = [self.major_k, ]
            elif not np.iterable(k):
                k = [k, ]
            mask = np.isin(self.ks, k)

        motifs_selected = self.motifs[mask]
        pts_ = self.pts[mask]
        matrix_ = lil_matrix(self.matrix)[mask, :][:, mask]
        ijs_ = matrix2edges(matrix_)
        if self.lbs is not None:
            lbs_ = self.lbs[mask]
        else:
            lbs_ = self.lbs
        return MotifsGraph(motifs_selected, pts_, ijs_, lbs_)

