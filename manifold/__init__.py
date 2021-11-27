from .umap import UMAP
from .force_graph_numba import ForceGraph
from .force_graph_new import ForceGraph2
from .multistage_force import ForceGraph3
#from .utils import compute_graph
#from .utils import compute_nodes
from .utils import compute_edges

from .force_graph4 import ForceGraph4
from .force_graph8 import compute_graph
from .force_graph8 import compute_nodes
from .force_graph8 import compute_pairs


from .force_graph6 import ForceGraph6
from .force_graph7 import ForceGraph7
from .force_graph8 import ForceGraph8

import numba
numba.config.THREADING_LAYER = 'workqueue'

__all__ = ['UMAP',
           'ForceGraph',
           'compute_graph',
           'compute_nodes',
           'compute_edges',
           'ForceGraph2',
           'ForceGraph3',
           'ForceGraph4',
           'ForceGraph6',
           'ForceGraph7',
           'ForceGraph8']