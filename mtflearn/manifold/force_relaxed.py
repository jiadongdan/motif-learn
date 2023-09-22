import numba
import numpy as np
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin, BaseEstimator


MACHINE_EPSILON = np.finfo(np.double).eps

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def calculate_asymmetric_Pij(dist_nn, perplexity=30, local_conectivity=1):
    rho = dist_nn[:, local_conectivity][:, np.newaxis]
    d_ = dist_nn - rho
    d_[d_ < 0] = 0
    tolerance = 1e-5
    target = np.log2(perplexity)
    n_steps = 100
    beta_list = []
    for row in d_:
        beta_min = 0.0
        beta_max = np.inf
        beta = 1.0
        for n in np.arange(n_steps):
            # Use row[1:], sum is applied from second element
            # Because first element is itself
            sum_Pi = np.exp(-row[1:] * beta).sum()
            if np.abs(sum_Pi - target) < tolerance:
                beta_list.append(beta)
                break

            if sum_Pi - target > 0:
                # beta should increase
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2.0
            if n == (n_steps - 1):
                beta_list.append(beta)
    beta_list = np.array(beta_list)[:, np.newaxis]
    P_ij = np.exp(-d_ * beta_list)
    P_ij[P_ij < MACHINE_EPSILON] = MACHINE_EPSILON
    P_ij[:, 0] = 0.0
    return P_ij


def calculate_graph(Pij, ind, set_op_mix_ratio=1.0, verbose=1):
    if verbose:
        print('Construct graph from data...')
    n_samples, k = Pij.shape
    P = sparse.csr_matrix((Pij.ravel(), ind.ravel(),
                           range(0, n_samples * k + 1, k)),
                          shape=(n_samples, n_samples))
    prod = P.multiply(P.T)
    P = set_op_mix_ratio * (P + P.T - prod) + (1 - set_op_mix_ratio) * prod
    return P


def compute_graph(X, n_neighbors, metric, perplexity=None, local_connectivity=1, set_op_mix_ratio=1.0):
    if perplexity == None:
        perplexity = n_neighbors
    knn = NearestNeighbors(algorithm='auto', n_neighbors=n_neighbors, metric=metric)
    knn.fit(X)
    d, ind = knn.kneighbors(X, n_neighbors=n_neighbors)

    # Do binary search for rho, and calculate asymmetric weights Pij
    P_ij = calculate_asymmetric_Pij(dist_nn=d,
                                    perplexity=perplexity,
                                    local_conectivity=local_connectivity)
    # Calculate symmetric weights,
    # and return a sparse matrix with shape of (n_samples, n_samples)
    P = calculate_graph(Pij=P_ij,
                        ind=ind,
                        set_op_mix_ratio=set_op_mix_ratio)
    # Here P is a sparse matrix in csr format
    return P, ind


def init_layout(X, random_state, dim=2, init_mode='pca'):
    """
    Initialize the two-dimensional layout
    Parameters
    ----------
    X: array,
        the high dimensional data to be embedded.
    random_state: int

    dim: the dimension of the low dimensional layout, default: 2
    init_mode: 'pca'|'random'

    Returns
    -------
    Y: array
        the low dimensional layout
    """
    if init_mode == 'random':
        print('Initialize {}-d embedding using random layout...'.format(dim))
        random_state = check_random_state(random_state)
        X_random = random_state.uniform(low=-10.0, high=10.0, size=(X.shape[0], dim))
        return X_random
    if init_mode == 'pca':
        print('Initialize {}-d embedding using PCA layout...'.format(dim))
        pca = PCA(n_components=dim)
        X_pca = pca.fit_transform(X)
        X_pca = X_pca / np.abs(X_pca).max() * 10
        return X_pca


def compute_nodes(init_xy):
    """
    Compute nodes (structured array) from initial layout

    Parameters
    ----------
    init_xy: array
        the initial layout

    Returns
    -------
    nodes: structured array

    Examples
    --------
    >>> xy = [(0., 0.), (1., 1.)]
    >>> node_dtype = np.dtype([('x', np.float64), ('y', np.float64)])
    >>> # Convert to structured array
    >>> nodes = np.array(xy, dtype=node_dtype)
    >>> # access x and y
    >>> nodes['x']
    >>> nodes['y']
    """
    node_dtype = np.dtype([('x', np.float64), ('y', np.float64)])
    num_nodes = init_xy.shape[0]
    # Convert to a list of tuples
    nodes = [tuple(n) for n in init_xy]
    # Convert to structured array
    nodes = np.array(nodes, dtype=node_dtype)
    return nodes


def compute_pairs(graph):
    """
    Compute pairs from nonzero entries of the graph

    Parameters
    ----------
    graph

    Returns
    -------

    """
    # np.int is deprecated
    pair_dtype = np.dtype([('node1', int), ('node2', int),
                           ('weight', np.float64)])
    if sparse.issparse(graph):
        g = graph.tocoo()
        pairs = [(i, j, w) for i, j, w in zip(g.row, g.col, g.data)]
    else:
        ij = np.asarray(graph.nonzero()).T
        # edges is a list of tuples
        pairs = [(i, j, graph[i, j]) for (i, j) in ij]
    pairs = np.array(pairs, dtype=pair_dtype)
    return pairs


@numba.njit(fastmath=True)
def apply_attraction_force(node1, node2, lr, weight, n, strength):
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y
    distance = np.hypot(x_dist, y_dist)
    force = strength / (distance ** n + 1)

    node1.x -= clip(x_dist * force) * lr * weight
    node1.y -= clip(y_dist * force) * lr * weight
    node2.x += clip(x_dist * force) * lr * weight
    node2.y += clip(y_dist * force) * lr * weight


@numba.njit(fastmath=True)
def apply_repulsion_force(node1, node2, lr, weight, m, strength):
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y
    distance = np.hypot(x_dist, y_dist)
    force = strength / (distance ** m + 1)

    node1.x += clip(x_dist * force) * lr
    node1.y += clip(y_dist * force) * lr
    node2.x -= clip(x_dist * force) * lr
    node2.y -= clip(y_dist * force) * lr


# helper functions
@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        the internal state of the rng
    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]

# Optimize layout using three settings
@numba.njit(fastmath=True, parallel=False)
def optimize_stage(num_iterations, nodes, pairs, force_params, num_negative_samples, nbrs_ind, learning_rate, rng_states,logs):
    num_nodes = len(nodes)
    lr = learning_rate
    N, M, alpha, beta = force_params # don't use n as it has been used

    # logs = []
    xy = np.stack((nodes.x, nodes.y)).T
    logs.append(xy)

    for n in range(0, num_iterations):
        for pair in pairs:
            weight = pair.weight
            node1 = nodes[pair.node1]
            node2 = nodes[pair.node2]
            apply_attraction_force(node1, node2, lr, weight, N, alpha)
            valid_negative_ind = nbrs_ind[pair.node1]
            for i in range(num_negative_samples):
                rand_ind = tau_rand_int(rng_states) % num_nodes
                #rand_ind = np.random.randint(low=0, high=num_nodes)
                do_repulsion = True
                for ind in nbrs_ind[pair.node1]:
                    if ind == rand_ind:
                        do_repulsion = False
                if do_repulsion:
                    negative_node = nodes[rand_ind]
                    apply_repulsion_force(node1, negative_node, lr, weight, M, beta)
        lr = learning_rate * (1.0 - (float(n) / float(num_iterations)))

        xy = np.stack((nodes.x, nodes.y)).T
        logs.append(xy)
    return logs


@numba.njit(fastmath=True, parallel=False)
def optimize_layout(num_iterations, nodes, pairs, num_negative_samples, nbrs_ind, learning_rate, force_params1, force_params2, rng_states, divide):
    lr = learning_rate
    logs = []
    xy = np.stack((nodes.x, nodes.y)).T
    logs.append(xy)
    # n_iter = num_iterations // divide
    n_iter = int(num_iterations * divide)
    logs = optimize_stage(n_iter, nodes, pairs, force_params1, num_negative_samples, nbrs_ind, lr, rng_states, logs)
    n_iter = int(num_iterations * (1 - divide))
    logs = optimize_stage(n_iter, nodes, pairs, force_params2, num_negative_samples, nbrs_ind, lr, rng_states, logs)
    return logs


class ForceGraph8(TransformerMixin, BaseEstimator):
    def __init__(self,
                 X=None,
                 # graph related params
                 n_neighbors=10,
                 metric='correlation',
                 local_connectivity=1,

                 # initial layout
                 random_state=48,
                 init_mode='pca',

                 # force params
                 num_negative_samples=10,

                 # attraction force params
                 edge_weight_influence=1.0,

                 # adjust convergence speed
                 learning_rate=1.0,

                 # tuning
                 num_iterations=100,
                 force_params1=(0, 2, 1, 1),
                 force_params2=(2, 4, 5, 2),
                 divide=0.5,
                 verbose=False
                 ):
        # Input data
        self.X = X
        # Graph related params
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.local_connectivity = local_connectivity

        # Initial layout params
        self.random_state = check_random_state(random_state)
        if init_mode is None:
            self.init_mode = 'random'
        else:
            self.init_mode = init_mode

        # forces params
        self.edge_weight_influence = edge_weight_influence
        self.num_negative_samples = num_negative_samples
        self.force_params1 = np.array(force_params1)
        self.force_params2 = np.array(force_params2)

        # adjust convergence
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # tuning fraction of two stages
        self.divide = divide
        self.verbose = verbose


        # params initialized when fit
        self.graph = None
        self.pts = None
        self.nodes = None
        self.pairs = None
        self.y = None
        self.logs = []

    def fit(self, X, y=None):
        # generate the graph
        self.graph, self.nbrs_ind = compute_graph(X, self.n_neighbors, self.metric, None, self.local_connectivity, 1.0)
        # initialize layout
        self.pts = init_layout(X, random_state=self.random_state, dim=2, init_mode=self.init_mode)
        # compute nodes and edges
        self.nodes = compute_nodes(self.pts)
        self.pairs = compute_pairs(self.graph)
        # the internal random states
        self.rng_states = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        # Optimize layout
        self.logs = optimize_layout(self.num_iterations, self.nodes, self.pairs,
                                    self.num_negative_samples, self.nbrs_ind, self.learning_rate, self.force_params1,
                                    self.force_params2, self.rng_states, self.divide)
        self.y = np.array([(node['x'], node['y']) for node in self.nodes])

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.y