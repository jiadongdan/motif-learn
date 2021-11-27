import numpy as np
from time import time
import numba

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import sparse
from scipy.optimize import curve_fit
from sklearn.utils import check_random_state

MACHINE_EPSILON = np.finfo(np.double).eps


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


def init_layout(X, graph, random_state, dim=2, init_mode='pca'):
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
    node_dtype = np.dtype([('x', np.float64), ('y', np.float64)])
    num_nodes = init_xy.shape[0]
    # Convert to a list of tuples
    nodes = [tuple(n) for n in init_xy]
    # Convert to structured array
    nodes = np.array(nodes, dtype=node_dtype)
    return nodes


def compute_pairs(graph):
    pair_dtype = np.dtype([('node1', np.int), ('node2', np.int),
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
def apply_attraction_force(node1, node2, alpha, lr, weight, strength):
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y
    distance = np.hypot(x_dist, y_dist)
    force = strength / (distance ** (alpha) + 1)

    node1.x -= clip(x_dist * force) * lr * weight
    node1.y -= clip(y_dist * force) * lr * weight
    node2.x += clip(x_dist * force) * lr * weight
    node2.y += clip(y_dist * force) * lr * weight


@numba.njit(fastmath=True)
def apply_repulsion_force(node1, node2, alpha, lr, weight, strength):
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y
    distance = np.hypot(x_dist, y_dist)
    force = strength / (distance ** alpha + 1)

    node1.x += clip(x_dist * force) * lr * weight
    node1.y += clip(y_dist * force) * lr * weight
    node2.x -= clip(x_dist * force) * lr * weight
    node2.y -= clip(y_dist * force) * lr * weight


# helper functions
@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


# Optimize layout using three settings
@numba.njit(fastmath=True)
def optimize_stage(num_iterations, nodes, pairs, alphas, num_negative_samples, nbrs_ind, learning_rate,
                    logs, strength_ratio):
    num_nodes = len(nodes)
    lr = learning_rate
    alpha1 = alphas[0]
    alpha2 = alphas[1]
    s1 = 1.0
    s2 = s1*strength_ratio

    # logs = []
    xy = np.stack((nodes.x, nodes.y)).T
    logs.append(xy)

    for n in range(0, num_iterations):
        for pair in pairs:
            weight = pair.weight
            node1 = nodes[pair.node1]
            node2 = nodes[pair.node2]
            apply_attraction_force(node1, node2, alpha1, lr, weight, s1)
            valid_negative_ind = nbrs_ind[pair.node1]
            for i in range(num_negative_samples):
                rand_ind = np.random.randint(low=0, high=num_nodes)
                do_repulsion = True
                for ind in nbrs_ind[pair.node1]:
                    if ind == rand_ind:
                        do_repulsion = False
                if do_repulsion:
                    negative_node = nodes[rand_ind]
                    apply_repulsion_force(node1, negative_node, alpha2, lr, weight, s2)
        lr = learning_rate * (1.0 - (float(n) / float(num_iterations)))

        xy = np.stack((nodes.x, nodes.y)).T
        logs.append(xy)
    return logs


@numba.njit(fastmath=True)
def optimize_layout(num_iterations, nodes, pairs, num_negative_samples, nbrs_ind, learning_rate, settings, divide):
    lr = learning_rate

    logs = []
    xy = np.stack((nodes.x, nodes.y)).T
    logs.append(xy)

    alphas = settings[0: 2]
    s_ratio = settings[2]
    # n_iter = num_iterations // divide
    n_iter = int(num_iterations * divide)
    logs = optimize_stage(n_iter, nodes, pairs, alphas, num_negative_samples, nbrs_ind, lr, logs, s_ratio)
    alphas = settings[3: 5]
    s_ratio = settings[5]
    n_iter = int(num_iterations * (1 - divide))
    logs = optimize_stage(n_iter, nodes, pairs, alphas, num_negative_samples, nbrs_ind, lr, logs, s_ratio)
    return logs


class ForceGraph4:
    def __init__(self,
                 X=None,
                 # Graph related params
                 n_neighbors=30,
                 metric='correlation',
                 local_connectivity=1,

                 # Initial layout
                 random_state=42,
                 init_mode='pca',

                 # Repusion force params
                 num_negative_samples=6,

                 # Attraction force params
                 edge_weight_influence=1.0,

                 # Adjust convergence speed
                 learning_rate=1.0,

                 # Tuning
                 num_iterations=100,
                 setting=[0, 2, 5, 1],
                 divide=3,
                 ):
        # self.graph = check_array(graph, accept_sparse=True)
        self.X = X
        # Graph related params
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.local_connectivity = local_connectivity

        # Initial layout params
        self.random_state = check_random_state(random_state)
        if self.X is None:
            self.init_mode = 'random'
        else:
            self.init_mode = init_mode

        # Forces params --> Three types of forces
        self.edge_weight_influence = edge_weight_influence
        self.num_negative_samples = num_negative_samples

        # Adjust convergence
        self.learning_rate = learning_rate

        self.num_iterations = num_iterations

        # Params initialized when fit
        self.graph = None
        self.pts = None
        self.nodes = None
        self.pairs = None
        self.logs = []
        self.setting = np.array(setting)
        self.divide = divide

    def fit(self, X):
        self.graph, self.nbrs_ind = compute_graph(X, self.n_neighbors, self.metric, None, self.local_connectivity, 1.0)
        # Initialize layout
        self.pts = init_layout(X, graph=self.graph, random_state=self.random_state, dim=2, init_mode=self.init_mode)
        # Compute nodes and edges
        print('Construct nodes...', end='')
        t0 = time()
        self.nodes = compute_nodes(self.pts)
        print('done in %.2fs.' % (time() - t0))

        print('Construct pairs...', end='')
        t0 = time()
        self.pairs = compute_pairs(self.graph)
        print('done in %.2fs.' % (time() - t0))
        # Optimize layout
        self.logs = optimize_layout(self.num_iterations, self.nodes, self.pairs,
                                    self.num_negative_samples, self.nbrs_ind, self.learning_rate, self.setting,
                                    self.divide)
        return np.array([(node['x'], node['y']) for node in self.nodes])
