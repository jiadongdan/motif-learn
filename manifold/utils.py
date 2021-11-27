from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors


import numpy as np

MACHINE_EPSILON = np.finfo(np.double).eps

def pairwise_dist(X, metric='euclidean'):
    # Returned matrix is condensed matrix
    d_matrix = pdist(X, metric=metric)
    # Converts a vector-form distance vector to a square-form distance matrix
    return squareform(d_matrix)

def init_spectral_layout(graph, embedding_dim=2, verbose=1):
    if verbose:
        print('Initialize {}-d embedding using spectral layout...'.format(embedding_dim))
    n_samples = graph.shape[0]
    n_components, labels = sparse.csgraph.connected_components(graph)
    assert(n_components == 1)
    diag_data = np.asarray(graph.sum(axis=0))
    I = sparse.identity(graph.shape[0], dtype=np.float64)
    D = sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D
    k = embedding_dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))

    eigenvalues, eigenvectors = sparse.linalg.eigsh(
        L,
        k,
        which="SM",
        ncv=num_lanczos_vectors,
        tol=1e-4,
        v0=np.ones(L.shape[0]),
        maxiter=graph.shape[0] * 5)
    order = np.argsort(eigenvalues)[1:k]
    init_embedding = eigenvectors[:, order]
    expansion = 10.0 / init_embedding.max()
    init_embedding = init_embedding * expansion
    return init_embedding


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
        return X_pca
    if init_mode == 'spectral':
        return init_spectral_layout(graph, 2)

def compute_nodes(graph, init_xy, mass=1.0, size=1.0):
    node_dtype = np.dtype([('x', np.float64), ('y', np.float64),
                       ('dx', np.float64), ('dy', np.float64),
                       ('old_dx', np.float64), ('old_dy', np.float64),
                       ('mass', np.float64), ('size', np.float64)])
    num_nodes = graph.shape[0]
    if mass is None:
        if sparse.issparse(graph):
            graph = graph.tolil()
            mass = np.array([len(graph.rows[i])+1 for i in range(num_nodes)])
        else:
            mass = np.count_nonzero(graph, axis=0)
    nodes = np.zeros((num_nodes, 8))
    nodes[:, 0:2] = init_xy
    nodes[:, 6] = mass
    if size is not None:
        nodes[:, 7] = size
    # Convert to a list of tuples
    nodes = [tuple(n) for n in nodes]
    # Convert to structured array
    nodes = np.array(nodes, dtype=node_dtype)
    return nodes

def compute_edges(graph):
    edge_dtype = np.dtype([('node1', np.int), ('node2', np.int),
                           ('weight', np.float64)])
    if sparse.issparse(graph):
        g = graph.tocoo()
        edges = [(i, j, w) for i, j, w in zip(g.row, g.col, g.data) if i < j]
    else:
        ij = np.asarray(graph.nonzero()).T
        # edges is a list of tuples
        edges = [(i, j, graph[i, j]) for (i, j) in ij if i < j]
    edges = np.array(edges, dtype=edge_dtype)
    return edges


def calculate_asymmetric_Pij(dist_nn, perplexity=30, local_conectivity=1):
    rho = dist_nn[:, local_conectivity][:, np.newaxis]
    d_ = dist_nn - rho
    d_[d_<0]=0
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
            sum_Pi = np.exp(-row[1:]*beta).sum()
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
            if n == (n_steps-1):
                beta_list.append(beta)
    beta_list = np.array(beta_list)[:, np.newaxis]
    P_ij = np.exp(-d_*beta_list)
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
    P = set_op_mix_ratio*(P+P.T-prod)+(1-set_op_mix_ratio)*prod
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
    return P



