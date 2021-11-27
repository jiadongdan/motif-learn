import numba
import scipy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from time import time

MACHINE_EPSILON = np.finfo(np.double).eps
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# Part 1:
# Functions - How to generate the weighted graph
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

def get_knn_info(X, n_neighbors, metric='correlation', verbose=True):
    n_samples = X.shape[0]
    knn = NearestNeighbors(algorithm='auto', n_neighbors=n_neighbors, metric=metric)
    t0 = time()
    knn.fit(X)
    duration = time() - t0
    if verbose:
        print("KNN: Indexed {} samples in {:.6f}s...".format(n_samples, duration))
    d, ind = knn.kneighbors(X, n_neighbors=n_neighbors)
    return d, ind

def calculate_asymmetric_Pij(dist_nn, perplexity=30, local_conectivity=1, verbose=True):
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
    P = csr_matrix((Pij.ravel(), ind.ravel(),
                    range(0, n_samples * k + 1, k)),
                   shape=(n_samples, n_samples))
    prod = P.multiply(P.T)
    P = set_op_mix_ratio*(P+P.T-prod)+(1-set_op_mix_ratio)*prod
    return P

def contruct_graph_from_X(X, n_neighbors,
                          metric='correlation',
                          local_connectivity=1,
                          perplexity=None,
                          set_op_mix_ratio=1.0,
                          verbose=True):

    if perplexity == None:
        perplexity = n_neighbors
    # Calculate KNN
    d, ind = get_knn_info(X,n_neighbors=n_neighbors,
                           metric=metric,
                           verbose=verbose)
    # Do binary search for rho, and calculate asymmetric weights Pij
    P_ij = calculate_asymmetric_Pij(dist_nn = d,
                                    perplexity=perplexity,
                                    local_conectivity=local_connectivity,
                                    verbose=verbose)
    # Calculate symmetric weights,
    # and return a sparse matrix with shape of (n_samples, n_samples)
    P = calculate_graph(Pij=P_ij,
                        ind=ind,
                        set_op_mix_ratio=set_op_mix_ratio)
    # Here P is a sparse matrix in csr format
    return P


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# Part 2:
# Functions - How to initialize embeddings
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

def init_spectral_layout(graph, random_state, embedding_dim=2, verbose=1):
    if verbose:
        print('Initialize {}-d embedding using spectral layout...'.format(embedding_dim))
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)
    assert(n_components == 1)
    diag_data = np.asarray(graph.sum(axis=0))
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D
    k = embedding_dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
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
    init_embedding = (init_embedding * expansion).astype(
        np.float32
    ) + random_state.normal(
        scale=0.0001, size=[graph.shape[0], embedding_dim]
    ).astype(
        np.float32
    )

    return init_embedding

def init_pca_layout(X, n_components, random_state):
    pca = PCA(n_components=n_components)
    init_embedding = pca.fit(X).transform(X)  # Convert to np.float32 to make numba work
    expansion = 10.0 / init_embedding.max()
    init_embedding = (init_embedding * expansion).astype(
        np.float32
    ) + random_state.normal(
        scale=0.0001, size=[X.shape[0], n_components]
    ).astype(
        np.float32
    )
    return init_embedding

def calculate_initial_embedding(X, graph, n_components, init_mode, random_state, verbose):
    if init_mode == 'spectral':
        return init_spectral_layout(graph=graph,
                                    random_state=random_state,
                                    embedding_dim=n_components,
                                    verbose=verbose)
    if init_mode == 'random':
        return random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    if init_mode == 'pca':
        return init_pca_layout(X=X, n_components=n_components, random_state=random_state)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# Part 3:
# Functions - How to optimize embeddings
# Using numba to make it faster
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    state[0] = (((state[0] & 4294967294) << 12) & 0xffffffff) ^ (
        (((state[0] << 13) & 0xffffffff) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xffffffff) ^ (
        (((state[1] << 2) & 0xffffffff) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xffffffff) ^ (
        (((state[2] << 3) & 0xffffffff) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]

@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit("f4(f4[:],f4[:])", fastmath=True)
def rdist(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result

@numba.njit(fastmath=True, parallel=True)
def optimize_embedding(head_embedding,
                        tail_embedding,
                        head,
                        tail,
                        n_epochs,
                        n_vertices,
                        sample_step,
                        learning_rate,
                        a, b,
                        num_negative_samples,
                        rng_state,
                        verbose,
                        record_position):
    gamma = 1.0

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = learning_rate

    num_pairs = sample_step.shape[0]
    # negative_sample_step = sample_step / negative_sample_rate
    # negative_sample_epoch_progress = negative_sample_step.copy()
    sample_epoch_progress = sample_step.copy()

    embedding_log = []
    for n in range(1, n_epochs+1):
        if record_position:
            embedding_log.append(head_embedding.copy())
        for i in range(num_pairs):
            if sample_epoch_progress[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_squared = rdist(current, other)
                # Updating current, i.e. low dimension embeddings
                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    # If We do not move other, it seems that convergence can be reached
                    if move_other:
                        other[d] += -grad_d * alpha

                sample_epoch_progress[i] += sample_step[i]

                # n_neg_samples = int((n - negative_sample_epoch_progress[i])/negative_sample_step[i])
                # Negative sampling, unlike original umap source code, we n_neg_samples a fixed value
                for p in range(num_negative_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:

                        grad_coeff = 2.0 * gamma * b
                        # Here 0.001 can be removed
                        grad_coeff /= (dist_squared) * (
                                a * pow(dist_squared, b) + 1
                        )
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha
                # It seems that calculation of negative_sample_epoch_progress is used only for n_neg_samples
                # if it will be selected next time
                #negative_sample_epoch_progress[i] += (n_neg_samples * negative_sample_step[i])

        # learning_rate is decreasing as iteration goes
        alpha = learning_rate * (1.0 - (float(n) / float(n_epochs)))
        if alpha < learning_rate*0.0001: alpha = learning_rate*0.0001
        if verbose and n % 50==0:
            print("completed ", n, " / ", n_epochs, "epochs")

    return head_embedding, embedding_log


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

class UMAP:
    def __init__(self,
                 X,
                 n_neighbors=30,
                 n_components=2,
                 metric="euclidean",
                 n_epochs=500,
                 learning_rate=1.0,
                 init="spectral",
                 min_dist=0.1,
                 spread=1.0,
                 set_op_mix_ratio=1.0,
                 local_connectivity=1.0,
                 num_negative_samples=5,
                 random_seed=42,
                 a=None,
                 b=None,
                 perplexity=None,
                 verbose=True,
                 keep_log = False
                 ):
        self.X = X
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.min_dist = min_dist
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = int(local_connectivity)
        self.num_negative_samples = num_negative_samples
        self.random_state = check_random_state(random_seed)
        self.keep_log = keep_log
        if keep_log:
            self.logs = []
        else:
            self.logs = None
        # Find a, b params
        if a is None or b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)
        else:
            self.a = a
            self.b = b
        if perplexity is None:
            self.perplexity = n_neighbors
        self.verbose = verbose
        if self.verbose:
            print(self)


    def _validate_params(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.min_dist <= 0.0:
            raise ValueError("min_dist must be greater than 0.0")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random", "pca"):
            raise ValueError('string init values must be "spectral", "pca" or "random"')
        if (
                isinstance(self.init, np.ndarray)
                and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value-{}".format(self.n_components))
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 2")

    def fit(self, X):
        # Validate parameters before fit
        self._validate_params()
        # Construct the graph

        self.graph = contruct_graph_from_X(X, n_neighbors=self.n_neighbors,
                                          metric=self.metric,
                                          local_connectivity=self.local_connectivity,
                                          perplexity=self.perplexity,
                                          set_op_mix_ratio=1.0,
                                          verbose=True)
        # Calculate initial embedding
        if isinstance(self.init, np.ndarray):
            self.init_embedding = self.init
        else:
            self.init_embedding = calculate_initial_embedding(X,
                                                    graph=self.graph,
                                                    random_state=self.random_state,
                                                    n_components= self.n_components,
                                                    init_mode=self.init,
                                                    verbose=self.verbose)
        init_embedding_copy = self.init_embedding.copy()
        # Optimize embedding
        csr_indices = self.graph.indices
        csr_indptr = self.graph.indptr

        graph = self.graph.tocoo()
        graph.sum_duplicates()
        n_vertices = graph.shape[1]
        sample_step = 1/graph.data
        head = graph.row
        tail = graph.col

        rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        self.embedding, self.logs = optimize_embedding(head_embedding=init_embedding_copy,
                                tail_embedding=init_embedding_copy,
                                head=head,
                                tail=tail,
                                n_epochs=self.n_epochs,
                                n_vertices=n_vertices,
                                sample_step=sample_step,
                                learning_rate=self.learning_rate,
                                a=self.a, b=self.b,
                                num_negative_samples=self.num_negative_samples,
                                rng_state=rng_state,
                                verbose=self.verbose,
                                record_position=self.keep_log)
        self.embedding[:, 0] = self.embedding[:, 0] - self.embedding[:, 0].mean()
        self.embedding[:, 1] = self.embedding[:, 1] - self.embedding[:, 1].mean()

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding



