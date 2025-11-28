import numpy as np
from scipy.sparse import issparse
from umap.umap_ import fuzzy_simplicial_set
import networkx as nx
from collections import defaultdict

def construct_umap_graph(
        X,
        n_neighbors=10,
        metric='euclidean',
        metric_kwds={},
        random_state=None,
        knn_indices=None,
        knn_dists=None,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=False,
        return_dists=False,
):

    results =  fuzzy_simplicial_set(X=X,
                                    n_neighbors=n_neighbors,
                                    random_state=random_state,
                                    metric=metric,
                                    metric_kwds=metric_kwds,
                                    knn_indices=knn_indices,
                                    knn_dists=knn_dists,
                                    angular=angular,
                                    set_op_mix_ratio=set_op_mix_ratio,
                                    local_connectivity=local_connectivity,
                                    apply_set_operations=apply_set_operations,
                                    verbose=verbose,
                                    return_dists=return_dists,
                                    )
    return results[0]

def calculate_modularity_from_graph(A, labels):
    """
    Compute the modularity of a graph given its adjacency matrix and community labels.

    Supports both dense (numpy.ndarray) and sparse (scipy sparse) matrices.

    Parameters:
        A (np.ndarray or scipy.sparse matrix): An n x n adjacency matrix representing the graph.
        labels (np.ndarray): A 1D array of length n containing community labels for each node.

    Returns:
        float: The modularity value.
    """
    n = A.shape[0]

    # Total number of edges (m) in an undirected graph
    m = A.sum() / 2.0

    if issparse(A):
        # For sparse matrix, extract degree vector and process community-by-community.
        k = np.array(A.sum(axis=1)).flatten()  # Ensure k is a 1D numpy array
        modularity_sum = 0.0
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # Get indices of nodes in the current community
            nodes = np.where(labels == label)[0]
            # Sum of actual edges within the community (stays sparse)
            A_comm = A[nodes, :][:, nodes]
            actual_sum = A_comm.sum()

            # Compute the expected sum for these nodes:
            k_comm = k[nodes]
            expected_sum = np.outer(k_comm, k_comm).sum() / (2*m)

            modularity_sum += (actual_sum - expected_sum)

        modularity = modularity_sum / (2*m)

    else:
        # For dense matrix, use the matrix formulation with an indicator matrix S.
        k = A.sum(axis=1)
        expected = np.outer(k, k) / (2*m)
        B = A - expected

        # Build community indicator matrix S: each column corresponds to a community.
        unique_labels = np.unique(labels)
        S = np.zeros((n, unique_labels.size))
        for j, label in enumerate(unique_labels):
            S[:, j] = (labels == label).astype(float)

        modularity = np.trace(S.T @ B @ S) / (2*m)

    return modularity

def calculate_modularity_from_X(X, labels, n_neighbors=10, metric='euclidean'):
    A = construct_umap_graph(X, n_neighbors=n_neighbors, metric=metric)
    return calculate_modularity_from_graph(A, labels)

def calculate_modularity_nx(A, labels):
    # Step 1: convert sparse or dense adjacency matrix to NetworkX graph
    if issparse(A):
        G = nx.from_scipy_sparse_array(A)
    else:
        G = nx.from_numpy_array(A)

    # Step 2: group nodes by their labels to define communities
    label_to_nodes = defaultdict(list)
    for node, label in enumerate(labels):
        label_to_nodes[label].append(node)

    communities = list(label_to_nodes.values())

    # Step 3: calculate modularity
    return nx.algorithms.community.quality.modularity(G, communities)

def calculate_distance_modularity(S, labels):
    """
    Calculate the distance-modularity Q_D.

    Parameters:
        S (numpy.ndarray): An (N x N) similarity matrix where S[i, j] is the similarity
                           between node i and node j.
        labels (numpy.ndarray or list): A 1D array or list of community labels for each node.

    Returns:
        Q_D (float): The computed distance-modularity.
    """
    # Total similarity
    S_tot = np.sum(S)

    # Compute node strengths
    s = np.sum(S, axis=1)

    # Compute the null model: expected similarity matrix
    P = np.outer(s, s) / S_tot

    # Compute the modularity matrix: B = S - P
    B = S - P

    # Create a boolean community indicator matrix, where entry (i,j) is True if nodes i and j
    # belong to the same community
    labels = np.array(labels)
    community_mask = np.equal.outer(labels, labels)

    # Calculate Q_D: sum over entries where community_mask is True and normalize by S_tot
    Q_D = np.sum(B[community_mask]) / S_tot
    return Q_D



