import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score, average_precision_score
from ..clustering import labels2matrix

def compute_similarity_matrix(X, metric='euclidean', sigma=None, k=10, normalize=False, symmetrize=False):
    """
    Compute a similarity matrix S from a feature matrix X using exponential decay.

    Parameters:
        X (np.ndarray): (N x D) feature matrix, one row per sample.
        metric (str): Distance metric (e.g., 'euclidean', 'manhattan', 'cosine').
        sigma (str, float, or None): Similarity scaling.
            - None or 'mean': Use mean pairwise distance.
            - 'self-tuning': sigma_i = distance to k-th neighbor of x_i.
            - 'umap': Similarity style from UMAP (adaptive local sigma).
            - float: Use directly.
        k (int): Neighborhood size for 'self-tuning' or 'umap' sigma.
        normalize (bool): Whether to row-normalize the similarity matrix.
        symmetrize (bool): Whether to symmetrize the matrix via (S + S.T) / 2.

    Returns:
        S (np.ndarray): (N x N) similarity matrix.
    """
    distances = pairwise_distances(X, metric=metric)
    N = distances.shape[0]

    if sigma is None or sigma == 'mean':
        sigma_value = np.mean(distances)
        S = np.exp(-distances**2 / (2 * sigma_value**2))

    elif isinstance(sigma, (float, int)):
        S = np.exp(-distances**2 / (2 * sigma**2))

    elif sigma == 'self-tuning':
        sorted_distances = np.sort(distances, axis=1)
        local_sigmas = sorted_distances[:, k] + 1e-8  # avoid division by zero
        sigma_matrix = np.outer(local_sigmas, local_sigmas)
        S = np.exp(-distances**2 / (sigma_matrix + 1e-8))

    elif sigma == 'umap':
        sorted_distances = np.sort(distances, axis=1)
        rho = sorted_distances[:, 1]  # Distance to first nearest neighbor (not self)
        sigma_umap = np.zeros(N)
        for i in range(N):
            for j in range(2, N):
                d = sorted_distances[i, j]
                if d > rho[i]:
                    sigma_umap[i] = d - rho[i]
                    break
            if sigma_umap[i] == 0.0:
                sigma_umap[i] = 1.0e-3  # fallback

        # Now compute the conditional probabilities
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    S[i, j] = 0.0
                else:
                    d = distances[i, j]
                    S[i, j] = np.exp(-(max(0.0, d - rho[i])) / sigma_umap[i])

        # UMAP symmetrization trick
        S = S + S.T - S * S.T

    else:
        raise ValueError(f"Unrecognized sigma setting: {sigma}")

    if symmetrize and sigma != 'umap':
        S = (S + S.T) / 2

    if normalize:
        S = S / (S.sum(axis=1, keepdims=True) + 1e-8)

    return S

def compute_clustering_score(X, y):
    S = compute_similarity_matrix(X, metric='euclidean', sigma=None)
    A = labels2matrix(y)
    # Extract upper triangle without diagonal (to avoid duplicates and self-pairs)
    triu_idx = np.triu_indices_from(A, k=1)

    A_flat = A[triu_idx].ravel()
    S_flat = S[triu_idx].ravel()

    auc = roc_auc_score(A_flat, S_flat)
    ap = average_precision_score(A_flat, S_flat)

    return {'AUC': auc, 'Average Precision': ap}