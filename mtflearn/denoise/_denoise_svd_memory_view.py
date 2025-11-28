import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm  # pip install tqdm if needed
import math

def extract_patches(data, patch_shape=64, extraction_step=1):
    data_ndim = data.ndim
    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * data_ndim)
    patch_strides = data.strides
    slices = tuple([slice(None, None, st) for st in (1, 1)])
    indexing_strides = data[slices].strides
    patch_indices_shape = (np.array(data.shape) - np.array(patch_shape)) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    patches = as_strided(data, shape=shape, strides=strides)
    return patches

def batch_flatten_patches(ps, batch_size):
    """
    Generator yielding flattened patches from the 4D array ps.
    ps: shape (n_rows, n_cols, patch_height, patch_width)
    """
    n_rows, n_cols, ph, pw = ps.shape
    batch = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch_flat = ps[i, j].ravel()  # Flatten the patch
            batch.append(patch_flat)
            if len(batch) >= batch_size:
                yield np.array(batch)
                batch = []
    if batch:
        yield np.array(batch)

def batch_flatten_patches_with_indices(ps, batch_size):
    """
    Like batch_flatten_patches but also yields the (i, j) indices.
    """
    n_rows, n_cols, ph, pw = ps.shape
    batch = []
    batch_indices = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch_flat = ps[i, j].ravel()
            batch.append(patch_flat)
            batch_indices.append((i, j))
            if len(batch) >= batch_size:
                yield np.array(batch), batch_indices
                batch = []
                batch_indices = []
    if batch:
        yield np.array(batch), batch_indices

def get_optimal_batch_size(ps, target_memory_bytes=100*2**20):
    # ps shape: (n_rows, n_cols, patch_height, patch_width)
    patch_size = ps.shape[2]  # assume square patches
    patch_memory = patch_size * patch_size * 8  # bytes per patch (float64)
    batch_size = max(1, int(target_memory_bytes // patch_memory))
    return batch_size

def compute_mean_and_cov(ps, batch_size, show_progress=True):
    total_count = 0
    mean = None
    n_rows, n_cols, ph, pw = ps.shape
    total_patches = n_rows * n_cols
    n_batches = math.ceil(total_patches / batch_size)

    # Compute mean
    for batch in tqdm(batch_flatten_patches(ps, batch_size), total=n_batches,
                      desc="Computing mean", disable=not show_progress):
        if mean is None:
            mean = batch.mean(axis=0)
        else:
            mean = (mean * total_count + batch.sum(axis=0)) / (total_count + batch.shape[0])
        total_count += batch.shape[0]

    dim = mean.shape[0]
    cov = np.zeros((dim, dim))
    total_count = 0

    # Compute covariance
    for batch in tqdm(batch_flatten_patches(ps, batch_size), total=n_batches,
                      desc="Computing covariance", disable=not show_progress):
        batch_centered = batch - mean
        cov += batch_centered.T @ batch_centered
        total_count += batch.shape[0]
    cov /= (total_count - 1)

    return mean, cov

def denoise_svd(image, patch_size, n_components=None, threshold=0.9,
                batch_size=None, target_memory_bytes=100*2**20, show_progress=True):
    """
    Performs PCA on overlapping patches of the image and reconstructs the image
    using the top principal components. Overlapping regions are averaged.

    Returns:
      recon: the reconstructed image,
      explained_variance: eigenvalues in descending order,
      explained_variance_ratio: explained variance ratio per component.
    """
    # 1. Extract patches
    patches = extract_patches(image, patch_size)
    if batch_size is None:
        batch_size = get_optimal_batch_size(patches, target_memory_bytes)

    # 2. Compute mean and covariance matrix over flattened patches
    mean, cov = compute_mean_and_cov(patches, batch_size=batch_size, show_progress=show_progress)

    # 3. Compute eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    explained_variance = eigvals_sorted
    explained_variance_ratio = eigvals_sorted / np.sum(eigvals_sorted)

    if n_components is None:
        n_components = np.sum(np.cumsum(explained_variance_ratio) < threshold) + 1

    # Choose the top n_components eigenvectors
    top_components = eigvecs[:, -n_components:]

    # 4. Reconstruct the image using the PCA model patch-by-patch.
    n_rows, n_cols, ph, pw = patches.shape
    recon = np.zeros_like(image, dtype=np.float64)
    weight = np.zeros_like(image, dtype=np.float64)
    total_patches = n_rows * n_cols
    n_batches = math.ceil(total_patches / batch_size)

    for batch, batch_indices in tqdm(batch_flatten_patches_with_indices(patches, batch_size),
                                     total=n_batches, desc="Reconstructing", disable=not show_progress):
        # Subtract the mean, project onto top components, then reconstruct.
        proj = (batch - mean) @ top_components        # (batch_size, n_components)
        rec_batch = proj @ top_components.T + mean      # (batch_size, patch_size*patch_size)
        rec_batch = rec_batch.reshape(-1, patch_size, patch_size)

        # For each reconstructed patch, add its value into the accumulator.
        for (i, j), rec_patch in zip(batch_indices, rec_batch):
            recon[i:i+patch_size, j:j+patch_size] += rec_patch
            weight[i:i+patch_size, j:j+patch_size] += 1.0

    # Average overlapping contributions.
    recon /= weight

    return recon, explained_variance_ratio, n_components
