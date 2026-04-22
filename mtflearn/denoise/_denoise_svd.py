import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from itertools import product
from sklearn.utils.extmath import randomized_svd
from time import time


def _patch_start_indices(image_extent, patch_extent, step):
    if step <= 0:
        raise ValueError("extraction_step must be a positive integer.")
    if patch_extent >= image_extent:
        raise ValueError("patch_size must be strictly smaller than the image size.")

    last_start = image_extent - patch_extent
    indices = np.arange(0, last_start, step)
    if indices.size == 0 or indices[-1] != last_start:
        indices = np.append(indices, last_start)
    return indices


def extract_patches(data, patch_shape=64, extraction_step=1):
    data_ndim = data.ndim
    # if patch_shape is a number, turn it into tuple
    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * data_ndim)

    patch_strides = data.strides

    # Extract all patches setting extraction_step to 1
    slices = tuple([slice(None, None, st) for st in (1, 1)])
    indexing_strides = data[slices].strides

    patch_indices_shape = (np.array(data.shape) - np.array(patch_shape)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    # Using strides and shape to get a 4d numpy array
    patches = as_strided(data, shape=shape, strides=strides)

    i = _patch_start_indices(data.shape[0], patch_shape[0], extraction_step)
    j = _patch_start_indices(data.shape[1], patch_shape[1], extraction_step)

    ij = np.array([(e[0], e[1]) for e in product(i, j)])

    patches = patches[ij[:, 0], ij[:, 1], :, :]

    return patches


def reconstruct_patches(patches, img_shape, reconstruction_step):
    if isinstance(img_shape, numbers.Number):
        img_shape = tuple([img_shape] * 2)
    img_height, img_width = img_shape
    patch_height, patch_width = patches.shape[1:3]

    i = _patch_start_indices(img_height, patch_height, reconstruction_step)
    j = _patch_start_indices(img_width, patch_width, reconstruction_step)
    # Patches accumulation
    img = np.zeros(img_shape)
    for p, (m, n) in zip(patches, product(i, j)):
        img[m:m + patch_height, n:n + patch_width] += p
    # Overlap accumulation
    overlap_cnt = np.zeros(img.shape)
    one = np.ones((patch_height, patch_width))
    for m, n in product(i, j):
        overlap_cnt[m:m + patch_height, n:n + patch_width] += one

    return img / overlap_cnt

def low_rank_svd(data, rank, compute_uv = False):
    u, s, v = randomized_svd(data, rank)
    if compute_uv == True:
        return u, s, v
    else:
        return s


def denoise_svd(img, patch_size, n_components, extraction_step=None, verbose=True, return_s=False):
    if isinstance(patch_size, numbers.Number):
        patch_size = tuple([patch_size] * 2)
    img_height, img_width = img.shape
    patch_height, patch_width = patch_size
    if patch_height >= img_height or patch_width >= img_width:
        raise ValueError(
            "patch_size must be strictly smaller than the image dimensions."
        )
    num_patches = (img_height - patch_height + 1) * (img_width - patch_width + 1)

    # Estimate extraction step
    # extraction_step = int(np.ceil(np.sqrt(num_patches / 10000)))
    if extraction_step is None:
        extraction_step = max(1, int(patch_size[0] / 4))

    if verbose == True:
        print('Extracting reference patches...')
    t0 = time()
    patches = extract_patches(img, patch_size, extraction_step)
    patches = patches.reshape(patches.shape[0], -1)

    if verbose == True:
        print('done in %.2fs.' % (time() - t0))
        print('Singular value decomposition...')
    t0 = time()
    u, s, v = randomized_svd(patches, n_components, random_state=None)

    if verbose == True:
        print('done in %.2fs.' % (time() - t0))
        print('Reconstructing...')
    t0 = time()
    S = np.diag(s)
    patches_ = np.dot(u, np.dot(S, v)).reshape(-1, patch_size[0], patch_size[1])
    img_clean = reconstruct_patches(patches_, img.shape, extraction_step)
    if verbose == True:
        print('done in %.2fs.' % (time() - t0))

    if return_s:
        return img_clean, s
    else:
        return img_clean


class DenoiseSVD:

    def __init__(self, image, n_components, patch_size, extraction_step):
        self.image = image
        self.n_components = n_components
        self.patch_size = patch_size
        self.extraction_step = extraction_step

        self.patches = None
        self.s_values = None
        self.img_clean = None

    def run(self, verbose=False):
        self.img_clean, self.s_values = denoise_svd(
            self.image,
            n_components=self.n_components,
            patch_size=self.patch_size,
            extraction_step=self.extraction_step,
            return_s=True,
            verbose=verbose,
        )
        return self.img_clean
