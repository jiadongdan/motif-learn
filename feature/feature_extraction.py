import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from itertools import product
from skimage.feature import hog
from skimage.feature import daisy


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

    h = data.shape[0] - patch_shape[0]
    w = data.shape[1] - patch_shape[1]

    i = np.arange(0, h, extraction_step)
    j = np.arange(0, w, extraction_step)

    if i[-1] != h:
        i = np.insert(i, i.size, h)
    if j[-1] != w:
        j = np.insert(j, j.size, w)

    ij = np.array([(e[0], e[1]) for e in product(i, j)])

    patches = patches[ij[:, 0], ij[:, 1], :, :]

    return patches


def reconstruct_patches(patches, img_shape, reconstruction_step):
    if isinstance(img_shape, numbers.Number):
        img_shape = tuple([img_shape] * 2)
    img_height, img_width = img_shape
    patch_height, patch_width = patches.shape[1:3]

    h = img_height - patch_height
    w = img_width - patch_width
    i = np.arange(0, h, reconstruction_step)
    j = np.arange(0, w, reconstruction_step)
    if i[-1] != h:
        i = np.insert(i, i.size, h)
    if j[-1] != w:
        j = np.insert(j, j.size, w)
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

def get_intensity_features(img, pts):
    I = np.array([img[y, x] for (x, y) in pts])
    return I

def get_raw_features(img, coords, size):
    pts = []
    features = []
    for (x, y) in coords:
        if (x > size and x < img.shape[1] - size - 1 and y > size and y < img.shape[0] - size - 1):
            pts.append([x, y])
            features.append(img[y-size:y+size+1, x-size:x+size+1].flatten())
    return pts, features

def hog_feature(img, orientations):
    img_shape = img.shape
    f = hog(img, orientations=orientations,
            pixels_per_cell=img_shape,
            cells_per_block=(1, 1),
            block_norm='L2-Hys',
            visualize=False, transform_sqrt=False,
            feature_vector=True, multichannel=None)
    return f

def daisy_feature(img, rings=2):
    step = img.shape[0]
    radius = img.shape[0] // 2

    f = daisy(img, step=step, radius=radius, rings=rings,
              histograms=8, orientations=8, normalization='l1',
              sigmas=None, ring_radii=None, visualize=False)
    return f[0, 0, :]

def extract_daisy(ps, rings=2):
    return np.array([daisy_feature(patch, rings) for patch in ps])

def extract_hog(ps, orientations):
    return np.array([hog_feature(patch, orientations) for patch in ps])


