import numbers
import numpy as np

from time import time
from sklearn.utils.extmath import randomized_svd

from ..feature import extract_patches
from ..feature import reconstruct_patches


def low_rank_svd(data, rank, compute_uv = False):
    u, s, v = randomized_svd(data, rank)
    if compute_uv == True:
        return u, s, v
    else:
        return s


def denoise_svd_single_image(img, patch_size, n_components, extraction_step=None, verbose=True):
    if isinstance(patch_size, numbers.Number):
        patch_size = tuple([patch_size] * 2)
    img_height, img_width = img.shape
    patch_height, patch_width = patch_size
    num_patches = (img_height - patch_height + 1) * (img_width - patch_width + 1)

    # Estimate extraction step
    # extraction_step = int(np.ceil(np.sqrt(num_patches / 10000)))
    if extraction_step is None:
        extraction_step = int(patch_size[0]/4)
    if verbose == True:
        print('Extracting reference patches...')
    t0 = time()
    patches = extract_patches(img, patch_size, extraction_step)
    patches = patches.reshape(patches.shape[0], -1)
    if verbose == True:
        print('done in %.2fs.' % (time() - t0))
        print('Singular value decomposition...')
    t0 = time()
    u, s, v = randomized_svd(patches, n_components, random_state=0)
    if verbose == True:
        print('done in %.2fs.' % (time() - t0))
        print('Reconstructing...')
    t0 = time()
    S = np.diag(s)
    patches_ = np.dot(u, np.dot(S, v)).reshape(-1, patch_size[0], patch_size[1])
    data = reconstruct_patches(patches_, img.shape, extraction_step)
    if verbose == True:
        print('done in %.2fs.' % (time() - t0))

    return data

def denoise_svd_stack(imgs, patch_size, n_components, extraction_step=None):
    imgfs = []
    print('denoising...')
    for i, img in enumerate(imgs):
        imgf = denoise_svd_single_image(img, patch_size, n_components, extraction_step, verbose=False)
        imgfs.append(imgf)
        print('{}, '.format(i), end='')
    return np.array(imgfs)

def denoise_svd_patch(data, patch_size, n_components, extraction_step=None, verbose=True):
    if len(data.shape) == 2:
        data_ = denoise_svd_single_image(data, patch_size, n_components, extraction_step, verbose=verbose)
    elif len(data.shape) == 3:
        data_ = denoise_svd_stack(data, patch_size, n_components, extraction_step)
    return data_


# Denoise 2D or 3D image data using direct svd method
def denoise_svd_no_patch_single_image(data, rank):
    u, s, v = low_rank_svd(data, rank, compute_uv=True)
    S = np.diag(s)
    return np.dot(u, np.dot(S, v))

def denoise_svd_no_patch_stack(data, rank):
    return np.array([denoise_svd_no_patch_single_image(img, rank) for img in data])

def denoise_svd_no_patch(data, rank):
    if len(data.shape) == 2:
        data_ = denoise_svd_no_patch_single_image(data, rank)
    elif len(data.shape) == 3:
        data_ = denoise_svd_no_patch_stack(data, rank)
    return data_


def denoise_svd(data, rank, patch_size = None, extraction_step=None, verbose=True):
    if patch_size == None:
        data_ = denoise_svd_no_patch(data, rank)
    else:
        data_ = denoise_svd_patch(data, patch_size, rank, extraction_step, verbose=verbose)
    return data_






