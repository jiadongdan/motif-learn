import numbers
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from time import time
from sklearn.utils.extmath import randomized_svd

from ..feature import extract_patches
from ..feature import reconstruct_patches

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
    u, s, v = randomized_svd(patches, n_components)
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

def denoise_svd(data, patch_size, n_components, extraction_step=None):
    if len(data.shape) == 2:
        data_ = denoise_svd_single_image(data, patch_size, n_components, extraction_step)
    elif len(data.shape) == 3:
        data_ = denoise_svd_stack(data, patch_size, n_components, extraction_step)
    return data_


def svd_components(img, patch_size, n_components, extraction_step=None):
    if isinstance(patch_size, numbers.Number):
        patch_size = tuple([patch_size] * 2)
    img_height, img_width = img.shape
    patch_height, patch_width = patch_size
    num_patches = (img_height - patch_height + 1) * (img_width - patch_width + 1)

    # Estimate extraction step
    # extraction_step = int(np.ceil(np.sqrt(num_patches / 10000)))
    if extraction_step is None:
        extraction_step = int(patch_size[0]/4)

    print('Extracting reference patches...')
    t0 = time()
    patches = extract_patches(img, patch_size, extraction_step)
    patches = patches.reshape(patches.shape[0], -1)
    print('done in %.2fs.' % (time() - t0))

    print('Singular value decomposition...')
    t0 = time()
    u, s, v = randomized_svd(patches, n_components)
    print('done in %.2fs.' % (time() - t0))

    print('Reconstructing...')
    data_list = []
    t0 = time()
    for i, e in enumerate(s):
        S = np.zeros_like(np.diag(s))
        S[i, i] = e
        patches_ = np.dot(u, np.dot(S, v)).reshape(-1, patch_size[0], patch_size[1])
        data = reconstruct_patches(patches_, img.shape, extraction_step)
        data_list.append(data)
    print('done in %.2fs.' % (time() - t0))

    return data_list


def denoise_fft(img, patch_size, n_components, extraction_step):
    f_complex = np.fft.fftshift(np.fft.fft2(img))
    f_real = np.real(f_complex)
    f_imag = np.imag(f_complex)
    f1 = denoise_svd(f_real, patch_size, n_components, extraction_step)
    f2 = denoise_svd(f_imag, patch_size, n_components, extraction_step)
    f = np.fft.fftshift(f1 + 1j*f2)
    return np.abs(np.fft.ifft2(f))



def singular_vals(img, patch_size, n_components):
    if isinstance(patch_size, numbers.Number):
        patch_size = tuple([patch_size] * 2)
    img_height, img_width = img.shape
    patch_height, patch_width = patch_size
    num_patches = (img_height - patch_height + 1) * (img_width - patch_width + 1)

    # Estimate extraction step
    extraction_step = int(np.ceil(np.sqrt(num_patches / 10000)))

    print('Extracting reference patches...')
    t0 = time()
    patches = extract_patches(img, patch_size, extraction_step)
    patches = patches.reshape(patches.shape[0], -1)
    print('done in %.2fs.' % (time() - t0))

    print('Singular value decomposition...')
    t0 = time()
    u, s, v = randomized_svd(patches, n_components)
    print('done in %.2fs.' % (time() - t0))

    return s

def usv(img, patch_size, n_components):
    if isinstance(patch_size, numbers.Number):
        patch_size = tuple([patch_size] * 2)
    img_height, img_width = img.shape
    patch_height, patch_width = patch_size
    num_patches = (img_height - patch_height + 1) * (img_width - patch_width + 1)

    # Estimate extraction step
    extraction_step = int(np.ceil(np.sqrt(num_patches / 10000)))

    print('Extracting reference patches...')
    t0 = time()
    patches = extract_patches(img, patch_size, extraction_step)
    patches = patches.reshape(patches.shape[0], -1)
    print('done in %.2fs.' % (time() - t0))

    print('Singular value decomposition...')
    t0 = time()
    u, s, v = randomized_svd(patches, n_components)
    print('done in %.2fs.' % (time() - t0))

    return u, s, v


def denoise_gaussian(img, sigma):
    # Gaussian filters
    img_f = gaussian_filter(img, sigma)
    return img_f


def quality_index(img, img_f):
    std1 = np.std(img)
    std2 = np.std(img_f)
    m1 = np.mean(img)
    m2 = np.mean(img_f)
    cov = np.cov(img.reshape(1,-1), img_f.reshape(1,-1))[0, 1]
    return 4*cov / (std1 * std2) * m1 * m2 / (m1 ** 2 + m2 ** 2) * std1 * std2 / (std1 ** 2 + std2 ** 2)
