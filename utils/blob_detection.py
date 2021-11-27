import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def clean_border(mask, width):
    for i in range(mask.ndim):
        mask = mask.swapaxes(0, i)
        mask[:width] = mask[-width:] = 0
        mask = mask.swapaxes(0, i)
    return mask

# Use maximum filters, nice results for smooth images
def local_max(img, min_distance, threshold=None, exclude_border=True, plot=True):
    if type(exclude_border) == bool:
        exclude_border = min_distance if exclude_border else 0
    if threshold == None:
        threshold = np.mean(img)
    # size for maximum filter
    size = 2 * min_distance + 1
    img_max = ndi.maximum_filter(img, size=size, mode='constant')
    mask = (img == img_max)
    mask &= (img > threshold)
    if exclude_border:
        mask = clean_border(mask, width=exclude_border)
    coords = np.column_stack(np.nonzero(mask))
    coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()

    if plot is True:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.imshow(img)
        ax.plot(coords[:, 0], coords[:, 1], 'r.')
    return coords

def LoG(img, sigma):
    log = -ndi.gaussian_laplace(img, sigma)
    return log

def blob_log_(img, sigmas, min_distance=3, threshold=None):
    img_smooth = np.zeros_like(img)
    for s in sigmas:
        img_smooth = img_smooth + LoG(img, s) * s ** 2
    img_smooth = img_smooth/len(sigmas)
    pts = local_max(img_smooth, min_distance=min_distance, threshold=threshold)
    if pts.size == 0:
        pts = np.empty((0, 2))
    return pts

def blob_log(img, sigmas, min_distance=3, threshold=None):
    imgs = np.stack([LoG(img, s) * s ** 2 for s in sigmas], axis=-1)
    pts = local_max(imgs, min_distance=min_distance, threshold=threshold)
    if pts.size == 0:
        return np.empty((0, 3))
    lm = pts.astype(np.float64)
    lm[:, -1] = sigmas[pts[:, -1]]
    return pts[:, 0:2]