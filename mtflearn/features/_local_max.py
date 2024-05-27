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
