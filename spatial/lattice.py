import numpy as np
from scipy.signal import fftconvolve

from ..signal.peak_finding import get_patch_size


def lattice_map(img, patch=None):
    if patch is None:
        patch_size = get_patch_size(img)
        patch = img[0:patch_size, 0:patch_size]

    tt = fftconvolve(img, patch, mode='same')

    gg = np.ones_like(patch) * (patch.mean())
    kk = fftconvolve(img, gg, mode='same')
    ll = tt / kk
    return ll