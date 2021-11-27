import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from skimage.transform import warp_polar
from skimage.filters import gaussian, window

from ..utils.baseline import baseline_correction


def find_fft_strong_peak_position(img_fft, sigma=2, need_shift=False):
    if need_shift:
        img_fft = np.fft.fftshift(img_fft)

    b = warp_polar(img_fft)

    # get radial intesnity
    l = b.mean(axis=0)
    l_log = np.log(l + 1)[:img_fft.shape[0] // 3]
    # smooth the curve
    l_log = gaussian(l_log, sigma)
    v = (l_log.max() + l_log.min()) / 2
    idx = (np.abs(l_log - v)).argmin()
    base_line = baseline_correction(l_log, 20)
    l_log_ = l_log - base_line
    l_log_[0:idx] = 0
    r = find_peaks(l_log_, distance=len(l_log_))[0][0]
    return r


class ImageFFT:

    def __init__(self, img, shift=True, use_win=True):
        if use_win:
            win = window('hann', img.shape)
        else:
            win = 1.

        self.image = img
        self.fft_complex = np.fft.fft2(self.image*win)
        if shift:
            self.fft_complex = np.fft.fftshift(self.fft_complex)
        self.r1 = find_fft_strong_peak_position(self.abs, need_shift=np.logical_not(shift))

    @property
    def real(self):
        return self.fft_complex.real


    @property
    def imag(self):
        return self.fft_complex.imag

    @property
    def abs(self):
        return np.abs(self.fft_complex)

    @property
    def log(self):
        return np.log(self.abs + 1)

    @property
    def log_real(self):
        return np.sign(self.real) * np.log(np.abs(self.real) + 1)

    @property
    def log_imag(self):
        return np.sign(self.imag) * np.log(np.abs(self.imag) + 1)

    def show(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.2))

        y0, x0 = self.image.shape
        for ax, e in zip(axes, [self.log, self.log_real, self.log_imag]):
            ax.imshow(e)
            ax.axis('off')
            ax.set_xlim(x0//2-self.r1*1.2, x0//2+self.r1*1.2)
            ax.set_ylim(y0//2-self.r1*1.2, y0//2+self.r1*1.2)


#=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# functions
#=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_fft_abs(img, use_win=False, use_shift=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.

    if use_shift:
        fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(img * win)))
    else:
        fft_abs = np.abs(np.fft.fft2(img * win))

    return fft_abs


def get_randial_intensity(fft_abs):
    # polar warp the image
    b = warp_polar(fft_abs)

    # get radial intesnity
    l = b.mean(axis=0)
    return l


def get_range_s(l, zoom=2):
    s1 = int(len(l) // np.sqrt(2))
    s2 = s1 // 2
    s = (s1 + s2) / 2 / zoom
    return int(s)


def get_central_s(l):
    s = np.diff(l[0:min(10, len(l))]).argmin()
    return max(1, s)


# need to add Cursor function ?
def fftshow(img, ax=None, use_win=False, zoom=2, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    fft_abs = get_fft_abs(img, use_win=use_win, use_shift=True)
    l = get_randial_intensity(fft_abs)

    y, x = np.unravel_index(np.argmax(fft_abs), fft_abs.shape)
    s0 = get_central_s(l)
    s = get_range_s(l, zoom)

    # set central region to zero for better visulization
    fft_abs[y - s0:y + s0+1, x - s0:x + s0+1] = 0
    fft_abs_crop = fft_abs[y - s: y + s + 1, x - s:x + s + 1]

    ax.imshow(fft_abs_crop, **kwargs)
    ax.axis('off')

def fft_compare(imgs, axes=None, use_win=False, zoom=2, **kwargs):
    if axes is None:
        _, axes = plt.subplots(1, len(imgs), figsize=(12, 12/len(imgs)), sharex=True, sharey=True)

    for ax, img in zip(axes, imgs):
        fftshow(img, ax, use_win=use_win, zoom=zoom, vmin=0, **kwargs)

    axes[0].figure.tight_layout()
