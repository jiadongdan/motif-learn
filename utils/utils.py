import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage.transform import rotate
from skimage.measure import profile_line
from skimage.filters import window

import os
import glob


def get_matplotlib_backends():
    return matplotlib.rcsetup.all_backends

def get_matplotlib_current_backend():
    return matplotlib.get_backend()

def normalize(data, low= 0, high = 1, vmin=None, vmax=None):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    data_norm = (data - vmin)/(vmax - vmin)*(high - low) + low
    data_norm[data_norm <low] = low
    data_norm[data_norm >high] = high
    return data_norm


# http://jdherman.github.io/colormap/
def cm_blue():
    g = np.arange(0, 256)
    b = np.arange(0, 256*2,2)
    b[128:] = 255
    r = np.arange(-256, 256, 2)
    r[0:128] = 0
    C = np.vstack([r, g, b]).T
    cm = matplotlib.colors.ListedColormap(C/255.0)
    return cm



def fft_spectrum(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.0
    fft_complex = np.fft.fftshift(np.fft.fft2(win*img))
    fft_abs = np.abs(fft_complex)
    fft_log = np.log(fft_abs+1)
    return fft_log

def fft_abs(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.0
    f_complex = np.fft.fftshift(np.fft.fft2(win*img))
    f_abs = np.abs(f_complex)
    return f_abs

def fft_real(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.0
    f_complex = np.fft.fftshift(np.fft.fft2(win*img))
    return np.real(f_complex)

def fft_imag(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.0
    f_complex = np.fft.fftshift(np.fft.fft2(win*img))
    return np.imag(f_complex)

def fft_complex(img, use_win=True):
    if use_win:
        win = window('hann', img.shape)
    else:
        win = 1.0
    return np.fft.fft2(win*img)


def crop(data, x, y, h, w):
    if len(data.shape) == 2:
        return data[y:y+h, x:x+w]
    elif len(data.shape) == 3:
        return data[:, y:y+h, x:x+w]

def rot(data, angle, resize = True):
    # rotate function requires data to be in (0, 1)
    max_val = data.max()
    data = data/max_val
    if len(data.shape) == 2:
        data_ = rotate(data, angle, resize=resize)
    elif len(data.shape) == 3:
        data_ = np.array([rotate(img, angle, resize) for img in data])
    return data_*max_val



def extract_max_intensity(img, pts):
    return np.array([img[y, x] for (x, y) in pts])


def gaussian(s, sigma):
    Y, X = np.ogrid[-s//2:s//2:1j*s, -s//2:s//2:1j*s]
    return np.exp(-(X*X+Y*Y)/(2*sigma*sigma))

def g1(s, sigma):
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    gg = np.exp((-(X-s//4)**2+(Y-s//4)**2)/(2*sigma*sigma))+\
         np.exp((-(X - s // 4) ** 2 + (Y - s // 4) ** 2) / (2 * sigma * sigma))
    return gg

def many_gaussians(sigma, pts, s):
    data = np.zeros((s,s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X-x)**2-(Y-y)**2)/(2*sigma*sigma))
    return data

def get_roi(data, x, y, s):
    return data[y-s:y+s,x-s:x+s]


def plot_line_profile(ax, img, pp, **kwargs):
    src = (pp[0][1], pp[0][0])
    dst = (pp[1][1], pp[1][0])
    l = profile_line(img, src, dst, linewidth=1, order=3, mode='constant')
    ax.plot(l, lw=2, **kwargs)



# https://stackoverflow.com/a/12118327
def _get_appdata_path():
    import ctypes
    from ctypes import wintypes, windll
    CSIDL_APPDATA = 26
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND,
                                 ctypes.c_int,
                                 wintypes.HANDLE,
                                 wintypes.DWORD,
                                 wintypes.LPCWSTR]
    path_buf = wintypes.create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_APPDATA, 0, 0, path_buf)
    return path_buf.value

def dropbox_home():
    from platform import system
    import base64
    import os.path
    _system = system()
    if _system in ('Windows', 'cli'):
        host_db_path = os.path.join(_get_appdata_path(),
                                    'Dropbox',
                                    'host.db')
    elif _system in ('Linux', 'Darwin'):
        host_db_path = os.path.expanduser('~'
                                          '/.dropbox'
                                          '/host.db')
    else:
        raise RuntimeError('Unknown system={}'
                           .format(_system))
    if not os.path.exists(host_db_path):
        raise RuntimeError("Config path={} doesn't exists"
                           .format(host_db_path))
    with open(host_db_path, 'r') as f:
        data = f.read().split()

    return base64.b64decode(data[1])


import re

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def get_filenames(s, path=None, sort=None):
    if path is None:
        path = os.getcwd()

    names = np.array([name for name in glob.glob(path + '\\' + s)])
    if sort:
        ind = np.argsort([get_trailing_number(e.split('.')[0]) for e in names])
        names = names[ind]
    return names




