import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.io import imsave
from .dm4 import DMfile

from ..utils.utils import normalize

from collections import namedtuple

Dataset = namedtuple('Dataset', ['image', 'pts', 'features'])


def load_image(file_name):
    file_extension = os.path.splitext(file_name)[1]
    if file_extension.lower() in ['.dm4', '.dm3']:
        img = DMfile(file_name).data
    else:
        img = imread(file_name)
    return img


def save_image(data, file_name, cmap=None):
    if cmap == None:
        if data.dtype == np.float64:
            data = np.array(data, dtype=np.float32)
        imsave(file_name, data)
    else:
        # Normaize image into [0, 255]
        data = normalize(data, 0, 255)
        data = np.round(data).astype(np.int)
        R, G, B = cmap(np.arange(256))[:, 0:3].T
        r, g, b= R[data], G[data], B[data]
        rgb = np.stack([r,g,b], axis=2)
        imsave(file_name, rgb)


def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def save_dataset(img, pts, file_name):
    ds = Dataset(image=img, pts=pts, features=None)
    save_pickle(ds, file_name)


def readstem(file_name):
    f = np.loadtxt(file_name)
    h, w = int(f[0]), int(f[1])
    d = f[2:]
    f1= d[0::2].reshape(h, w)
    f2= d[1::2].reshape(h, w)
    return f1, f2


# Load black background vesta png image
def load_vesta_image(file_name):
    fname1 = file_name+'_back_bg.png'
    fname2 = file_name+'_white_bg.png'
    img = load_image(fname1)
    h, w, _ = img.shape
    alpha = np.ones((h, w), dtype=img.dtype)*255
    # add alpha channel
    img_rgba = np.dstack((img, alpha))
    # https://stackoverflow.com/a/52737768/5855131
    black_pixels_mask = np.all(img_rgba == [0, 0, 0, 255], axis=-1)
    # set black background to transparent
    img_white = load_image(fname2)
    img_rgba_white = np.dstack((img_white, alpha))
    img_rgba_white[:, :, 3][black_pixels_mask] = 0
    return img_rgba_white

