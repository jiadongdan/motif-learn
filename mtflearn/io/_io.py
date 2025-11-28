import os
import numpy as np
from skimage.io import imread
from pathlib import Path
from typing import Union
from os import PathLike
from ._dm4 import DMfile
from ._dm_ncempy import dmReader


def normalize(image, low=0., high=1.):
    img_max = image.max()
    img_min = image.min()
    img_norm = (image - img_min) / (img_max - img_min) * (high - low) + low
    return img_norm


def load_image(
        file_name: Union[str, PathLike],
        normalized: bool = False
) -> np.ndarray:
    # turn any PathLike (including pathlib.Path) into a str
    fn = os.fspath(file_name)
    # split off the extension
    _, ext = os.path.splitext(fn)
    ext = ext.lower()

    if ext in ('.dm4', '.dm3'):
        img = DMfile(fn).data
    else:
        img = imread(fn)

    if normalized:
        img = normalize(img)

    return img

def load_dm(filename):
    """
    Parameters
    ----------
    filename : str or pathlib.Path
        The path and name of the file to attempt to load. This chooses the Reader() function based on the file
        suffix.
    """

    out = {}

    # check filename type
    if isinstance(filename, str):
        filename = Path(filename)
    elif isinstance(filename, Path):
        pass
    else:
        raise TypeError('Filename is supposed to be a string or pathlib.Path')

    if not filename.exists():
        raise FileNotFoundError

    # ensure lowercase suffix
    suffix = filename.suffix.lower()

    if suffix in ('.dm3', '.dm4'):
        out = dmReader(filename)
    else:
        print('File suffix {} is not recognized.'.format(suffix))
        print('Supported formats are dm3, dm4.')

    return out