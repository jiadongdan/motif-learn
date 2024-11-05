import numpy as np
from skimage.io import imread
from pathlib import Path
from ._dm4 import DMfile
from ._dm_ncempy import dmReader


def normalize(image, low=0., high=1.):
    img_max = image.max()
    img_min = image.min()
    img_norm = (image - img_min) / (img_max - img_min) * (high - low) + low
    return img_norm


def load_image(file_name, normalized=False):
    file_extension = np.char.split(file_name, sep='.').tolist()[-1]
    file_extension = '.' + file_extension
    # file_extension = os.path.splitext(file_name)[1]
    if file_extension.lower() in ['.dm4', '.dm3']:
        img = DMfile(file_name).data
    else:
        img = imread(file_name)
    if normalized is True:
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