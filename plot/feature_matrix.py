import numpy as np
from matplotlib.colors import to_rgba
from skimage.transform import rescale


def add_grids(rgba, s=20):
    h, w = rgba.shape[0:2]

    ys = np.arange(0, h, 20)[1:]
    xs = np.arange(0, w, 20)[1:]

    d = 1
    for y in ys:
        rgba[y - d:y + d, ::] = [255, 255, 255, 255]

    for x in xs:
        rgba[:, x - d:x + d, :] = [255, 255, 255, 255]

    return rgba

def data2rgba(data, colors):
    # self.pxiels is rgba representation
    data_256 = rescale(data, scale=20, order=0).astype(int)

    rgba = (colors[data_256] * 255).astype(np.uint8)
    # add grids
    rgba = add_grids(rgba)
    return rgba


class FeatureMatrix:

    def __init__(self, nrows=10, ncols=10, rows_hide=None, cols_hide=None, colors=None):

        if colors is None:
            self.colors = ['#cee4cc'] + ['C{}'.format(i) for i in np.arange(10)] + ['white']
            self.colors = np.array([to_rgba(c) for c in self.colors])

        self.shape = (nrows, ncols)
        self.data =np.zeros(self.shape)


    def set_row_color(self, ind, lbs):
        self.data[ind, :] = lbs

    @property
    def image(self):
        return data2rgba(self.data, self.colors)

    def plot(self, ax=None):
        pass

