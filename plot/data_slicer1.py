import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_array

class DataSlicer(object):
    def __init__(self, ax, data, **kwargs):
        self.ax = ax
        self.data = data
        if isinstance(data, list):
            self.plot_type = 'points'
            points_stack = np.stack(self.data)
            vmax = points_stack.max()
            vmin = points_stack.min()
            margin = points_stack.ptp()*0.05

            self.artist = self.ax.scatter(self.data[0][:, 0], self.data[0][:, 1], **kwargs)
            ax.set_xlim(vmin-margin, vmax+margin)
            ax.set_ylim(vmin-margin, vmax+margin)
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 3:
                self.plot_type = 'images'
                self.artist = self.ax.imshow(self.data[0], **kwargs)
            elif len(data.shape) == 2:
                self.plot_type = 'lines'
                self.artist = self.ax.plot(self.data[0], **kwargs)[0]
        else:
            raise ValueError('Invalid data format.')
        self.num_slices = len(data)
        self.ind = 0
        self.ax.set_xlabel('slice {}'.format(self.ind), fontsize=14)
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.press_key)

    def press_key(self, event):
        if event.key == 'right':
            self.ind = (self.ind + 1) % self.num_slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.num_slices
        # update data according to plot_types
        if self.plot_type == 'images':
            self.artist.set_data(self.data[self.ind])
        elif self.plot_type == 'lines':
            vmin, vmax = self.data.min(), self.data.max()
            self.ax.set_ylim(vmin, vmax)
            x = range(len(self.data[self.ind]))
            self.artist.set_data(x, self.data[self.ind])
        elif self.plot_type == 'points':
            self.artist.set_offsets(self.data[self.ind])
        # update slice number
        self.ax.set_xlabel('slice {}'.format(self.ind), fontsize=14)
        self.ax.figure.canvas.draw()

class DataCursor:

    def __init__(self, artist):
        pass


def imshow(imgs, ax=None, hvlines=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    imgs = check_array(imgs, allow_nd=True)
    shape = imgs.shape
    if len(shape) == 3:
        # single image
        if shape[0] == 1:
            h, w = shape[1:3]
            im = ax.imshow(imgs[0], **kwargs)
        # color image (we don't show images with size <= 3)
        elif shape[2] == 3:
            h, w = shape[0:2]
            im = ax.imshow(imgs, **kwargs)
        else:
            h, w = shape[1:3]
            if 'vmin' not in kwargs:
                kwargs['vmin'] = imgs.min()
            if 'vmax' not in kwargs:
                kwargs['vmax'] = imgs.max()
            im = DataSlicer(ax, imgs, **kwargs)
    elif len(shape) == 2:
        h, w = shape[0:2]
        im = ax.imshow(imgs, **kwargs)
    else:
        raise ValueError('The length of data shape must be 2 or 3.')
    if hvlines:
        ax.axhline(h/2, color='r')
        ax.axvline(w/2, color='r')
    return im


def plot(lines, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    lines = np.array(lines)
    if len(lines.shape) == 2:
        ds = DataSlicer(ax, lines, **kwargs)
    elif len(lines.shape) == 1:
        ds = ax.plot(lines, **kwargs)
    else:
        raise ValueError('The length of data shape must be 2 or 3.')
    return ds


def scatter(points, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

    if isinstance(points, list):
        if len(points) == 1:
            sc = ax.scatter(points[0][:, 0], points[0][:, 1], **kwargs)
        else:
            sc = DataSlicer(ax, points, **kwargs)
    else:
        sc = ax.scatter(points[:, 0], points[:, 1], **kwargs)
    return sc
