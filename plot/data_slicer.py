import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_array

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# DataSlicer to display data
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class DataSlicer(object):
    def __init__(self, ax, data, **kwargs):
        self.ax = ax
        self.data = data
        if len(data.shape) == 3:
            self.plot_type = 'image'
            self.artist = self.ax.imshow(self.data[0], **kwargs)
        elif len(data.shape) == 2:
            self.plot_type = 'lines'
            self.artist = self.ax.plot(self.data[0], **kwargs)[0]
        else:
            raise ValueError('The length of data shape must be 2 or 3.')
        self.num_slices = data.shape[0]
        self.ind = 0
        self.ax.set_xlabel('slice {}'.format(self.ind), fontsize=14)
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.press_key)

    def press_key(self, event):
        if event.key == 'right':
            self.ind = (self.ind + 1) % self.num_slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.num_slices
        if self.plot_type == 'image':
            self.artist.set_data(self.data[self.ind])
        elif self.plot_type == 'lines':
            vmin, vmax = self.data.min(), self.data.max()
            self.ax.set_ylim(vmin, vmax)
            x = range(len(self.data[self.ind]))
            self.artist.set_data(x, self.data[self.ind])
        self.ax.set_xlabel('slice {}'.format(self.ind), fontsize=14)
        self.ax.figure.canvas.draw()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# imshow implemented via DataSlicer
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def imshow(data, flat=False, **kwargs):
    """

    """
    data = check_array(data, allow_nd=True)
    if len(data.shape) == 3:
        if flat:
            n = int(np.sqrt(np.minimum(len(data), 49)))+1
            fig, axes = plt.subplots(n, n, figsize=(7.2, 7.2))
            for i, (ax, img) in enumerate(zip(axes.ravel(), data)):
                ax.imshow(img, **kwargs)
                ax.axis('off')
                ax.text(0, 1.05, s=i, transform=ax.transAxes)
            fig.tight_layout()
            ds = None
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
            if 'vmin' not in kwargs:
                kwargs['vmin'] = data.min()
            if 'vmax' not in kwargs:
                kwargs['vmax'] = data.max()
            ds = DataSlicer(ax, data, **kwargs)

    elif len(data.shape) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ds = ax.imshow(data, **kwargs)
    else:
        raise ValueError('The length of data shape must be 2 or 3.')
    return ds


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# plot implemented via DataSlicer
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def plot(x, **kwargs):
    """

    """
    x = np.array(x)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    if len(x.shape) == 2:
        ds = DataSlicer(ax, x, *kwargs)
    elif len(x.shape) == 1:
        ds = ax.plot(x, **kwargs)
    else:
        raise ValueError('The length of data shape must be 2 or 3.')
    return ds

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# plot zernike moments
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def plot_zm(moments, states, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2/2))
    if not np.iterable(states):
        if states < 0:
            uniq = np.unique(np.abs(moments.m))
            states = uniq[np.nonzero(uniq)]
        else:
            states = [states]
    # plot symmetry indicators
    for ii, state in enumerate(states):
        ind = np.where(np.in1d(np.abs(moments.m), state))[0]
        for e in ind:
            ax.axvline(e, alpha=0.7, color='C{}'.format(ii+1), lw=2, ls='--')

    if len(moments.shape) == 2:
        ds = DataSlicer(ax, moments, *kwargs)
    elif len(moments.shape) == 1:
        ds = ax.plot(moments, **kwargs)
    else:
        raise ValueError('The length of data shape must be 2 or 3.')
    return ds

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Visualize scatter points
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
class ScatterTracker(object):
    def __init__(self, ax, data, **kwargs):
        self.ax = ax
        self.data = data
        self.slices = len(data)
        self.ind = 0

        self.path_collection = self.ax.scatter(self.data[0][:, 0], data[0][:, 1], **kwargs)
        self.ax.set_xlabel('slice %s' % 0)
        # Connect event functiont (process_key) to event manager
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.process_key)

    def process_key(self, event):
        if event.key == 'right':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.slices
        self.path_collection.set_offsets(self.data[self.ind])
        self.ax.set_xlabel('slice %s' % self.ind)
        # re-draw the image
        self.ax.figure.canvas.draw()

def plot_scatters(data, **kwargs):
    """
    """
    xmin = np.min([e[:, 0].min() for e in data])
    ymin = np.min([e[:, 1].min() for e in data])
    xmax = np.max([e[:, 0].max() for e in data])
    ymax = np.max([e[:, 1].max() for e in data])

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    tracker = ScatterTracker(ax, data, **kwargs)
    return tracker