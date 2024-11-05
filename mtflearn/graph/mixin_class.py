import pathlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import connected_components

def is_array_like(a):
    return isinstance(a, (list, tuple, range, np.ndarray))


def extract_file_extension(file_name):
    ext = pathlib.Path(file_name).suffix[1:]
    return ext


def remove_file_extension(file_name):
    file_name_ = pathlib.Path(file_name).stem
    return file_name_

def check_array_like(*args):
    status = np.array([is_array_like(arg) for arg in args])
    return status

class SaveGraphMixin:

    def save(self, file_name, path=None):
        file_name = remove_file_extension(file_name)
        if path is not None:
            file_name = path + file_name
        # keys and data
        names = np.array(['img', 'nodes', 'edges', 'regions',])
        data_tuple = (self.img, self.nodes, self.edges, self.regions)
        mask = check_array_like(*data_tuple)
        # we need atleast one 'True' to proceed
        if not np.any(mask):
            raise ValueError('no valid inputs provided')

        names_ = names[mask]
        data_ = np.array(data_tuple, dtype=object)[mask]
        # create HDF5 file
        with h5py.File(file_name + '.hdf5', "w") as f:
            # create dataset
            for name, ds in zip(names_, data_):
                if name == 'regions':
                    regions_g = f.create_group('regions')
                    # get unique k
                    ks = np.array([len(region) for region in self.regions])
                    for k in np.unique(ks):
                        #g = regions_g.create_group(str(k))
                        data = np.vstack(ds[ks==k])
                        regions_g.create_dataset(str(k), data.shape, data.dtype, data)
                else:
                    data = np.array(ds)
                    dset = f.create_dataset(name, data.shape, data.dtype, data)


class ShowGraphMixin:

    def show(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        if 'color' not in kwargs:
            kwargs['color'] = '#2d3742ff'
        lines = np.array([(self.pts[i], self.pts[j]) for (i, j) in self.edges])
        segs = LineCollection(lines, **kwargs)
        ax.add_collection(segs)
        ax.scatter(self.pts[:, 0], self.pts[:, 1], alpha=0)
        ax.axis('equal')

    def show_regions(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        for region in self.regions:
            if len(region) <= 9:
                fc = 'C{}'.format(len(region)-3)
                poly = plt.Polygon(self.pts[region], fc=fc, ec='#2d3742', alpha=0.5)
                ax.add_patch(poly)
        ax.scatter(self.pts[:, 0], self.pts[:, 1], alpha=0)
        ax.axis('equal')
