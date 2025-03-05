import numpy as np
import matplotlib.pyplot as plt
from ..io._io import load_dm

class EELSData:

    def __init__(self, filename):
        eels_data_dict = load_dm(filename)

        self.data = eels_data_dict['data']
        self.shape = self.data.shape
        self.units = eels_data_dict['pixelUnit']
        idx = self.units.index('eV')
        self.energy = eels_data_dict['coords'][idx]

        self.mean_spectrum = self.data.mean(axis=(1, 2))

        self.bg = None

    def remove_bg(self, method='snip'):
        pass

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.plot(self.energy, self.mean_spectrum)
        idx1 = np.random.choice(self.shape[1])
        idx2 = np.random.choice(self.shape[2])
        # ax.plot(self.energy, self.data[:, idx1, idx2])