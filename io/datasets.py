import numpy as np

import os
from sklearn.utils import Bunch
from .io import load_pickle

def load_mose2_40KeV():
    data_dir = os.path.dirname(__file__) + '\data'
    ds = load_pickle(data_dir+'\dataset_mose2_40KeV.pkl')
    return ds

def load_mose2_2H3R():
    data_dir = os.path.dirname(__file__) + '\data'
    ds = load_pickle(data_dir + '\dataset_bilayer_mose2_2H3R.pkl')
    return ds

def load_mose2_clean():
    data_dir = os.path.dirname(__file__) + '\data'
    ds = load_pickle(data_dir + '\dataset_mose2_clean_80KeV.pkl')
    return ds

def load_MoSe2_clean():
    data_dir = os.path.dirname(__file__) + '\data'
    X = load_pickle(data_dir + '\dataset_MoSe2_clean_sklearn.pkl')
    data, target = X[:, -2], X[:, -1]
    return Bunch(data=data, target=target)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Generate synthetic datasets
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def make_blobs(pts, size=128, sigma=7):
    data = np.zeros((size, size))
    Y, X = np.ogrid[-size // 2:size // 2:1j * size, -size // 2:size // 2:1j * size]
    for (x, y) in pts:
        data = data + np.exp((-(X - x) ** 2 - (Y - y) ** 2) / (2 * sigma * sigma))
    return data