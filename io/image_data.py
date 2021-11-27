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

def load_mose2_small():
    data_dir = os.path.dirname(__file__) + '\data'
    ds = load_pickle(data_dir + '\image_MoSe2_80K_samll_region.pkl')
    return ds


def load_MoSe2_clean():
    data_dir = os.path.dirname(__file__) + '\data'
    X = load_pickle(data_dir + '\dataset_MoSe2_clean_sklearn.pkl')
    data, target = X[:, 0:-2], X[:, -1]
    return Bunch(data=data, target=target)
