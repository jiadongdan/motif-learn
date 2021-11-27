from collections import namedtuple
import os
from ..io import load_image
from ..io import load_pickle


Dataset = namedtuple('Dataset', ['image', 'pts', 'features'])


def load_mose2():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir+'/MoSe2_80K_large_pts.pkl')
    features = load_pickle(data_dir + '/MoSe2_80K_large_features.pkl')
    image = load_image(data_dir+'/MoSe2_80K_large.tif')
    return Dataset(image=image, pts=pts, features=features)

def load_gb():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir+'/WS2Te2-2_0040_1_Cut_kp.pkl')
    features = load_pickle(data_dir + '/WS2Te2-2_0040_1_Cut_features.pkl')
    image = load_image(data_dir+'/WS2Te2-2_0040_1_Cut.tif')
    return Dataset(image=image, pts=pts, features=features)

def load_small_mose2():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/MoSe2_80K_small_pts.pkl')
    features = load_pickle(data_dir + '/MoSe2_80K_small_features.pkl')
    image = load_pickle(data_dir + '/MoSe2_80K_small_img.pkl')
    return Dataset(image=image, pts=pts, features=features)


def load_small_mose2_clean():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/MoSe2_80K_small_pts_clean.pkl')
    features = load_pickle(data_dir + '/MoSe2_80K_small_features_clean.pkl')
    image = load_pickle(data_dir + '/MoSe2_80K_small_img_clean.pkl')
    return Dataset(image=image, pts=pts, features=features)

def load_mose2_40KeV():
    data_dir = os.path.dirname(__file__) + '/data'
    ds = load_pickle('dataset_mose2_40KeV.pkl')
    return ds

def load_gb_momo_mose2():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/gb_mono_MoSe2_pts.pkl')
    features = None
    image = load_pickle(data_dir + '/gb_mono_MoSe2_image.pkl')
    return Dataset(image=image, pts=pts, features=features)

def load_gb_2H3R():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/gb_2H3R_MoSe2_pts.pkl')
    features = None
    image = load_pickle(data_dir + '/gb_2H3R_MoSe2_image.pkl')
    return Dataset(image=image, pts=pts, features=features)

def load_bilayer_mos2_3_deg():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/bilayer_mos2_3_deg_pts.pkl')
    features = None
    image = load_pickle(data_dir + '/bilayer_mos2_3_deg_image.pkl')
    return Dataset(image=image, pts=pts, features=features)

def load_qian_pentagon():
    data_dir = os.path.dirname(__file__) + '/data'
    pts = load_pickle(data_dir + '/qian_pentagon_pts.pkl')
    features = None
    image = load_pickle(data_dir + '/qian_pentagon_image.pkl')
    return Dataset(image=image, pts=pts, features=features)