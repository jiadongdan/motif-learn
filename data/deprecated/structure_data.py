import os
from ..io import load_image


def gb_2h3r():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/GB_2H3R_MoSe2.tif'
    return load_image(file_name)

def mono_mose2():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/monolayer_MoSe2_80K.tif'
    return load_image(file_name)

def gb_mono_mose2():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/GB_mono_MoSe2.tif'
    return load_image(file_name)


