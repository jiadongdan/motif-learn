import os
from ..io.io_image import load_pickle

__all__ = ['halide',
           'ice',
           'deep_sea',
           'avocado',
           'temperature',
           'orange',
           'parula']

data_dir = os.path.dirname(__file__) + '/colors/'

halide = load_pickle(data_dir+'cm_halide.pkl')
ice = load_pickle(data_dir+'cm_ice.pkl')

deep_sea = load_pickle(data_dir+'cm_deep_sea.pkl')
avocado = load_pickle(data_dir+'cm_avocado.pkl')
temperature = load_pickle(data_dir+'cm_temperature.pkl')
orange = load_pickle(data_dir+'cm_orange.pkl')
parula = load_pickle(data_dir+'LinearSegmentedColormap_parula.pkl')
