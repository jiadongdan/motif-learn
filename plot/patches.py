import numpy as np
from matplotlib.patches import Polygon


def make_parallelogram(x, y, a, b, angle=60, **kwargs):
    t = np.deg2rad(angle)
    p4 = np.array([(0, 0), (a, 0), (a*np.cos(t)+a, b*np.sin(t)), (a*np.cos(t), b*np.sin(t))])
    p4[:, 0] += x
    p4[:, 1] += y

    parallelogram = Polygon(xy=p4, **kwargs)
    return parallelogram
