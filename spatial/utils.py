import numpy as np


def dist(pts):
    return np.hypot(pts[:, 0], pts[:, 1])


