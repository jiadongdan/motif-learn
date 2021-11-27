import numpy as np

def disk(nx=256, r=0.3):
    t = np.linspace(-1, 1, nx)
    x, y = np.meshgrid(t, t)
    disk = (x*x + y*y<=r)*1.0
    return disk

def gauss2d(nx, sigma):
    pass


